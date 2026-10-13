// The modern renderer (video.renderer = modern, the default): runs a recorded
// frame (gfx_core.cpp) at the resolution it is shown at, not the game's.
//
// - The back buffer has the pixels of the area the picture takes in the
//   window (times video.render_scale percent), multisampled (video.msaa,
//   4 by default). 3D draws get their viewport scaled up to it.
//   Pre-transformed draws (menus, text, HUD) keep the game's coordinates, so
//   they keep their size on screen and stay readable at any resolution.
// - Alpha-tested draws (foliage, fences, spectators) get anti-aliased edges
//   through alpha to coverage; textures are filtered anisotropically
//   (video.anisotropy, 8 by default).
// - Stage geometry stays on the GPU (gfx_core.cpp draws it by reference).
// - Shader data is split by how often it changes (transform, lighting,
//   texture coordinates, texture stages) and each block is pushed only when
//   it changed; pipelines, viewports, buffers and textures are bound only
//   when they change.
//
// The shaders compute what the classic renderer's do: the picture is the
// original's, only sharper.

#include "platform/platform.h"
#include "port/sys.h"
#include "render/gpu_context.h"
#include "render/presentation.h"

#include "shaders_spirv.h"

#ifdef OPENCMR2_SHADERS_MSL
#define SHADER_CODE(name) g_##name, sizeof(g_##name), g_##name##_msl, sizeof(g_##name##_msl)
#else
#define SHADER_CODE(name) g_##name, sizeof(g_##name), nullptr, 0
#endif

#include <math.h>
#include <string.h>

#include <algorithm>
#include <unordered_map>

namespace {

// ---- uniform blocks (std140, see shaders/modern_common.glsl, modern.frag) ----

struct LightUniform {
    float diffuse[4], specular[4], ambient[4], position[4], direction[4], attenuation[4], spot[4];
};

struct TransformUniforms {
    float worldView[16];
    float projection[16];
    float target[4];
};

struct LightingUniforms {
    float materialDiffuse[4], materialAmbient[4], materialSpecular[4], materialEmissive[4];
    float globalAmbient[4];
    LightUniform lights[8];
    int32_t lightFlags[4];
    int32_t sources[4];
    int32_t fogFlags[4];
    float fogParams[4];
};

struct TexGenUniforms {
    float textureMatrix[2][16];
    int32_t texCoordIndex[4];
};

struct CombinerUniforms {
    int32_t stage0[4], stage0b[4], stage1[4], stage1b[4];
    float textureFactor[4];
    int32_t alphaTest[4];
    float alphaRef[4];
    float fogColor[4];
    int32_t fogMode[4];
    float fogParams[4];
    float bumpMatrix[4];
    float bumpLuminance[4];
};

struct ClearUniforms {
    float color[4];
    float depth[4];
};

// A uniform block pushed only when its contents change.
template <typename T>
struct Pushed {
    T last;
    bool valid = false;
    // True when value differs from what was pushed last (and is now taken as pushed).
    bool Changed(const T &value)
    {
        if (valid && memcmp(&last, &value, sizeof(T)) == 0)
            return false;
        last = value;
        valid = true;
        return true;
    }
};

struct ModernExecutor : GfxExecutor {
    GpuContext gpu;
    SDL_GPUShader *vs3d = nullptr, *vsTL = nullptr, *fs = nullptr, *vsClear = nullptr, *fsClear = nullptr;
    // backColor: the picture shown, one sample per pixel; with multisampling
    // the passes draw to backMsaa and resolve into it.
    SDL_GPUTexture *backColor = nullptr, *backMsaa = nullptr, *backDepth = nullptr;
    int backWidth = 0, backHeight = 0;          // the game's resolution
    int renderWidth = 0, renderHeight = 0;      // pixels of the back buffer
    SDL_GPUSampleCount backSamples = SDL_GPU_SAMPLECOUNT_1;
    SDL_GPUSampleCount wantedSamples = SDL_GPU_SAMPLECOUNT_1;   // video.msaa, as supported
    int renderScale = 100;                                      // video.render_scale
    std::unordered_map<uint32_t, SDL_GPUGraphicsPipeline *> pipelines;

    ~ModernExecutor() override
    {
        if (gpu.device == NULL)
            return;
        SDL_WaitForGPUIdle(gpu.device);
        for (auto &p : pipelines)
            SDL_ReleaseGPUGraphicsPipeline(gpu.device, p.second);
        SDL_ReleaseGPUTexture(gpu.device, backColor);
        SDL_ReleaseGPUTexture(gpu.device, backMsaa);
        SDL_ReleaseGPUTexture(gpu.device, backDepth);
        SDL_ReleaseGPUShader(gpu.device, vs3d);
        SDL_ReleaseGPUShader(gpu.device, vsTL);
        SDL_ReleaseGPUShader(gpu.device, fs);
        SDL_ReleaseGPUShader(gpu.device, vsClear);
        SDL_ReleaseGPUShader(gpu.device, fsClear);
    }

    bool CreateDevice()
    {
        if (!gpu.Create(Platform_GetSettingInt("video.anisotropy", 8)))
            return false;
        Sys_Log("gfx: modern renderer");
        wantedSamples = BackSamples();
        renderScale = std::max(25, std::min(400, Platform_GetSettingInt("video.render_scale", 100)));
        vs3d = gpu.Shader(SHADER_CODE(modern3d_vert), SDL_GPU_SHADERSTAGE_VERTEX, 0, 3);
        vsTL = gpu.Shader(SHADER_CODE(moderntl_vert), SDL_GPU_SHADERSTAGE_VERTEX, 0, 3);
        fs = gpu.Shader(SHADER_CODE(modern_frag), SDL_GPU_SHADERSTAGE_FRAGMENT, 4, 1);
        vsClear = gpu.Shader(SHADER_CODE(clear_vert), SDL_GPU_SHADERSTAGE_VERTEX, 0, 1);
        fsClear = gpu.Shader(SHADER_CODE(clear_frag), SDL_GPU_SHADERSTAGE_FRAGMENT, 0, 1);
        return vs3d && vsTL && fs && vsClear && fsClear;
    }

    bool SetVideoMode(int width, int height, bool fullscreen) override
    {
        if (gpu.device == NULL && !CreateDevice())
            return false;
        backWidth = width;
        backHeight = height;
        SDL_SetWindowResizable(gpu.window, true);
        SetFullscreen(fullscreen);
        SDL_ShowWindow(gpu.window);
        return EnsureBackBuffer();
    }

    // Fullscreen on the desktop resolution, or a centred window of the
    // game's resolution times video.window_scale.
    void SetFullscreen(bool fullscreen) override
    {
        SDL_Window *window = gpu.window;
        if (window == NULL)
            return;
        bool wasFullscreen = (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) != 0;
        SDL_SetWindowFullscreen(window, fullscreen);
        if (fullscreen)
            return;
        if (wasFullscreen)
            SDL_SyncWindow(window);
        int scale = std::max(1, Platform_GetSettingInt("video.window_scale", 1));
        SDL_SetWindowSize(window, backWidth * scale, backHeight * scale);
        SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    }

    void DestroyTexture(GfxTexture *t) override { gpu.DestroyTexture(t); }
    void DestroyVertexBuffer(GfxVertexBuffer *b) override { gpu.DestroyVertexBuffer(b); }
    void ReadTexture(GfxTexture *t) override { gpu.ReadTexture(t); }

    // ---- back buffer ----

    // The most samples up to video.msaa that the GPU supports for both back
    // buffer formats.
    SDL_GPUSampleCount BackSamples()
    {
        int wanted = Platform_GetSettingInt("video.msaa", 4);
        const struct { int n; SDL_GPUSampleCount count; } counts[] = {
            { 8, SDL_GPU_SAMPLECOUNT_8 }, { 4, SDL_GPU_SAMPLECOUNT_4 }, { 2, SDL_GPU_SAMPLECOUNT_2 } };
        for (const auto &c : counts)
            if (c.n <= wanted && SDL_GPUTextureSupportsSampleCount(gpu.device, gpu.colorFormat, c.count) &&
                SDL_GPUTextureSupportsSampleCount(gpu.device, gpu.depthFormat, c.count))
                return c.count;
        return SDL_GPU_SAMPLECOUNT_1;
    }

    // The area the picture takes in the window, times video.render_scale.
    void RenderSize(int &w, int &h)
    {
        w = backWidth;
        h = backHeight;
        int ww = 0, wh = 0;
        if (!SDL_GetWindowSizeInPixels(gpu.window, &ww, &wh) || ww <= 0 || wh <= 0)
            return;
        PresentationRect rect = GetPresentationRect(backWidth, backHeight, ww, wh, false);
        w = std::max(1, rect.width * renderScale / 100);
        h = std::max(1, rect.height * renderScale / 100);
    }

    // (Re)creates the back buffer when the window, the mode or the settings changed.
    bool EnsureBackBuffer()
    {
        int w, h;
        RenderSize(w, h);
        SDL_GPUSampleCount samples = wantedSamples;
        if (backColor != NULL && renderWidth == w && renderHeight == h && backSamples == samples)
            return true;
        SDL_WaitForGPUIdle(gpu.device);
        SDL_ReleaseGPUTexture(gpu.device, backColor);
        SDL_ReleaseGPUTexture(gpu.device, backMsaa);
        SDL_ReleaseGPUTexture(gpu.device, backDepth);
        backMsaa = NULL;
        backColor = gpu.CreateTexture(SDL_GPU_TEXTURETYPE_2D, gpu.colorFormat, w, h, 1, 1,
                                      SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
        if (samples != SDL_GPU_SAMPLECOUNT_1)
            backMsaa = gpu.CreateTexture(SDL_GPU_TEXTURETYPE_2D, gpu.colorFormat, w, h, 1, 1,
                                         SDL_GPU_TEXTUREUSAGE_COLOR_TARGET, samples);
        backDepth = gpu.CreateTexture(SDL_GPU_TEXTURETYPE_2D, gpu.depthFormat, w, h, 1, 1,
                                      SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET, samples);
        renderWidth = w;
        renderHeight = h;
        backSamples = samples;
        if (backColor == NULL || backDepth == NULL || (samples != SDL_GPU_SAMPLECOUNT_1 && backMsaa == NULL))
            return false;
        Sys_Log("gfx: %dx%d drawn at %dx%d, %d samples", backWidth, backHeight, w, h,
                samples == SDL_GPU_SAMPLECOUNT_8 ? 8 : samples == SDL_GPU_SAMPLECOUNT_4 ? 4
                : samples == SDL_GPU_SAMPLECOUNT_2 ? 2 : 1);
        return true;
    }

    // ---- pipelines ----

    // Alpha-tested draws that do not blend get anti-aliased edges from their
    // alpha when multisampling.
    static bool AlphaToCoverage(const GfxDrawState &s, SDL_GPUSampleCount samples)
    {
        return samples != SDL_GPU_SAMPLECOUNT_1 && s.alphaTest && !s.alphaBlend;
    }

    SDL_GPUGraphicsPipeline *Pipeline(const GfxCommand &cmd, const GfxDrawState &s, SDL_GPUSampleCount samples)
    {
        bool depthTest = s.depthTest;
        bool coverage = AlphaToCoverage(s, samples);
        uint32_t key = (cmd.transformed ? 1u : 0u) | (uint32_t)(cmd.primitive & 7) << 1 |
                       (s.alphaBlend ? 1u : 0u) << 4 | (uint32_t)(s.srcBlend & 15) << 5 |
                       (uint32_t)(s.destBlend & 15) << 9 | (depthTest ? 1u : 0u) << 13 |
                       ((depthTest && s.depthWrite) ? 1u : 0u) << 14 | (uint32_t)(s.depthFunc & 15) << 15 |
                       (uint32_t)(s.cull & 3) << 19 | (uint32_t)(samples & 3) << 21 | (coverage ? 1u : 0u) << 23;
        auto it = pipelines.find(key);
        if (it != pipelines.end())
            return it->second;

        SDL_GPUVertexBufferDescription vb = {};
        vb.slot = 0;
        vb.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
        SDL_GPUVertexAttribute attrs[6] = {};
        int attrCount;
        if (cmd.transformed) {
            vb.pitch = sizeof(GfxTLVertex);
            attrs[0] = { 0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, 0 };
            attrs[1] = { 1, 0, SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM, 16 };
            attrs[2] = { 2, 0, SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM, 20 };
            attrs[3] = { 3, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, 24 };
            attrCount = 4;
        } else {
            vb.pitch = sizeof(GfxVertex);
            attrs[0] = { 0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0 };
            attrs[1] = { 1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 12 };
            attrs[2] = { 2, 0, SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM, 24 };
            attrs[3] = { 3, 0, SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM, 28 };
            attrs[4] = { 4, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, 32 };
            attrs[5] = { 5, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, 40 };
            attrCount = 6;
        }

        SDL_GPUColorTargetDescription color = {};
        color.format = gpu.colorFormat;
        color.blend_state.enable_blend = s.alphaBlend;
        color.blend_state.src_color_blendfactor = Gpu_BlendFactor(s.srcBlend);
        color.blend_state.dst_color_blendfactor = Gpu_BlendFactor(s.destBlend);
        color.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
        color.blend_state.src_alpha_blendfactor = Gpu_BlendFactor(s.srcBlend);
        color.blend_state.dst_alpha_blendfactor = Gpu_BlendFactor(s.destBlend);
        color.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;

        SDL_GPUGraphicsPipelineCreateInfo info = {};
        info.vertex_shader = cmd.transformed ? vsTL : vs3d;
        info.fragment_shader = fs;
        info.vertex_input_state.vertex_buffer_descriptions = &vb;
        info.vertex_input_state.num_vertex_buffers = 1;
        info.vertex_input_state.vertex_attributes = attrs;
        info.vertex_input_state.num_vertex_attributes = attrCount;
        info.primitive_type = Gpu_PrimitiveType(cmd.primitive);
        info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
        info.rasterizer_state.front_face = SDL_GPU_FRONTFACE_CLOCKWISE;
        info.rasterizer_state.cull_mode = s.cull == GFX_CULL_CW    ? SDL_GPU_CULLMODE_FRONT
                                          : s.cull == GFX_CULL_CCW ? SDL_GPU_CULLMODE_BACK
                                                                   : SDL_GPU_CULLMODE_NONE;
        info.rasterizer_state.enable_depth_clip = true;
        info.depth_stencil_state.enable_depth_test = depthTest;
        info.depth_stencil_state.enable_depth_write = depthTest && s.depthWrite;
        info.depth_stencil_state.compare_op = Gpu_CompareOp(s.depthFunc);
        info.multisample_state.sample_count = samples;
        info.multisample_state.enable_alpha_to_coverage = coverage;
        info.target_info.color_target_descriptions = &color;
        info.target_info.num_color_targets = 1;
        info.target_info.depth_stencil_format = gpu.depthFormat;
        info.target_info.has_depth_stencil_target = true;
        SDL_GPUGraphicsPipeline *p = SDL_CreateGPUGraphicsPipeline(gpu.device, &info);
        if (p == NULL)
            Sys_Log("gfx: pipeline: %s", SDL_GetError());
        pipelines[key] = p;
        return p;
    }

    SDL_GPUGraphicsPipeline *ClearPipeline(bool color, bool depth, SDL_GPUSampleCount samples)
    {
        uint32_t key = 0x80000000u | (color ? 1u : 0u) | (depth ? 2u : 0u) | (uint32_t)(samples & 3) << 2;
        auto it = pipelines.find(key);
        if (it != pipelines.end())
            return it->second;
        SDL_GPUColorTargetDescription target = {};
        target.format = gpu.colorFormat;
        target.blend_state.enable_color_write_mask = true;
        target.blend_state.color_write_mask = color ? 0xf : 0;
        SDL_GPUGraphicsPipelineCreateInfo info = {};
        info.vertex_shader = vsClear;
        info.fragment_shader = fsClear;
        info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
        info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
        info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
        info.depth_stencil_state.enable_depth_test = depth;
        info.depth_stencil_state.enable_depth_write = depth;
        info.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_ALWAYS;
        info.multisample_state.sample_count = samples;
        info.target_info.color_target_descriptions = &target;
        info.target_info.num_color_targets = 1;
        info.target_info.depth_stencil_format = gpu.depthFormat;
        info.target_info.has_depth_stencil_target = true;
        SDL_GPUGraphicsPipeline *p = SDL_CreateGPUGraphicsPipeline(gpu.device, &info);
        if (p == NULL)
            Sys_Log("gfx: clear pipeline: %s", SDL_GetError());
        pipelines[key] = p;
        return p;
    }

    // ---- uniforms ----

    static void FillTransform(const GfxDrawState &s, int gameW, int gameH, TransformUniforms &u)
    {
        Gpu_Multiply(s.world, s.view, u.worldView);
        memcpy(u.projection, &s.projection, sizeof(u.projection));
        u.target[0] = (float)gameW;
        u.target[1] = (float)gameH;
        // Direct3D 7's half-pixel offset of the viewport transform.
        u.target[2] = s.viewport.width ? 1.0f / s.viewport.width : 0.0f;
        u.target[3] = s.viewport.height ? -1.0f / s.viewport.height : 0.0f;
    }

    // Whether the state's lighting inputs equal the previous one's (the
    // lighting block, with its per-light work, is then the same).
    static bool SameLighting(const GfxDrawState &a, const GfxDrawState &b)
    {
        return memcmp(&a.view, &b.view, sizeof(a.view)) == 0 && memcmp(a.lights, b.lights, sizeof(a.lights)) == 0 &&
               memcmp(a.lightEnabled, b.lightEnabled, sizeof(a.lightEnabled)) == 0 &&
               memcmp(&a.material, &b.material, sizeof(a.material)) == 0 && a.lighting == b.lighting &&
               a.ambient == b.ambient && a.specular == b.specular && a.localViewer == b.localViewer &&
               a.normalizeNormals == b.normalizeNormals && a.colorVertex == b.colorVertex &&
               a.diffuseSource == b.diffuseSource && a.specularSource == b.specularSource &&
               a.ambientSource == b.ambientSource && a.emissiveSource == b.emissiveSource && a.fog == b.fog &&
               a.fogTableMode == b.fogTableMode && a.fogVertexMode == b.fogVertexMode &&
               a.fogStart == b.fogStart && a.fogEnd == b.fogEnd;
    }

    static void FillLighting(const GfxDrawState &s, LightingUniforms &u)
    {
        memset(&u, 0, sizeof(u));
        Gpu_Copy4(s.material.diffuse, u.materialDiffuse);
        Gpu_Copy4(s.material.ambient, u.materialAmbient);
        Gpu_Copy4(s.material.specular, u.materialSpecular);
        Gpu_Copy4(s.material.emissive, u.materialEmissive);
        Gpu_Color(s.ambient, u.globalAmbient);
        u.globalAmbient[3] = s.material.power;
        for (int i = 0; i < 8; i++) {
            if (!s.lightEnabled[i])
                continue;   // the shader skips disabled lights (spot.z = 0)
            const GfxLight &l = s.lights[i];
            LightUniform &lu = u.lights[i];
            Gpu_Copy4(l.diffuse, lu.diffuse);
            Gpu_Copy4(l.specular, lu.specular);
            Gpu_Copy4(l.ambient, lu.ambient);
            Gpu_TransformPoint(l.position, 1.0f, s.view, lu.position);
            lu.position[3] = (float)l.type;
            Gpu_TransformPoint(l.direction, 0.0f, s.view, lu.direction);
            lu.direction[3] = l.range > 0.0f ? l.range : 1e30f;
            lu.attenuation[0] = l.attenuation0;
            lu.attenuation[1] = l.attenuation1;
            lu.attenuation[2] = l.attenuation2;
            lu.attenuation[3] = l.falloff;
            lu.spot[0] = cosf(l.theta * 0.5f);
            lu.spot[1] = cosf(l.phi * 0.5f);
            lu.spot[2] = 1.0f;
        }
        u.lightFlags[0] = s.lighting;
        u.lightFlags[1] = s.colorVertex;
        u.lightFlags[2] = s.specular;
        u.lightFlags[3] = s.localViewer;
        u.sources[0] = s.diffuseSource;
        u.sources[1] = s.specularSource;
        u.sources[2] = s.ambientSource;
        u.sources[3] = s.emissiveSource;
        u.fogFlags[0] = (s.fog && s.fogTableMode == GFX_FOG_NONE) ? s.fogVertexMode : 0;
        u.fogFlags[1] = s.normalizeNormals;
        u.fogParams[0] = s.fogStart;
        u.fogParams[1] = s.fogEnd;
    }

    static void FillTexGen(const GfxDrawState &s, TexGenUniforms &u)
    {
        memcpy(u.textureMatrix[0], &s.texture[0], 64);
        memcpy(u.textureMatrix[1], &s.texture[1], 64);
        u.texCoordIndex[0] = (int32_t)s.stages[0].texCoordIndex;
        u.texCoordIndex[1] = (int32_t)s.stages[1].texCoordIndex;
        u.texCoordIndex[2] = s.stages[0].textureTransform;
        u.texCoordIndex[3] = s.stages[1].textureTransform;
    }

    static void FillCombiner(const GfxDrawState &s, SDL_GPUSampleCount samples, CombinerUniforms &u)
    {
        memset(&u, 0, sizeof(u));
        const GfxStageState &a = s.stages[0], &b = s.stages[1];
        u.stage0[0] = a.colorOp;
        u.stage0[1] = a.colorArg1;
        u.stage0[2] = a.colorArg2;
        u.stage0[3] = a.alphaOp;
        u.stage0b[0] = a.alphaArg1;
        u.stage0b[1] = a.alphaArg2;
        u.stage0b[2] = GpuContext::TextureKind(a);
        u.stage1[0] = b.colorOp;
        u.stage1[1] = b.colorArg1;
        u.stage1[2] = b.colorArg2;
        u.stage1[3] = b.alphaOp;
        u.stage1b[0] = b.alphaArg1;
        u.stage1b[1] = b.alphaArg2;
        u.stage1b[2] = GpuContext::TextureKind(b);
        Gpu_Color(s.textureFactor, u.textureFactor);
        u.alphaTest[0] = s.alphaTest;
        u.alphaTest[1] = s.alphaFunc;
        u.alphaTest[2] = AlphaToCoverage(s, samples);
        u.alphaRef[0] = (s.alphaRef & 0xff) / 255.0f;
        Gpu_Color(s.fogColor, u.fogColor);
        u.fogMode[0] = s.fog;
        u.fogMode[1] = s.fogTableMode;
        u.fogMode[2] = s.specular;
        u.fogParams[0] = s.fogStart;
        u.fogParams[1] = s.fogEnd;
        memcpy(u.bumpMatrix, a.bumpMat, sizeof(u.bumpMatrix));
        u.bumpLuminance[0] = a.bumpLumScale;
        u.bumpLuminance[1] = a.bumpLumOffset;
    }

    // ---- frame execution ----

    // Begins a pass on a render target texture or the back buffer: w, h are
    // its pixels, gameW, gameH the size the game draws it at.
    SDL_GPURenderPass *BeginPass(SDL_GPUCommandBuffer *cmd, GfxTexture *target, int &w, int &h, int &gameW,
                                 int &gameH, SDL_GPUSampleCount &samples)
    {
        SDL_GPUColorTargetInfo color = {};
        SDL_GPUDepthStencilTargetInfo depth = {};
        color.load_op = SDL_GPU_LOADOP_LOAD;
        color.store_op = SDL_GPU_STOREOP_STORE;
        depth.load_op = SDL_GPU_LOADOP_LOAD;
        depth.store_op = SDL_GPU_STOREOP_STORE;
        depth.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
        depth.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;
        if (target == NULL) {
            color.texture = backMsaa != NULL ? backMsaa : backColor;
            depth.texture = backDepth;
            w = renderWidth;
            h = renderHeight;
            gameW = backWidth;
            gameH = backHeight;
            samples = backSamples;
        } else {
            GfxTexture *root = target->cube != NULL ? target->cube : target;
            GpuTexture *g = gpu.Ensure(root);
            color.texture = g->texture;
            color.layer_or_depth_plane = target->cube != NULL ? target->face : 0;
            depth.texture = g->depth;
            w = gameW = root->width;
            h = gameH = root->height;
            samples = SDL_GPU_SAMPLECOUNT_1;
        }
        return SDL_BeginGPURenderPass(cmd, &color, 1, &depth);
    }

    void Execute(GfxFrame &frame, std::vector<GfxTexture *> &textures, GfxReadback &readback, bool present,
                 bool vsync, bool stretch) override
    {
        if (gpu.device == NULL)
            return;
        // Follow the window: a resize changes the back buffer's pixels.
        if (present && !EnsureBackBuffer())
            return;
        SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(gpu.device);
        if (cmd == NULL) {
            Sys_Log("gfx: %s", SDL_GetError());
            return;
        }
        gpu.EnsureFrameTextures(frame);
        gpu.UploadTextures(cmd, textures);
        gpu.UploadVertexBuffers(cmd, frame);
        gpu.UploadGeometry(cmd, frame);

        // Uniform data stays with the command buffer across passes; bindings
        // do not outlive a pass.
        Pushed<TransformUniforms> transform;
        Pushed<LightingUniforms> lighting;
        Pushed<TexGenUniforms> texGen;
        Pushed<CombinerUniforms> combiner;
        const GfxDrawState *lightingState = NULL;
        LightingUniforms lightingValue;

        SDL_GPURenderPass *pass = NULL;
        GfxTexture *target = NULL;
        int targetW = renderWidth, targetH = renderHeight, gameW = backWidth, gameH = backHeight;
        SDL_GPUSampleCount samples = backSamples;
        SDL_GPUGraphicsPipeline *boundPipeline = NULL;
        SDL_GPUBuffer *boundVertices = NULL;
        uint32_t boundVertexOffset = 0;
        bool indicesBound = false;
        const GfxDrawState *boundTextures = NULL;
        SDL_Rect boundScissor = { -1, -1, -1, -1 };
        SDL_GPUViewport boundViewport = { -1, -1, -1, -1, -1, -1 };

        auto scaleX = [&](float x) { return x * targetW / gameW; };
        auto scaleY = [&](float y) { return y * targetH / gameH; };
        auto scaledRect = [&](const GfxViewport &v) {
            int x0 = (int)lroundf(scaleX((float)v.x)), y0 = (int)lroundf(scaleY((float)v.y));
            int x1 = (int)lroundf(scaleX((float)(v.x + v.width))), y1 = (int)lroundf(scaleY((float)(v.y + v.height)));
            return SDL_Rect{ x0, y0, x1 - x0, y1 - y0 };
        };
        auto setViewport = [&](const SDL_GPUViewport &vp, const SDL_Rect &scissor) {
            if (memcmp(&vp, &boundViewport, sizeof(vp)) != 0) {
                SDL_SetGPUViewport(pass, &vp);
                boundViewport = vp;
            }
            if (memcmp(&scissor, &boundScissor, sizeof(scissor)) != 0) {
                SDL_SetGPUScissor(pass, &scissor);
                boundScissor = scissor;
            }
        };

        for (const GfxCommand &c : frame.commands) {
            if (c.type == GFX_CMD_TARGET) {
                if (pass != NULL) {
                    SDL_EndGPURenderPass(pass);
                    pass = NULL;
                }
                target = c.target;
                continue;
            }
            if (pass == NULL) {
                pass = BeginPass(cmd, target, targetW, targetH, gameW, gameH, samples);
                boundPipeline = NULL;
                boundVertices = NULL;
                indicesBound = false;
                boundTextures = NULL;
                boundScissor = { -1, -1, -1, -1 };
                boundViewport = { -1, -1, -1, -1, -1, -1 };
                transform.valid = false;    // the target's size is part of it
            }
            if (c.type == GFX_CMD_CLEAR) {
                bool colorClear = (c.clearFlags & GFX_CLEAR_TARGET) != 0;
                bool depthClear = (c.clearFlags & GFX_CLEAR_ZBUFFER) != 0;
                if (!colorClear && !depthClear)
                    continue;
                SDL_GPUGraphicsPipeline *p = ClearPipeline(colorClear, depthClear, samples);
                SDL_BindGPUGraphicsPipeline(pass, p);
                boundPipeline = p;
                SDL_Rect scissor = scaledRect(c.clearViewport);
                setViewport({ (float)scissor.x, (float)scissor.y, (float)scissor.w, (float)scissor.h, 0.0f, 1.0f },
                            scissor);
                ClearUniforms u = {};
                Gpu_Color(c.clearColor, u.color);
                u.depth[0] = c.clearZ;
                // The clear shaders use slot 0 too: the draws' blocks must be pushed again.
                SDL_PushGPUVertexUniformData(cmd, 0, &u, sizeof(u));
                SDL_PushGPUFragmentUniformData(cmd, 0, &u, sizeof(u));
                transform.valid = false;
                combiner.valid = false;
                SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
                continue;
            }
            const GfxDrawState &s = frame.states[c.state];
            SDL_GPUGraphicsPipeline *pipeline = Pipeline(c, s, samples);
            if (pipeline == NULL)
                continue;
            if (pipeline != boundPipeline) {
                SDL_BindGPUGraphicsPipeline(pass, pipeline);
                boundPipeline = pipeline;
            }
            // 3D draws get the viewport in the target's pixels. Pre-transformed
            // ones cover the target and keep the game's coordinates (the
            // vertex shader maps them by the game's size), so they scale.
            SDL_Rect scissor = scaledRect(s.viewport);
            if (c.transformed)
                setViewport({ 0.0f, 0.0f, (float)targetW, (float)targetH, 0.0f, 1.0f }, scissor);
            else
                setViewport({ scaleX((float)s.viewport.x), scaleY((float)s.viewport.y),
                              scaleX((float)s.viewport.width), scaleY((float)s.viewport.height), s.viewport.minZ,
                              s.viewport.maxZ },
                            scissor);

            TransformUniforms tu;
            FillTransform(s, gameW, gameH, tu);
            if (transform.Changed(tu))
                SDL_PushGPUVertexUniformData(cmd, 0, &tu, sizeof(tu));
            if (!c.transformed) {
                if (lightingState == NULL || !SameLighting(*lightingState, s)) {
                    FillLighting(s, lightingValue);
                    lightingState = &s;
                }
                if (lighting.Changed(lightingValue))
                    SDL_PushGPUVertexUniformData(cmd, 1, &lightingValue, sizeof(lightingValue));
                TexGenUniforms tg;
                FillTexGen(s, tg);
                if (texGen.Changed(tg))
                    SDL_PushGPUVertexUniformData(cmd, 2, &tg, sizeof(tg));
            }
            CombinerUniforms cu;
            FillCombiner(s, samples, cu);
            if (combiner.Changed(cu))
                SDL_PushGPUFragmentUniformData(cmd, 0, &cu, sizeof(cu));
            if (boundTextures == NULL || memcmp(boundTextures->stages, s.stages, sizeof(s.stages)) != 0) {
                gpu.BindTextures(pass, s);
                boundTextures = &s;
            }

            SDL_GPUBuffer *vertices = c.buffer != NULL ? gpu.VertexBufferOf(c.buffer) : gpu.frameVertices;
            if (vertices != boundVertices || c.firstVertexByte != boundVertexOffset) {
                SDL_GPUBufferBinding vb = { vertices, c.firstVertexByte };
                SDL_BindGPUVertexBuffers(pass, 0, &vb, 1);
                boundVertices = vertices;
                boundVertexOffset = c.firstVertexByte;
            }
            if (c.firstIndex != ~0u) {
                if (!indicesBound) {
                    SDL_GPUBufferBinding ib = { gpu.frameIndices, 0 };
                    SDL_BindGPUIndexBuffer(pass, &ib, SDL_GPU_INDEXELEMENTSIZE_16BIT);
                    indicesBound = true;
                }
                SDL_DrawGPUIndexedPrimitives(pass, c.indexCount, 1, c.firstIndex, 0, 0);
            } else {
                SDL_DrawGPUPrimitives(pass, c.vertexCount, 1, 0, 0);
            }
        }
        if (pass != NULL)
            SDL_EndGPURenderPass(pass);

        if (!present) {
            SDL_SubmitGPUCommandBuffer(cmd);
            return;
        }

        // The multisampled picture into backColor, which is shown and read.
        if (backMsaa != NULL) {
            SDL_GPUColorTargetInfo resolve = {};
            resolve.texture = backMsaa;
            resolve.load_op = SDL_GPU_LOADOP_LOAD;
            resolve.store_op = SDL_GPU_STOREOP_RESOLVE_AND_STORE;
            resolve.resolve_texture = backColor;
            SDL_EndGPURenderPass(SDL_BeginGPURenderPass(cmd, &resolve, 1, NULL));
        }
        gpu.Present(cmd, backColor, renderWidth, renderHeight, backWidth, backHeight, readback, vsync, stretch);
    }
};

} // namespace

GfxExecutor *Gfx_CreateModernExecutor(void)
{
    return new ModernExecutor;
}
