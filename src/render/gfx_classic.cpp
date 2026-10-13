// The classic renderer: runs a recorded frame (gfx_core.cpp) as the original
// drew it, at the game's resolution, with one set of shaders that emulate
// Direct3D 7's fixed-function pipeline. video.renderer = classic.

#include "platform/platform.h"
#include "port/sys.h"
#include "render/gpu_context.h"

#include "shaders_spirv.h"

// SPIR-V and, when built, MSL code of one shader for GpuContext::Shader.
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

// ---- uniform blocks (std140, see shaders/common.glsl and fixed.frag) ----

struct LightUniform {
    float diffuse[4], specular[4], ambient[4], position[4], direction[4], attenuation[4], spot[4];
};

struct VertexUniforms {
    float worldView[16];
    float projection[16];
    float textureMatrix[2][16];
    float materialDiffuse[4], materialAmbient[4], materialSpecular[4], materialEmissive[4];
    float globalAmbient[4];
    LightUniform lights[8];
    int32_t lightFlags[4];
    int32_t sources[4];
    int32_t fogFlags[4];
    float fogParams[4];
    int32_t texCoordIndex[4];
    float target[4];
};

struct FragmentUniforms {
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

struct ClassicExecutor : GfxExecutor {
    GpuContext gpu;
    SDL_GPUShader *vs3d = nullptr, *vsTL = nullptr, *fs = nullptr, *vsClear = nullptr, *fsClear = nullptr;
    SDL_GPUTexture *backColor = nullptr, *backDepth = nullptr;
    int backWidth = 0, backHeight = 0;
    std::unordered_map<uint32_t, SDL_GPUGraphicsPipeline *> pipelines;

    ~ClassicExecutor() override
    {
        if (gpu.device == NULL)
            return;
        SDL_WaitForGPUIdle(gpu.device);
        for (auto &p : pipelines)
            SDL_ReleaseGPUGraphicsPipeline(gpu.device, p.second);
        SDL_ReleaseGPUTexture(gpu.device, backColor);
        SDL_ReleaseGPUTexture(gpu.device, backDepth);
        SDL_ReleaseGPUShader(gpu.device, vs3d);
        SDL_ReleaseGPUShader(gpu.device, vsTL);
        SDL_ReleaseGPUShader(gpu.device, fs);
        SDL_ReleaseGPUShader(gpu.device, vsClear);
        SDL_ReleaseGPUShader(gpu.device, fsClear);
    }

    bool CreateDevice()
    {
        if (!gpu.Create(Platform_GetSettingInt("video.anisotropy", 1)))
            return false;
        Sys_Log("gfx: classic renderer");
        vs3d = gpu.Shader(SHADER_CODE(fixed3d_vert), SDL_GPU_SHADERSTAGE_VERTEX, 0, 1);
        vsTL = gpu.Shader(SHADER_CODE(fixedtl_vert), SDL_GPU_SHADERSTAGE_VERTEX, 0, 1);
        fs = gpu.Shader(SHADER_CODE(fixed_frag), SDL_GPU_SHADERSTAGE_FRAGMENT, 4, 1);
        vsClear = gpu.Shader(SHADER_CODE(clear_vert), SDL_GPU_SHADERSTAGE_VERTEX, 0, 1);
        fsClear = gpu.Shader(SHADER_CODE(clear_frag), SDL_GPU_SHADERSTAGE_FRAGMENT, 0, 1);
        return vs3d && vsTL && fs && vsClear && fsClear;
    }

    bool SetVideoMode(int width, int height, bool fullscreen) override
    {
        if (gpu.device == NULL && !CreateDevice())
            return false;
        if (backColor == NULL || backWidth != width || backHeight != height) {
            SDL_WaitForGPUIdle(gpu.device);
            SDL_ReleaseGPUTexture(gpu.device, backColor);
            SDL_ReleaseGPUTexture(gpu.device, backDepth);
            backColor = gpu.CreateTexture(SDL_GPU_TEXTURETYPE_2D, gpu.colorFormat, width, height, 1, 1,
                                          SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
            backDepth = gpu.CreateTexture(SDL_GPU_TEXTURETYPE_2D, gpu.depthFormat, width, height, 1, 1,
                                          SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET);
            backWidth = width;
            backHeight = height;
            if (backColor == NULL || backDepth == NULL)
                return false;
        }
        SDL_SetWindowResizable(gpu.window, true);
        SetFullscreen(fullscreen);
        SDL_ShowWindow(gpu.window);
        return true;
    }

    // Fullscreen on the desktop resolution, or a centred window of the back
    // buffer size times video.window_scale.
    void SetFullscreen(bool fullscreen) override
    {
        SDL_Window *window = gpu.window;
        if (window == NULL)
            return;
        bool wasFullscreen = (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) != 0;
        SDL_SetWindowFullscreen(window, fullscreen);
        if (fullscreen)
            return;
        // Leaving fullscreen restores the earlier window size; the new size
        // only applies once the window is out of fullscreen.
        if (wasFullscreen)
            SDL_SyncWindow(window);
        int scale = Platform_GetSettingInt("video.window_scale", 1);
        if (scale < 1)
            scale = 1;
        SDL_SetWindowSize(window, backWidth * scale, backHeight * scale);
        SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    }

    void DestroyTexture(GfxTexture *t) override { gpu.DestroyTexture(t); }
    void DestroyVertexBuffer(GfxVertexBuffer *b) override { gpu.DestroyVertexBuffer(b); }
    void ReadTexture(GfxTexture *t) override { gpu.ReadTexture(t); }

    // ---- pipelines ----

    SDL_GPUGraphicsPipeline *Pipeline(const GfxCommand &cmd, const GfxDrawState &s)
    {
        bool depthTest = s.depthTest;
        uint32_t key = (cmd.transformed ? 1u : 0u) | (uint32_t)(cmd.primitive & 7) << 1 |
                       (s.alphaBlend ? 1u : 0u) << 4 | (uint32_t)(s.srcBlend & 15) << 5 |
                       (uint32_t)(s.destBlend & 15) << 9 | (depthTest ? 1u : 0u) << 13 |
                       ((depthTest && s.depthWrite) ? 1u : 0u) << 14 | (uint32_t)(s.depthFunc & 15) << 15 |
                       (uint32_t)(s.cull & 3) << 19;
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
        // Direct3D's front faces are clockwise; CULL_CCW (its default)
        // removes the back faces.
        info.rasterizer_state.front_face = SDL_GPU_FRONTFACE_CLOCKWISE;
        info.rasterizer_state.cull_mode = s.cull == GFX_CULL_CW    ? SDL_GPU_CULLMODE_FRONT
                                          : s.cull == GFX_CULL_CCW ? SDL_GPU_CULLMODE_BACK
                                                                   : SDL_GPU_CULLMODE_NONE;
        info.rasterizer_state.enable_depth_clip = true;
        info.depth_stencil_state.enable_depth_test = depthTest;
        info.depth_stencil_state.enable_depth_write = depthTest && s.depthWrite;
        info.depth_stencil_state.compare_op = Gpu_CompareOp(s.depthFunc);
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

    SDL_GPUGraphicsPipeline *ClearPipeline(bool color, bool depth)
    {
        uint32_t key = 0x80000000u | (color ? 1u : 0u) | (depth ? 2u : 0u);
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

    void FillVertexUniforms(const GfxDrawState &s, int targetW, int targetH, VertexUniforms &u)
    {
        memset(&u, 0, sizeof(u));
        Gpu_Multiply(s.world, s.view, u.worldView);
        memcpy(u.projection, &s.projection, sizeof(u.projection));
        memcpy(u.textureMatrix[0], &s.texture[0], 64);
        memcpy(u.textureMatrix[1], &s.texture[1], 64);
        Gpu_Copy4(s.material.diffuse, u.materialDiffuse);
        Gpu_Copy4(s.material.ambient, u.materialAmbient);
        Gpu_Copy4(s.material.specular, u.materialSpecular);
        Gpu_Copy4(s.material.emissive, u.materialEmissive);
        Gpu_Color(s.ambient, u.globalAmbient);
        u.globalAmbient[3] = s.material.power;
        for (int i = 0; i < 8; i++) {
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
            lu.spot[2] = s.lightEnabled[i] ? 1.0f : 0.0f;
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
        u.texCoordIndex[0] = (int32_t)s.stages[0].texCoordIndex;
        u.texCoordIndex[1] = (int32_t)s.stages[1].texCoordIndex;
        u.texCoordIndex[2] = s.stages[0].textureTransform;
        u.texCoordIndex[3] = s.stages[1].textureTransform;
        u.target[0] = (float)targetW;
        u.target[1] = (float)targetH;
        // Direct3D 7's half-pixel offset of the viewport transform.
        u.target[2] = s.viewport.width ? 1.0f / s.viewport.width : 0.0f;
        u.target[3] = s.viewport.height ? -1.0f / s.viewport.height : 0.0f;
    }

    void FillFragmentUniforms(const GfxDrawState &s, FragmentUniforms &u)
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

    SDL_GPURenderPass *BeginPass(SDL_GPUCommandBuffer *cmd, GfxTexture *target, int &w, int &h)
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
            color.texture = backColor;
            depth.texture = backDepth;
            w = backWidth;
            h = backHeight;
        } else {
            GfxTexture *root = target->cube != NULL ? target->cube : target;
            GpuTexture *g = gpu.Ensure(root);
            color.texture = g->texture;
            color.layer_or_depth_plane = target->cube != NULL ? target->face : 0;
            depth.texture = g->depth;
            w = root->width;
            h = root->height;
        }
        return SDL_BeginGPURenderPass(cmd, &color, 1, &depth);
    }

    void Execute(GfxFrame &frame, std::vector<GfxTexture *> &textures, GfxReadback &readback, bool present,
                 bool vsync, bool stretch) override
    {
        if (gpu.device == NULL)
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

        SDL_GPURenderPass *pass = NULL;
        GfxTexture *target = NULL;
        int targetW = backWidth, targetH = backHeight;
        for (const GfxCommand &c : frame.commands) {
            if (c.type == GFX_CMD_TARGET) {
                if (pass != NULL) {
                    SDL_EndGPURenderPass(pass);
                    pass = NULL;
                }
                target = c.target;
                continue;
            }
            if (pass == NULL)
                pass = BeginPass(cmd, target, targetW, targetH);
            if (c.type == GFX_CMD_CLEAR) {
                bool colorClear = (c.clearFlags & GFX_CLEAR_TARGET) != 0;
                bool depthClear = (c.clearFlags & GFX_CLEAR_ZBUFFER) != 0;
                if (!colorClear && !depthClear)
                    continue;
                SDL_BindGPUGraphicsPipeline(pass, ClearPipeline(colorClear, depthClear));
                SDL_GPUViewport vp = { (float)c.clearViewport.x, (float)c.clearViewport.y,
                                       (float)c.clearViewport.width, (float)c.clearViewport.height, 0.0f, 1.0f };
                SDL_SetGPUViewport(pass, &vp);
                SDL_Rect scissor = { (int)c.clearViewport.x, (int)c.clearViewport.y, (int)c.clearViewport.width,
                                     (int)c.clearViewport.height };
                SDL_SetGPUScissor(pass, &scissor);
                ClearUniforms u = {};
                Gpu_Color(c.clearColor, u.color);
                u.depth[0] = c.clearZ;
                SDL_PushGPUVertexUniformData(cmd, 0, &u, sizeof(u));
                SDL_PushGPUFragmentUniformData(cmd, 0, &u, sizeof(u));
                SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
                continue;
            }
            const GfxDrawState &s = frame.states[c.state];
            SDL_GPUGraphicsPipeline *pipeline = Pipeline(c, s);
            if (pipeline == NULL)
                continue;
            SDL_BindGPUGraphicsPipeline(pass, pipeline);
            SDL_GPUViewport vp;
            if (c.transformed)
                vp = { 0.0f, 0.0f, (float)targetW, (float)targetH, 0.0f, 1.0f };
            else
                vp = { (float)s.viewport.x, (float)s.viewport.y, (float)s.viewport.width, (float)s.viewport.height,
                       s.viewport.minZ, s.viewport.maxZ };
            SDL_SetGPUViewport(pass, &vp);
            SDL_Rect scissor = { (int)s.viewport.x, (int)s.viewport.y, (int)s.viewport.width, (int)s.viewport.height };
            SDL_SetGPUScissor(pass, &scissor);
            VertexUniforms vu;
            FillVertexUniforms(s, targetW, targetH, vu);
            SDL_PushGPUVertexUniformData(cmd, 0, &vu, sizeof(vu));
            FragmentUniforms fu;
            FillFragmentUniforms(s, fu);
            SDL_PushGPUFragmentUniformData(cmd, 0, &fu, sizeof(fu));
            gpu.BindTextures(pass, s);
            SDL_GPUBufferBinding vb = { c.buffer != NULL ? gpu.VertexBufferOf(c.buffer) : gpu.frameVertices,
                                        c.firstVertexByte };
            SDL_BindGPUVertexBuffers(pass, 0, &vb, 1);
            if (c.firstIndex != ~0u) {
                SDL_GPUBufferBinding ib = { gpu.frameIndices, 0 };
                SDL_BindGPUIndexBuffer(pass, &ib, SDL_GPU_INDEXELEMENTSIZE_16BIT);
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
        gpu.Present(cmd, backColor, backWidth, backHeight, backWidth, backHeight, readback, vsync, stretch);
    }
};

} // namespace

GfxExecutor *Gfx_CreateClassicExecutor(void)
{
    return new ClassicExecutor;
}
