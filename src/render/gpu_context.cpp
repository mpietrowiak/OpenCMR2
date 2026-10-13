// What the SDL_GPU renderers share: see gpu_context.h.

#include "platform/platform.h"
#include "port/sys.h"
#include "render/gpu_context.h"
#include "render/presentation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

// ---- conversions ---------------------------------------------------------------

SDL_GPUBlendFactor Gpu_BlendFactor(int d3d)
{
    switch (d3d) {
    case GFX_BLEND_ZERO: return SDL_GPU_BLENDFACTOR_ZERO;
    case GFX_BLEND_ONE: return SDL_GPU_BLENDFACTOR_ONE;
    case GFX_BLEND_SRCCOLOR: return SDL_GPU_BLENDFACTOR_SRC_COLOR;
    case GFX_BLEND_INVSRCCOLOR: return SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_COLOR;
    case GFX_BLEND_SRCALPHA: return SDL_GPU_BLENDFACTOR_SRC_ALPHA;
    case GFX_BLEND_INVSRCALPHA: return SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    case GFX_BLEND_DESTALPHA: return SDL_GPU_BLENDFACTOR_DST_ALPHA;
    case GFX_BLEND_INVDESTALPHA: return SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_ALPHA;
    case GFX_BLEND_DESTCOLOR: return SDL_GPU_BLENDFACTOR_DST_COLOR;
    case GFX_BLEND_INVDESTCOLOR: return SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_COLOR;
    case GFX_BLEND_SRCALPHASAT: return SDL_GPU_BLENDFACTOR_SRC_ALPHA_SATURATE;
    default: return SDL_GPU_BLENDFACTOR_ONE;
    }
}

SDL_GPUCompareOp Gpu_CompareOp(int d3d)
{
    switch (d3d) {
    case GFX_CMP_NEVER: return SDL_GPU_COMPAREOP_NEVER;
    case GFX_CMP_LESS: return SDL_GPU_COMPAREOP_LESS;
    case GFX_CMP_EQUAL: return SDL_GPU_COMPAREOP_EQUAL;
    case GFX_CMP_LESSEQUAL: return SDL_GPU_COMPAREOP_LESS_OR_EQUAL;
    case GFX_CMP_GREATER: return SDL_GPU_COMPAREOP_GREATER;
    case GFX_CMP_NOTEQUAL: return SDL_GPU_COMPAREOP_NOT_EQUAL;
    case GFX_CMP_GREATEREQUAL: return SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;
    default: return SDL_GPU_COMPAREOP_ALWAYS;
    }
}

SDL_GPUPrimitiveType Gpu_PrimitiveType(int d3d)
{
    switch (d3d) {
    case GFX_POINTLIST: return SDL_GPU_PRIMITIVETYPE_POINTLIST;
    case GFX_LINELIST: return SDL_GPU_PRIMITIVETYPE_LINELIST;
    case GFX_LINESTRIP: return SDL_GPU_PRIMITIVETYPE_LINESTRIP;
    case GFX_TRIANGLESTRIP: return SDL_GPU_PRIMITIVETYPE_TRIANGLESTRIP;
    default: return SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    }
}

void Gpu_Color(DWORD c, float out[4])
{
    out[0] = ((c >> 16) & 0xff) / 255.0f;
    out[1] = ((c >> 8) & 0xff) / 255.0f;
    out[2] = (c & 0xff) / 255.0f;
    out[3] = ((c >> 24) & 0xff) / 255.0f;
}

void Gpu_Copy4(const GfxColorValue &c, float out[4])
{
    out[0] = c.r;
    out[1] = c.g;
    out[2] = c.b;
    out[3] = c.a;
}

void Gpu_Multiply(const GfxMatrix &a, const GfxMatrix &b, float *out)
{
    const float *pa = &a._11, *pb = &b._11;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            float s = 0;
            for (int k = 0; k < 4; k++)
                s += pa[r * 4 + k] * pb[k * 4 + c];
            out[r * 4 + c] = s;
        }
}

void Gpu_TransformPoint(const GfxVector &v, float w, const GfxMatrix &m, float out[3])
{
    const float *p = &m._11;
    for (int c = 0; c < 3; c++)
        out[c] = v.x * p[c] + v.y * p[4 + c] + v.z * p[8 + c] + w * p[12 + c];
}

// ---- device ----------------------------------------------------------------------

GpuContext::~GpuContext()
{
    if (device == NULL)
        return;
    SDL_WaitForGPUIdle(device);
    for (auto &s : samplers)
        SDL_ReleaseGPUSampler(device, s.second);
    SDL_ReleaseGPUTexture(device, white2D);
    SDL_ReleaseGPUTexture(device, whiteCube);
    SDL_ReleaseGPUBuffer(device, frameVertices);
    SDL_ReleaseGPUBuffer(device, frameIndices);
    SDL_ReleaseGPUTransferBuffer(device, staging);
    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyGPUDevice(device);
}

SDL_GPUDevice *GpuContext::CreateDevice(SDL_GPUShaderFormat formats, bool debug, bool withAnisotropy)
{
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN,
                           (formats & SDL_GPU_SHADERFORMAT_SPIRV) != 0);
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_MSL_BOOLEAN,
                           (formats & SDL_GPU_SHADERFORMAT_MSL) != 0);
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_DEBUGMODE_BOOLEAN, debug);
    // SDL asks Vulkan for these features by default. The renderers do not use
    // them (anisotropy only when asked for), and many Android GPUs lack some,
    // for which SDL would refuse the device.
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_FEATURE_CLIP_DISTANCE_BOOLEAN, false);
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_FEATURE_DEPTH_CLAMPING_BOOLEAN, false);
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_FEATURE_INDIRECT_DRAW_FIRST_INSTANCE_BOOLEAN, false);
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_FEATURE_ANISOTROPY_BOOLEAN, withAnisotropy);
    SDL_GPUDevice *created = SDL_CreateGPUDeviceWithProperties(props);
    SDL_DestroyProperties(props);
    return created;
}

bool GpuContext::Create(int wantedAnisotropy)
{
    window = Platform_GetWindow();
    if (window == NULL) {
        Sys_Log("gfx: no window");
        return false;
    }
    bool debug = Platform_GetSettingInt("video.gpu_debug", 0) != 0;
    SDL_GPUShaderFormat formats = SDL_GPU_SHADERFORMAT_SPIRV;
#ifdef OPENCMR2_SHADERS_MSL
    formats |= SDL_GPU_SHADERFORMAT_MSL;
#endif
    anisotropy = std::max(1, std::min(16, wantedAnisotropy));
    smoothScaling = Platform_GetSettingInt("video.smooth_scaling", 1) != 0;
    device = CreateDevice(formats, debug, anisotropy > 1);
    if (device == NULL && anisotropy > 1) {
        // Some GPUs (many on Android) lack anisotropic filtering, for which
        // SDL refuses the whole device: go without it.
        anisotropy = 1;
        device = CreateDevice(formats, debug, false);
    }
    if (device == NULL) {
        Sys_Log("gfx: SDL_CreateGPUDevice: %s", SDL_GetError());
        return false;
    }
    if (!(SDL_GetGPUShaderFormats(device) & SDL_GPU_SHADERFORMAT_SPIRV))
        shaderFormat = SDL_GPU_SHADERFORMAT_MSL;
    if (!SDL_ClaimWindowForGPUDevice(device, window)) {
        Sys_Log("gfx: SDL_ClaimWindowForGPUDevice: %s", SDL_GetError());
        return false;
    }
    Sys_Log("gfx: SDL_GPU driver %s", SDL_GetGPUDeviceDriver(device));
    if (!SDL_GPUTextureSupportsFormat(device, depthFormat, SDL_GPU_TEXTURETYPE_2D,
                                      SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET))
        depthFormat = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    bcSupported = SDL_GPUTextureSupportsFormat(device, SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM,
                                               SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_SAMPLER);

    white2D = CreateTexture(SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM, 1, 1, 1, 1,
                            SDL_GPU_TEXTUREUSAGE_SAMPLER);
    whiteCube = CreateTexture(SDL_GPU_TEXTURETYPE_CUBE, SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM, 1, 1, 6, 1,
                              SDL_GPU_TEXTUREUSAGE_SAMPLER);
    uint32_t white[6] = { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff };
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(device);
    SDL_GPUTransferBufferCreateInfo tbi = { SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, sizeof(white), 0 };
    SDL_GPUTransferBuffer *tb = SDL_CreateGPUTransferBuffer(device, &tbi);
    memcpy(SDL_MapGPUTransferBuffer(device, tb, false), white, sizeof(white));
    SDL_UnmapGPUTransferBuffer(device, tb);
    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    for (int layer = 0; layer < 7; layer++) {
        SDL_GPUTextureTransferInfo src = { tb, (Uint32)(layer < 6 ? layer * 4 : 0), 0, 0 };
        SDL_GPUTextureRegion dst = {};
        dst.texture = layer < 6 ? whiteCube : white2D;
        dst.layer = layer < 6 ? layer : 0;
        dst.w = dst.h = dst.d = 1;
        SDL_UploadToGPUTexture(copy, &src, &dst, false);
    }
    SDL_EndGPUCopyPass(copy);
    SDL_SubmitGPUCommandBuffer(cmd);
    SDL_ReleaseGPUTransferBuffer(device, tb);
    return true;
}

SDL_GPUShader *GpuContext::Shader(const unsigned char *spirv, size_t spirvSize, const unsigned char *msl,
                                  size_t mslSize, SDL_GPUShaderStage stage, int samplerCount, int uniforms)
{
    SDL_GPUShaderCreateInfo info = {};
    if (shaderFormat == SDL_GPU_SHADERFORMAT_MSL) {
        info.code = msl;
        info.code_size = mslSize - 1; // without the NUL the embedding adds
        info.entrypoint = "main0";    // SPIRV-Cross renames main for MSL
    } else {
        info.code = spirv;
        info.code_size = spirvSize;
        info.entrypoint = "main";
    }
    info.format = shaderFormat;
    info.stage = stage;
    info.num_samplers = samplerCount;
    info.num_uniform_buffers = uniforms;
    SDL_GPUShader *shader = SDL_CreateGPUShader(device, &info);
    if (shader == NULL)
        Sys_Log("gfx: shader: %s", SDL_GetError());
    return shader;
}

SDL_GPUTexture *GpuContext::CreateTexture(SDL_GPUTextureType type, SDL_GPUTextureFormat format, int w, int h,
                                          int layers, int levels, SDL_GPUTextureUsageFlags usage,
                                          SDL_GPUSampleCount samples)
{
    SDL_GPUTextureCreateInfo info = {};
    info.type = type;
    info.format = format;
    info.usage = usage;
    info.width = w;
    info.height = h;
    info.layer_count_or_depth = layers;
    info.num_levels = levels;
    info.sample_count = samples;
    SDL_GPUTexture *t = SDL_CreateGPUTexture(device, &info);
    if (t == NULL)
        Sys_Log("gfx: texture %dx%d: %s", w, h, SDL_GetError());
    return t;
}

// ---- textures ----------------------------------------------------------------------

SDL_GPUTextureFormat GpuContext::GpuFormat(int format, bool &cpuDecode)
{
    cpuDecode = false;
    switch (format) {
    case GFX_FORMAT_BC1:
        if (bcSupported)
            return SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM;
        cpuDecode = true;
        return SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    case GFX_FORMAT_BC3:
        if (bcSupported)
            return SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM;
        cpuDecode = true;
        return SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    case GFX_FORMAT_BUMP_DUDVL:
        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    default:
        return SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    }
}

GpuTexture *GpuContext::Ensure(GfxTexture *t)
{
    if (t->gpu != NULL)
        return (GpuTexture *)t->gpu;
    GpuTexture *g = new GpuTexture;
    g->format = GpuFormat(t->format, g->cpuDecode);
    bool cube = (t->flags & GFX_TEXTURE_CUBE) != 0;
    bool target = (t->flags & GFX_TEXTURE_RENDER_TARGET) != 0;
    SDL_GPUTextureUsageFlags usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    if (target)
        usage |= SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    g->texture = CreateTexture(cube ? SDL_GPU_TEXTURETYPE_CUBE : SDL_GPU_TEXTURETYPE_2D, g->format, t->width,
                               t->height, cube ? 6 : 1, t->mipLevels, usage);
    if (target)
        g->depth = CreateTexture(SDL_GPU_TEXTURETYPE_2D, depthFormat, t->width, t->height, 1, 1,
                                 SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET);
    t->gpu = g;
    t->dirty = true;
    return g;
}

void GpuContext::EnsureFrameTextures(const GfxFrame &frame)
{
    // Textures used by the frame must exist before the passes start.
    for (const GfxDrawState &s : frame.states)
        for (const GfxStageState &stage : s.stages)
            if (stage.texture != NULL)
                Ensure(stage.texture->cube != NULL ? stage.texture->cube : stage.texture);
    for (const GfxCommand &c : frame.commands)
        if (c.type == GFX_CMD_TARGET && c.target != NULL)
            Ensure(c.target->cube != NULL ? c.target->cube : c.target);
}

size_t GpuContext::UploadSize(GfxTexture *t, GpuTexture *g, int w, int h)
{
    if ((t->format == GFX_FORMAT_BC1 || t->format == GFX_FORMAT_BC3) && !g->cpuDecode)
        return Gfx_LevelSize(t->format, w, h, NULL);
    return (size_t)w * h * 4;
}

void GpuContext::ConvertLevel(GfxTexture *t, GpuTexture *g, const uint8_t *src, int w, int h, uint8_t *dst)
{
    size_t n = (size_t)w * h;
    switch (t->format) {
    case GFX_FORMAT_BGRX8:
        for (size_t i = 0; i < n; i++)
            ((uint32_t *)dst)[i] = ((const uint32_t *)src)[i] | 0xff000000;
        break;
    case GFX_FORMAT_BUMP_DUDVL:
        for (size_t i = 0; i < n; i++) {
            dst[i * 4] = src[i * 3];
            dst[i * 4 + 1] = src[i * 3 + 1];
            dst[i * 4 + 2] = src[i * 3 + 2];
            dst[i * 4 + 3] = 0xff;
        }
        break;
    case GFX_FORMAT_BC1:
    case GFX_FORMAT_BC3:
        if (g->cpuDecode) {
            int blockBytes = t->format == GFX_FORMAT_BC1 ? 8 : 16;
            int bw = (w + 3) / 4, bh = (h + 3) / 4;
            uint32_t block[16];
            for (int by = 0; by < bh; by++)
                for (int bx = 0; bx < bw; bx++) {
                    Gfx_DecodeBlock(t->format, src + (by * bw + bx) * blockBytes, block);
                    for (int y = 0; y < 4; y++)
                        for (int x = 0; x < 4; x++)
                            if (bx * 4 + x < w && by * 4 + y < h)
                                ((uint32_t *)dst)[(size_t)(by * 4 + y) * w + bx * 4 + x] = block[y * 4 + x];
                }
        } else {
            memcpy(dst, src, Gfx_LevelSize(t->format, w, h, NULL));
        }
        break;
    default:
        memcpy(dst, src, n * 4);
        break;
    }
}

void GpuContext::UploadTextures(SDL_GPUCommandBuffer *cmd, std::vector<GfxTexture *> &textures)
{
    size_t total = 0;
    for (GfxTexture *t : textures) {
        if (!t->dirty)
            continue;
        GpuTexture *g = Ensure(t);
        if (g->texture == NULL) {
            t->dirty = false;
            continue;
        }
        int faces = (t->flags & GFX_TEXTURE_CUBE) ? 6 : 1;
        for (int f = 0; f < faces; f++)
            for (int m = 0; m < t->mipLevels; m++)
                total += UploadSize(t, g, std::max(1, t->width >> m), std::max(1, t->height >> m));
    }
    if (total == 0)
        return;
    SDL_GPUTransferBufferCreateInfo tbi = { SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, (Uint32)total, 0 };
    SDL_GPUTransferBuffer *tb = SDL_CreateGPUTransferBuffer(device, &tbi);
    uint8_t *map = (uint8_t *)SDL_MapGPUTransferBuffer(device, tb, false);
    struct Pending {
        SDL_GPUTexture *texture;
        int face, level, w, h;
        size_t offset;
    };
    std::vector<Pending> pending;
    size_t offset = 0;
    for (GfxTexture *t : textures) {
        if (!t->dirty)
            continue;
        GpuTexture *g = (GpuTexture *)t->gpu;
        int faces = (t->flags & GFX_TEXTURE_CUBE) ? 6 : 1;
        for (int f = 0; f < faces; f++) {
            for (int m = 0; m < t->mipLevels; m++) {
                int w = std::max(1, t->width >> m), h = std::max(1, t->height >> m);
                ConvertLevel(t, g, t->levels[f][m].data(), w, h, map + offset);
                pending.push_back({ g->texture, f, m, w, h, offset });
                offset += UploadSize(t, g, w, h);
            }
        }
        t->dirty = false;
    }
    SDL_UnmapGPUTransferBuffer(device, tb);
    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    for (const Pending &p : pending) {
        SDL_GPUTextureTransferInfo src = { tb, (Uint32)p.offset, 0, 0 };
        SDL_GPUTextureRegion dst = {};
        dst.texture = p.texture;
        dst.mip_level = p.level;
        dst.layer = p.face;
        dst.w = p.w;
        dst.h = p.h;
        dst.d = 1;
        SDL_UploadToGPUTexture(copy, &src, &dst, false);
    }
    SDL_EndGPUCopyPass(copy);
    SDL_ReleaseGPUTransferBuffer(device, tb);
}

void GpuContext::DestroyTexture(GfxTexture *t)
{
    GpuTexture *g = (GpuTexture *)t->gpu;
    if (g == NULL || device == NULL)
        return;
    SDL_ReleaseGPUTexture(device, g->texture);
    SDL_ReleaseGPUTexture(device, g->depth);
    delete g;
    t->gpu = NULL;
}

void GpuContext::ReadTexture(GfxTexture *t)
{
    GpuTexture *g = (GpuTexture *)t->gpu;
    if (g == NULL || (t->format != GFX_FORMAT_BGRX8 && t->format != GFX_FORMAT_BGRA8))
        return;
    int faces = (t->flags & GFX_TEXTURE_CUBE) ? 6 : 1;
    size_t level = (size_t)t->width * t->height * 4;
    SDL_GPUTransferBufferCreateInfo tbi = { SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, (Uint32)(level * faces), 0 };
    SDL_GPUTransferBuffer *tb = SDL_CreateGPUTransferBuffer(device, &tbi);
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(device);
    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    for (int f = 0; f < faces; f++) {
        SDL_GPUTextureRegion src = {};
        src.texture = g->texture;
        src.layer = f;
        src.w = t->width;
        src.h = t->height;
        src.d = 1;
        SDL_GPUTextureTransferInfo dst = { tb, (Uint32)(level * f), 0, 0 };
        SDL_DownloadFromGPUTexture(copy, &src, &dst);
    }
    SDL_EndGPUCopyPass(copy);
    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    SDL_WaitForGPUFences(device, true, &fence, 1);
    SDL_ReleaseGPUFence(device, fence);
    const uint8_t *map = (const uint8_t *)SDL_MapGPUTransferBuffer(device, tb, false);
    for (int f = 0; f < faces; f++)
        memcpy(t->levels[f][0].data(), map + level * f, level);
    SDL_UnmapGPUTransferBuffer(device, tb);
    SDL_ReleaseGPUTransferBuffer(device, tb);
}

// ---- buffers ----------------------------------------------------------------------

void GpuContext::EnsureBuffer(SDL_GPUBuffer *&buffer, uint32_t &capacity, uint32_t bytes,
                              SDL_GPUBufferUsageFlags usage)
{
    if (bytes <= capacity && buffer != NULL)
        return;
    SDL_ReleaseGPUBuffer(device, buffer);
    capacity = std::max(bytes, capacity * 2);
    SDL_GPUBufferCreateInfo info = { usage, capacity, 0 };
    buffer = SDL_CreateGPUBuffer(device, &info);
}

SDL_GPUTransferBuffer *GpuContext::Staging(uint32_t bytes)
{
    if (bytes > stagingCapacity || staging == NULL) {
        SDL_ReleaseGPUTransferBuffer(device, staging);
        stagingCapacity = std::max(bytes, stagingCapacity * 2);
        SDL_GPUTransferBufferCreateInfo tbi = { SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, stagingCapacity, 0 };
        staging = SDL_CreateGPUTransferBuffer(device, &tbi);
    }
    return staging;
}

void GpuContext::UploadVertexBuffers(SDL_GPUCommandBuffer *cmd, const GfxFrame &frame)
{
    SDL_GPUCopyPass *copy = NULL;
    for (GfxVertexBuffer *b : frame.buffers) {
        GpuVertexBuffer *g = (GpuVertexBuffer *)b->gpu;
        if (g == NULL)
            b->gpu = g = new GpuVertexBuffer;
        if (g->version == b->version && g->buffer != NULL)
            continue;
        uint32_t bytes = (uint32_t)(b->vertices.size() * sizeof(GfxVertex));
        if (bytes == 0)
            continue;
        if (bytes > g->capacity) {
            SDL_ReleaseGPUBuffer(device, g->buffer);
            SDL_GPUBufferCreateInfo info = { SDL_GPU_BUFFERUSAGE_VERTEX, bytes, 0 };
            g->buffer = SDL_CreateGPUBuffer(device, &info);
            g->capacity = bytes;
        }
        SDL_GPUTransferBufferCreateInfo tbi = { SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, bytes, 0 };
        SDL_GPUTransferBuffer *tb = SDL_CreateGPUTransferBuffer(device, &tbi);
        memcpy(SDL_MapGPUTransferBuffer(device, tb, false), b->vertices.data(), bytes);
        SDL_UnmapGPUTransferBuffer(device, tb);
        if (copy == NULL)
            copy = SDL_BeginGPUCopyPass(cmd);
        SDL_GPUTransferBufferLocation src = { tb, 0 };
        SDL_GPUBufferRegion dst = { g->buffer, 0, bytes };
        // Cycling keeps the contents earlier frames still draw from.
        SDL_UploadToGPUBuffer(copy, &src, &dst, true);
        SDL_ReleaseGPUTransferBuffer(device, tb);
        g->version = b->version;
    }
    if (copy != NULL)
        SDL_EndGPUCopyPass(copy);
}

void GpuContext::DestroyVertexBuffer(GfxVertexBuffer *b)
{
    GpuVertexBuffer *g = (GpuVertexBuffer *)b->gpu;
    if (g == NULL)
        return;
    if (device != NULL)
        SDL_ReleaseGPUBuffer(device, g->buffer);
    delete g;
    b->gpu = NULL;
}

void GpuContext::UploadGeometry(SDL_GPUCommandBuffer *cmd, const GfxFrame &frame)
{
    uint32_t vertexBytes = (uint32_t)frame.vertices.size();
    uint32_t indexBytes = (uint32_t)(frame.indices.size() * 2);
    if (vertexBytes == 0 && indexBytes == 0)
        return;
    EnsureBuffer(frameVertices, frameVertexCapacity, std::max(vertexBytes, 4u), SDL_GPU_BUFFERUSAGE_VERTEX);
    EnsureBuffer(frameIndices, frameIndexCapacity, std::max(indexBytes, 4u), SDL_GPU_BUFFERUSAGE_INDEX);
    SDL_GPUTransferBuffer *tb = Staging(vertexBytes + indexBytes);
    // Cycled: a fresh area when the GPU still reads the previous frame's.
    uint8_t *map = (uint8_t *)SDL_MapGPUTransferBuffer(device, tb, true);
    memcpy(map, frame.vertices.data(), vertexBytes);
    if (indexBytes)
        memcpy(map + vertexBytes, frame.indices.data(), indexBytes);
    SDL_UnmapGPUTransferBuffer(device, tb);
    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    if (vertexBytes) {
        SDL_GPUTransferBufferLocation src = { tb, 0 };
        SDL_GPUBufferRegion dst = { frameVertices, 0, vertexBytes };
        SDL_UploadToGPUBuffer(copy, &src, &dst, true);
    }
    if (indexBytes) {
        SDL_GPUTransferBufferLocation isrc = { tb, vertexBytes };
        SDL_GPUBufferRegion idst = { frameIndices, 0, indexBytes };
        SDL_UploadToGPUBuffer(copy, &isrc, &idst, true);
    }
    SDL_EndGPUCopyPass(copy);
}

// ---- samplers and bindings ----------------------------------------------------------

SDL_GPUSampler *GpuContext::Sampler(const GfxStageState &stage)
{
    uint32_t key = (uint32_t)(stage.minFilter & 3) | (uint32_t)(stage.magFilter & 3) << 2 |
                   (uint32_t)(stage.mipFilter & 3) << 4 | (uint32_t)(stage.address & 3) << 6;
    auto it = samplers.find(key);
    if (it != samplers.end())
        return it->second;
    SDL_GPUSamplerCreateInfo info = {};
    info.min_filter = stage.minFilter == GFX_FILTER_LINEAR ? SDL_GPU_FILTER_LINEAR : SDL_GPU_FILTER_NEAREST;
    info.mag_filter = stage.magFilter == GFX_FILTER_LINEAR ? SDL_GPU_FILTER_LINEAR : SDL_GPU_FILTER_NEAREST;
    info.mipmap_mode = stage.mipFilter == GFX_MIPFILTER_LINEAR ? SDL_GPU_SAMPLERMIPMAPMODE_LINEAR
                                                               : SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    SDL_GPUSamplerAddressMode mode = stage.address == GFX_ADDRESS_CLAMP    ? SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE
                                     : stage.address == GFX_ADDRESS_MIRROR ? SDL_GPU_SAMPLERADDRESSMODE_MIRRORED_REPEAT
                                                                           : SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    info.address_mode_u = info.address_mode_v = info.address_mode_w = mode;
    info.min_lod = 0.0f;
    info.max_lod = stage.mipFilter == GFX_MIPFILTER_NONE ? 0.0f : 1000.0f;
    if (anisotropy > 1 && stage.mipFilter != GFX_MIPFILTER_NONE) {
        info.enable_anisotropy = true;
        info.max_anisotropy = (float)anisotropy;
    }
    SDL_GPUSampler *s = SDL_CreateGPUSampler(device, &info);
    samplers[key] = s;
    return s;
}

int GpuContext::TextureKind(const GfxStageState &stage)
{
    GfxTexture *t = stage.texture;
    if (t == NULL)
        return 0;
    if (t->cube != NULL)
        t = t->cube;
    if (t->flags & GFX_TEXTURE_CUBE)
        return 2;
    if (t->format == GFX_FORMAT_BUMP_DUDVL)
        return 3;
    return 1;
}

void GpuContext::BindTextures(SDL_GPURenderPass *pass, const GfxDrawState &s)
{
    SDL_GPUTextureSamplerBinding bindings[4];
    for (int i = 0; i < 2; i++) {
        const GfxStageState &stage = s.stages[i];
        GfxTexture *t = stage.texture;
        if (t != NULL && t->cube != NULL)
            t = t->cube;
        GpuTexture *g = t != NULL ? Ensure(t) : NULL;
        SDL_GPUSampler *sampler = Sampler(stage);
        bool cube = t != NULL && (t->flags & GFX_TEXTURE_CUBE);
        SDL_GPUTexture *texture = g != NULL ? g->texture : NULL;
        bindings[i * 2] = { (texture != NULL && !cube) ? texture : white2D, sampler };
        bindings[i * 2 + 1] = { (texture != NULL && cube) ? texture : whiteCube, sampler };
    }
    SDL_BindGPUFragmentSamplers(pass, 0, bindings, 4);
}

// ---- presentation --------------------------------------------------------------------

void GpuContext::Present(SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *picture, int renderW, int renderH, int gameW,
                         int gameH, GfxReadback &readback, bool vsync, bool stretch)
{
    // Read back before the frame is shown (the lens flare test). The request
    // is in the game's pixels; each is taken from the middle of the picture
    // pixels it covers.
    SDL_GPUTransferBuffer *readbackBuffer = NULL;
    int readW = 0, readH = 0;
    if (readback.requested && readback.width > 0 && readback.height > 0) {
        int x = std::max(0, std::min(readback.x, gameW - 1));
        int y = std::max(0, std::min(readback.y, gameH - 1));
        int w = std::min(readback.width, gameW - x), h = std::min(readback.height, gameH - y);
        int readX = x * renderW / gameW, readY = y * renderH / gameH;
        readW = std::max(1, std::min(renderW - readX, (x + w) * renderW / gameW - readX));
        readH = std::max(1, std::min(renderH - readY, (y + h) * renderH / gameH - readY));
        SDL_GPUTransferBufferCreateInfo tbi = { SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, (Uint32)(readW * readH * 4), 0 };
        readbackBuffer = SDL_CreateGPUTransferBuffer(device, &tbi);
        SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
        SDL_GPUTextureRegion src = {};
        src.texture = picture;
        src.x = readX;
        src.y = readY;
        src.w = readW;
        src.h = readH;
        src.d = 1;
        SDL_GPUTextureTransferInfo dst = { readbackBuffer, 0, 0, 0 };
        SDL_DownloadFromGPUTexture(copy, &src, &dst);
        SDL_EndGPUCopyPass(copy);
        readback.readyWidth = w;
        readback.readyHeight = h;
    }

    // OPENCMR2_CAPTURE=dir: the picture shown every OPENCMR2_CAPTURE_EVERY
    // frames (60) as dir/frame-N.bmp, to compare renderers and builds.
    static const char *captureDir = getenv("OPENCMR2_CAPTURE");
    static int captureEvery = getenv("OPENCMR2_CAPTURE_EVERY") ? atoi(getenv("OPENCMR2_CAPTURE_EVERY")) : 60;
    static unsigned presented = 0;
    SDL_GPUTransferBuffer *captureBuffer = NULL;
    if (captureDir != NULL && captureEvery > 0 && presented % captureEvery == 0) {
        SDL_GPUTransferBufferCreateInfo tbi = { SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, (Uint32)(renderW * renderH * 4), 0 };
        captureBuffer = SDL_CreateGPUTransferBuffer(device, &tbi);
        SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
        SDL_GPUTextureRegion src = {};
        src.texture = picture;
        src.w = renderW;
        src.h = renderH;
        src.d = 1;
        SDL_GPUTextureTransferInfo dst = { captureBuffer, 0, 0, 0 };
        SDL_DownloadFromGPUTexture(copy, &src, &dst);
        SDL_EndGPUCopyPass(copy);
    }
    unsigned frameNumber = presented++;

    if (vsync != presentModeVsync) {
        SDL_GPUPresentMode mode = vsync ? SDL_GPU_PRESENTMODE_VSYNC : SDL_GPU_PRESENTMODE_IMMEDIATE;
        if (!vsync && !SDL_WindowSupportsGPUPresentMode(device, window, mode))
            mode = SDL_GPU_PRESENTMODE_MAILBOX;
        if (SDL_WindowSupportsGPUPresentMode(device, window, mode))
            SDL_SetGPUSwapchainParameters(device, window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, mode);
        presentModeVsync = vsync;
    }

    SDL_GPUTexture *swapchain = NULL;
    Uint32 sw = 0, sh = 0;
    if (SDL_WaitAndAcquireGPUSwapchainTexture(cmd, window, &swapchain, &sw, &sh) && swapchain != NULL) {
        // Movie frames fill the drawable; ordinary game frames retain their aspect ratio.
        PresentationRect rect = GetPresentationRect(gameW, gameH, sw, sh, stretch);
        SDL_GPUBlitInfo blit = {};
        blit.source.texture = picture;
        blit.source.w = renderW;
        blit.source.h = renderH;
        blit.destination.texture = swapchain;
        blit.destination.x = rect.x;
        blit.destination.y = rect.y;
        blit.destination.w = rect.width;
        blit.destination.h = rect.height;
        blit.load_op = SDL_GPU_LOADOP_CLEAR;
        blit.clear_color = { 0.0f, 0.0f, 0.0f, 1.0f };
        blit.filter = smoothScaling ? SDL_GPU_FILTER_LINEAR : SDL_GPU_FILTER_NEAREST;
        SDL_BlitGPUTexture(cmd, &blit);
    }

    if (readbackBuffer != NULL || captureBuffer != NULL) {
        SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
        SDL_WaitForGPUFences(device, true, &fence, 1);
        SDL_ReleaseGPUFence(device, fence);
    }
    if (captureBuffer != NULL) {
        void *pixels = SDL_MapGPUTransferBuffer(device, captureBuffer, false);
        SDL_Surface *surface = SDL_CreateSurfaceFrom(renderW, renderH, SDL_PIXELFORMAT_BGRX32, pixels, renderW * 4);
        char path[1024];
        snprintf(path, sizeof(path), "%s/frame-%06u.bmp", captureDir, frameNumber);
        if (surface == NULL || !SDL_SaveBMP(surface, path))
            Sys_Log("gfx: capture %s: %s", path, SDL_GetError());
        SDL_DestroySurface(surface);
        SDL_UnmapGPUTransferBuffer(device, captureBuffer);
        SDL_ReleaseGPUTransferBuffer(device, captureBuffer);
    }
    if (readbackBuffer != NULL) {
        const uint32_t *map = (const uint32_t *)SDL_MapGPUTransferBuffer(device, readbackBuffer, false);
        int w = readback.readyWidth, h = readback.readyHeight;
        readback.pixels.resize((size_t)w * h);
        for (int y = 0; y < h; y++) {
            int py = std::min(readH - 1, (2 * y + 1) * readH / (2 * h));
            for (int x = 0; x < w; x++) {
                int px = std::min(readW - 1, (2 * x + 1) * readW / (2 * w));
                readback.pixels[(size_t)y * w + x] = map[(size_t)py * readW + px];
            }
        }
        SDL_UnmapGPUTransferBuffer(device, readbackBuffer);
        SDL_ReleaseGPUTransferBuffer(device, readbackBuffer);
        readback.ready = true;
        readback.requested = false;
    } else if (captureBuffer == NULL) {
        SDL_SubmitGPUCommandBuffer(cmd);
    }
}
