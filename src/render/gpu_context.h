// What the SDL_GPU renderers share: the device and window, the GPU copies of
// textures and vertex buffers, samplers, and the conversions of Direct3D
// state to SDL_GPU's.
#ifndef OPENCMR2_GPU_CONTEXT_H
#define OPENCMR2_GPU_CONTEXT_H

#include "render/gfx_internal.h"

#include <SDL3/SDL.h>

#include <unordered_map>
#include <vector>

SDL_GPUBlendFactor Gpu_BlendFactor(int d3d);
SDL_GPUCompareOp Gpu_CompareOp(int d3d);
SDL_GPUPrimitiveType Gpu_PrimitiveType(int d3d);
// 0xAARRGGBB to RGBA floats.
void Gpu_Color(DWORD c, float out[4]);
void Gpu_Copy4(const GfxColorValue &c, float out[4]);
// out = a * b, Direct3D row-vector matrices.
void Gpu_Multiply(const GfxMatrix &a, const GfxMatrix &b, float *out);
void Gpu_TransformPoint(const GfxVector &v, float w, const GfxMatrix &m, float out[3]);

struct GpuTexture {
    SDL_GPUTexture *texture = nullptr;
    SDL_GPUTexture *depth = nullptr;       // render targets
    SDL_GPUTextureFormat format;
    bool cpuDecode = false;                // BC data decoded on the CPU (no BC support)
};

struct GpuVertexBuffer {
    SDL_GPUBuffer *buffer = nullptr;
    uint32_t capacity = 0;
    uint32_t version = 0;                  // GfxVertexBuffer::version uploaded
};

struct GpuContext {
    SDL_GPUDevice *device = nullptr;
    SDL_Window *window = nullptr;
    SDL_GPUShaderFormat shaderFormat = SDL_GPU_SHADERFORMAT_SPIRV;
    SDL_GPUTextureFormat colorFormat = SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    SDL_GPUTextureFormat depthFormat = SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT;
    SDL_GPUTexture *white2D = nullptr, *whiteCube = nullptr;
    bool bcSupported = true;
    int anisotropy = 1;
    bool smoothScaling = true;      // video.smooth_scaling: linear filtering to the window
    bool presentModeVsync = true;

    ~GpuContext();

    // Creates the device for the game window; anisotropy (1 = off) is
    // dropped when the GPU cannot do it.
    bool Create(int wantedAnisotropy);

    SDL_GPUShader *Shader(const unsigned char *spirv, size_t spirvSize, const unsigned char *msl, size_t mslSize,
                          SDL_GPUShaderStage stage, int samplers, int uniforms);
    SDL_GPUTexture *CreateTexture(SDL_GPUTextureType type, SDL_GPUTextureFormat format, int w, int h, int layers,
                                  int levels, SDL_GPUTextureUsageFlags usage,
                                  SDL_GPUSampleCount samples = SDL_GPU_SAMPLECOUNT_1);

    // Textures: the GPU copy of a game texture, made on first use.
    GpuTexture *Ensure(GfxTexture *t);
    void EnsureFrameTextures(const GfxFrame &frame);
    void UploadTextures(SDL_GPUCommandBuffer *cmd, std::vector<GfxTexture *> &textures);
    void DestroyTexture(GfxTexture *t);
    void ReadTexture(GfxTexture *t);

    // Vertex buffers drawn by reference: uploads those the game changed.
    void UploadVertexBuffers(SDL_GPUCommandBuffer *cmd, const GfxFrame &frame);
    void DestroyVertexBuffer(GfxVertexBuffer *b);
    SDL_GPUBuffer *VertexBufferOf(GfxVertexBuffer *b) { return ((GpuVertexBuffer *)b->gpu)->buffer; }

    // The frame's own vertices and indices, in buffers kept between frames.
    void UploadGeometry(SDL_GPUCommandBuffer *cmd, const GfxFrame &frame);
    SDL_GPUBuffer *frameVertices = nullptr, *frameIndices = nullptr;

    SDL_GPUSampler *Sampler(const GfxStageState &stage);
    // The two texture stages' 2D and cube bindings (white when unused).
    void BindTextures(SDL_GPURenderPass *pass, const GfxDrawState &s);
    // 0 none, 1 2D, 2 cube, 3 bump map.
    static int TextureKind(const GfxStageState &stage);

    // Shows a picture of renderW x renderH pixels, drawn for gameW x gameH,
    // in the window (letterboxed unless stretched), after reading back the
    // lens flare test area. Submits cmd.
    void Present(SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *picture, int renderW, int renderH, int gameW, int gameH,
                 GfxReadback &readback, bool vsync, bool stretch);

private:
    SDL_GPUDevice *CreateDevice(SDL_GPUShaderFormat formats, bool debug, bool withAnisotropy);
    SDL_GPUTextureFormat GpuFormat(int format, bool &cpuDecode);
    size_t UploadSize(GfxTexture *t, GpuTexture *g, int w, int h);
    void ConvertLevel(GfxTexture *t, GpuTexture *g, const uint8_t *src, int w, int h, uint8_t *dst);
    void EnsureBuffer(SDL_GPUBuffer *&buffer, uint32_t &capacity, uint32_t bytes, SDL_GPUBufferUsageFlags usage);
    // An upload buffer of at least bytes, reused between frames (cycled).
    SDL_GPUTransferBuffer *Staging(uint32_t bytes);

    std::unordered_map<uint32_t, SDL_GPUSampler *> samplers;
    uint32_t frameVertexCapacity = 0, frameIndexCapacity = 0;
    SDL_GPUTransferBuffer *staging = nullptr;
    uint32_t stagingCapacity = 0;
};

#endif
