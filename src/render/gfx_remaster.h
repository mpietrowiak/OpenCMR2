// The remaster renderer (video.graphics = remaster): what the modern renderer
// (gfx_modern.cpp) adds around the game's own drawing.
//
// The game lights its stages with colours baked into the vertices, the same
// whichever way a surface faces, and gives the cars a dim directional light.
// The remaster takes the baked light out of the stage's colours and lights
// the scene again with a sun and a sky: the sun is the game's directional
// light (it follows the sun drawn in the sky), the sky a colour from above
// and one from below. The 3D part of the frame is drawn into an HDR scene
// target; at its end, bloom and tone mapping turn it into the picture the HUD
// is drawn over.
//
// The sun casts shadows from four depth maps, seen along the light:
// - Three static cascades around the view, near to far, of the stage (what
//   the game draws with GFX_SHADOW_STATIC). They are kept from frame to
//   frame: each covers its part of the view with a margin, on whole texels,
//   and is drawn again only as the view nears its edge (or, now and then, to
//   take in what the game began drawing since). The frame before, the game
//   is asked for the stage it culls near the camera (Gfx_ShadowCastersWanted).
// - A dynamic map around the camera of the rest (the cars), every frame.
// The scene's shader takes the darker of the two.
#ifndef OPENCMR2_GFX_REMASTER_H
#define OPENCMR2_GFX_REMASTER_H

#include "render/gpu_context.h"

#include <stdint.h>

#include <array>
#include <unordered_map>
#include <vector>

// The camera of the frame's 3D view.
struct SceneView {
    float projection[4];    // P11, P22, P31, P32
    GfxMatrix view;
};

// The frame's sun: the game's directional light, in world space, towards
// where its light goes. The game also dims it while the car is in the
// shade of the stage, so a dark light only means night when it has not been
// bright since the stage began.
struct SunLight {
    bool bright = false;
    float direction[3] = { 0, -1, 0 };
};

// What one scan of a recorded frame finds of its scene: the back buffer's
// drawing up to its last 3D draw, when it shows a stage (GFX_SHADOW_STATIC).
struct RemasterScene {
    size_t last = SIZE_MAX;         // the last 3D draw; SIZE_MAX: no scene
    bool hasView = false;
    SceneView view;                 // the main camera (the largest viewport's)
    bool hasSun = false;
    SunLight sun;
    std::vector<size_t> casters;    // the solid draws, which cast shadows
};

// How the remaster lights a draw (the scene's fragment shader's mode).
enum RemasterSurface {
    REMASTER_NONE = 0,          // as the game draws it (sky, effects, night)
    REMASTER_BAKED = 1,         // stage surfaces: their vertex colour is their tint
    REMASTER_LIT = 2,           // objects the game lights (cars)
    REMASTER_REFLECTION = 3,    // the cars' environment reflection pass
    REMASTER_FOLIAGE = 4        // billboards (trees, bushes) facing the camera
};

struct RemasterSettings {
    bool enabled = false;           // video.graphics = remaster
    float bloom = 0.06f;            // video.bloom, percent / 100
    float exposure = 1.0f;          // video.exposure, percent / 100
    bool tonemap = true;            // video.tonemap
    bool shadows = true;            // video.shadows
    int shadowSize = 2048;          // video.shadow_size: texels of each shadow map
    float sun = 1.0f;               // video.sun, percent / 100: sunlight
    float sky = 1.0f;               // video.sky, percent / 100: skylight

    void Load();
};

// The scene fragment shader's uniform block of the remaster (modern.frag).
struct RemasterUniforms {
    float eyeToShadow[4][16];   // eye space to each shadow map's uv and depth (3: dynamic)
    float cascadeEnds[4];       // eye depth where each static cascade ends; w: shadow map texel
    float sunDirection[4];      // eye space, towards the sun; w: sun on
    float sunColor[4];          // w: shadows on
    float skyColor[4];          // light from above
    float groundColor[4];       // light from below
    float upDirection[4];       // eye space
    float bakedShade[4];        // the game's baked light ramp: full shade
    float bakedLight[4];        // full light; w: ramp known
};

class Remaster {
public:
    explicit Remaster(GpuContext &gpu) : gpu(gpu) {}
    ~Remaster();

    RemasterSettings settings;
    static const SDL_GPUTextureFormat sceneFormat = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
    // Textures the scene's fragment shader samples after the game's four.
    static const int kSceneTextureCount = 1;

    bool Create();
    // Switches the remaster on or off (its targets follow EnsureTargets).
    bool SetEnabled(bool enabled);
    // (Re)creates the targets for a scene of w x h pixels drawn with samples.
    bool EnsureTargets(int w, int h, SDL_GPUSampleCount samples);
    // The scene's colour target: multisampled when samples > 1.
    SDL_GPUTexture *SceneTarget() const { return sceneMsaa != nullptr ? sceneMsaa : sceneColor; }

    // Scans a recorded frame (or part of one) for its scene.
    void FindScene(const GfxFrame &frame, RemasterScene &scene) const;
    // Starts drawing the scene: its lighting, and the sun's shadow maps from
    // its casters. complete: the frame is whole (not a part flushed early,
    // which may lack some of the stage: the static cascades then wait).
    // Runs outside render passes; uses the uniform slots 0.
    void BeginScene(SDL_GPUCommandBuffer *cmd, const GfxFrame &frame, const RemasterScene &scene, bool complete);
    // A frame presented without a scene (menus, loading).
    void FrameWithoutScene();
    // The buffers a draw's vertices are in may be gone (bounds kept for them).
    void ForgetVertexBuffers() { rangeBounds.clear(); }

    bool ShadowsOn() const { return sunOn && settings.shadows; }
    bool ShadowCastersWanted() const { return castersWanted; }
    RemasterSurface Surface(const GfxCommand &c, const GfxDrawState &s) const;
    // The scene's fragment uniforms for draws seen through view (the main
    // view or another one, such as a mirror): the lighting is in eye space.
    const RemasterUniforms &Uniforms(const GfxMatrix &view);
    // Binds the shadow maps after the game's textures (slot 4).
    void BindSceneTextures(SDL_GPURenderPass *pass);

    // Turns the drawn scene into the picture in output (a colour target of
    // the back buffer's format and sample count). Runs outside render passes.
    void Finish(SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *output, SDL_GPUSampleCount outputSamples);

private:
    static const int kCascades = 3;
    static const int kDynamic = kCascades;      // the dynamic map's layer
    static const int kLayers = kCascades + 1;

    // A shadow map's box along the light: a square of half width radius
    // around centre (light space x, y), depth from zMin over depth.
    struct ShadowBox {
        float centre[2] = {}, radius = 0.0f, zMin = 0.0f, depth = 0.0f;
        GfxMatrix worldToClip;
        GfxMatrix worldToShadow;    // to uv and depth
    };
    struct StaticCascade {
        ShadowBox box;
        bool valid = false;
        unsigned age = 0;           // frames since drawn
    };
    // A shadow caster of the frame: its draw, world matrix (billboards
    // turned to the sun) and bounding sphere.
    struct Caster {
        size_t command;
        GfxMatrix world;
        float centre[3], radius;
        bool isStatic;
    };
    struct RangeKey {
        const GfxVertexBuffer *buffer;
        uint32_t version, firstVertexByte, vertexCount;
        bool operator==(const RangeKey &o) const
        {
            return buffer == o.buffer && version == o.version && firstVertexByte == o.firstVertexByte &&
                   vertexCount == o.vertexCount;
        }
    };
    struct RangeHash {
        size_t operator()(const RangeKey &k) const
        {
            return std::hash<const void *>()(k.buffer) ^ (size_t)k.version * 0x9e3779b9u ^
                   (size_t)k.firstVertexByte * 31u ^ (size_t)k.vertexCount * 131u;
        }
    };
    struct Level {
        SDL_GPUTexture *texture = nullptr;
        int width = 0, height = 0;
    };

    void PlaceSun(const SunLight &sun);
    void FitBox(ShadowBox &box, const float centre[3], float radius) const;
    void UpdateShadows(SDL_GPUCommandBuffer *cmd, const GfxFrame &frame, const RemasterScene &scene, bool full);
    void FindCasters(const GfxFrame &frame, const RemasterScene &scene);
    void DrawCasters(SDL_GPUCommandBuffer *cmd, const GfxFrame &frame, int layer, const ShadowBox &box,
                     bool statics);
    SDL_GPURenderPass *BeginShadowPass(SDL_GPUCommandBuffer *cmd, int layer);
    SDL_GPUGraphicsPipeline *ShadowPipeline(int primitive);
    void InvalidateShadows();
    bool CreateShadowMaps(int size);

    SDL_GPUGraphicsPipeline *FullscreenPipeline(SDL_GPUShader *fragment, SDL_GPUTextureFormat format,
                                                SDL_GPUSampleCount samples, bool additive);
    void Fullscreen(SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *target, bool clear, SDL_GPUGraphicsPipeline *pipeline,
                    SDL_GPUTexture *const *sources, int sourceCount, const void *uniforms, size_t uniformSize);
    void Bloom(SDL_GPUCommandBuffer *cmd);
    void ReleaseTargets();

    GpuContext &gpu;
    SDL_GPUShader *vsFullscreen = nullptr, *fsBloomDown = nullptr, *fsBloomUp = nullptr, *fsTonemap = nullptr;
    SDL_GPUShader *vsShadow = nullptr, *fsShadow = nullptr;
    SDL_GPUSampler *linearClamp = nullptr, *shadowSampler = nullptr;
    std::unordered_map<uint64_t, SDL_GPUGraphicsPipeline *> pipelines;

    int width = 0, height = 0;
    SDL_GPUSampleCount samples = SDL_GPU_SAMPLECOUNT_1;
    SDL_GPUTexture *sceneColor = nullptr, *sceneMsaa = nullptr;
    std::vector<Level> bloom;               // half the scene's size, then halving
    SDL_GPUTexture *shadowMaps = nullptr;   // kLayers layers (one texel when shadows are off)
    int shadowMapSize = 0;

    // The sun: on while it is day in a stage (see SunLight).
    bool sunOn = false, day = false;
    int framesWithoutScene = 0;
    SunLight lastSun;
    float toSun[3] = {};                    // world space
    float lightAxes[3][3] = {};             // across (r, u) and along (L) the light
    float sunAcross[3] = {};                // towards the sun, level (billboards)

    StaticCascade cascades[kCascades];
    bool castersWanted = false;             // asked of the game for the next frame
    ShadowBox dynamicBox;
    std::vector<Caster> casters;
    std::unordered_map<RangeKey, std::array<float, 4>, RangeHash> rangeBounds;

    RemasterUniforms uniforms = {};
    GfxMatrix uniformsView;                 // the view uniforms was made for
    bool uniformsValid = false;
};

#endif
