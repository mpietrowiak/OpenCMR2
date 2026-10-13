// The remaster renderer: see gfx_remaster.h.

#include "render/gfx_remaster.h"

#include "platform/platform.h"
#include "port/sys.h"

#include "shaders_spirv.h"

#ifdef OPENCMR2_SHADERS_MSL
#define SHADER_CODE(name) g_##name, sizeof(g_##name), g_##name##_msl, sizeof(g_##name##_msl)
#else
#define SHADER_CODE(name) g_##name, sizeof(g_##name), nullptr, 0
#endif

#include <math.h>
#include <string.h>

#include <algorithm>

namespace {

// Bloom levels below the scene: half its size down to 1/64.
const int kBloomLevels = 6;

// Eye depth (metres) where each static cascade ends; each begins where the
// one before is 85% done, and the two are blended over the overlap.
const float kCascadeEnds[3] = { 16.0f, 60.0f, 220.0f };
// How much larger than its part of the view a static cascade is drawn, so
// it lasts while the view moves within it.
const float kCascadeMargin = 1.3f;
// Frames after which a static cascade is drawn again anyway, for what the
// game began drawing since (sectors coming into view); staggered.
const unsigned kCascadeRefresh = 120;
// A static cascade is drawn again once the view is this close to its edge
// (fraction of the view part's radius), with the stage the game culls, which
// is asked for from a little earlier.
const float kCascadeSlackDue = 0.25f, kCascadeSlackWanted = 0.35f;
// Half width (metres) of the dynamic map around the camera.
const float kDynamicReach = 80.0f;
// How far towards the sun (metres) beyond its box a map takes casters in.
const float kCasterReach = 300.0f;

struct BloomDownUniforms {
    float texel[4];
    float threshold[4];
};

struct BloomUpUniforms {
    float texel[4];
};

struct TonemapUniforms {
    float params[4];
};

// How the shadow pass cuts a caster out with its texture's alpha.
struct CutoutUniforms {
    float cutout[4];    // mode (0 none, 1 texture, 2 texture x vertex), reference, -, -
};

GfxMatrix Identity()
{
    GfxMatrix m;
    memset(&m, 0, sizeof(m));
    m._11 = m._22 = m._33 = m._44 = 1.0f;
    return m;
}

GfxMatrix Multiply(const GfxMatrix &a, const GfxMatrix &b)
{
    GfxMatrix out;
    Gpu_Multiply(a, b, &out._11);
    return out;
}

// General 4x4 inverse (Gauss-Jordan with partial pivoting).
bool Invert(const GfxMatrix &in, GfxMatrix &out)
{
    double a[4][8];
    const float *p = &in._11;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            a[r][c] = c < 4 ? p[r * 4 + c] : (c - 4 == r ? 1.0 : 0.0);
    for (int c = 0; c < 4; c++) {
        int pivot = c;
        for (int r = c + 1; r < 4; r++)
            if (fabs(a[r][c]) > fabs(a[pivot][c]))
                pivot = r;
        if (fabs(a[pivot][c]) < 1e-12)
            return false;
        for (int k = 0; k < 8; k++)
            std::swap(a[c][k], a[pivot][k]);
        double inv = 1.0 / a[c][c];
        for (int k = 0; k < 8; k++)
            a[c][k] *= inv;
        for (int r = 0; r < 4; r++)
            if (r != c) {
                double f = a[r][c];
                for (int k = 0; k < 8; k++)
                    a[r][k] -= f * a[c][k];
            }
    }
    float *q = &out._11;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            q[r * 4 + c] = (float)a[r][c + 4];
    return true;
}

void Transform(const float v[3], float w, const GfxMatrix &m, float out[3])
{
    GfxVector vec = { v[0], v[1], v[2] };
    Gpu_TransformPoint(vec, w, m, out);
}

float Dot(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Whether a scene draw has a solid surface (one that casts shadows): drawn
// in 3D with depth written, not added light.
bool SolidSurface(const GfxCommand &c, const GfxDrawState &s)
{
    if (c.type != GFX_CMD_DRAW || c.transformed || !s.depthTest || !s.depthWrite || c.vertexCount == 0)
        return false;
    if (c.primitive != GFX_TRIANGLELIST && c.primitive != GFX_TRIANGLESTRIP)
        return false;
    return !(s.alphaBlend && s.destBlend == GFX_BLEND_ONE);
}

// A billboard: geometry in its local XY plane that its world matrix turns
// about the vertical to face the camera (trees, bushes, spectators): as the
// game says (GFX_SHADOW_BILLBOARD), or a lone quad turned so.
bool Billboard(const GfxCommand &c, const GfxDrawState &s)
{
    const GfxMatrix &w = s.world;
    bool upright = w._12 == 0.0f && w._21 == 0.0f && w._23 == 0.0f && w._32 == 0.0f && w._22 > 0.0f;
    if ((s.shadowFlags & GFX_SHADOW_BILLBOARD) != 0)
        return upright;
    return c.vertexCount == 4 && s.alphaBlend && upright && (w._13 != 0.0f || w._31 != 0.0f);
}

// How the shadow pass cuts a caster out: as the scene cuts it out with its
// texture's alpha. Billboards always are (the game draws far ones without
// blending, which would make their shadows squares).
void FillCutout(const GfxCommand &c, const GfxDrawState &s, CutoutUniforms &u)
{
    memset(&u, 0, sizeof(u));
    bool billboard = Billboard(c, s);
    if (!s.alphaTest && !s.alphaBlend && !billboard)
        return;
    const GfxStageState &st = s.stages[0];
    if (st.texture == NULL || st.colorOp == GFX_TOP_DISABLE)
        return;
    bool texture1 = (st.alphaArg1 & 0xf) == GFX_TA_TEXTURE, texture2 = (st.alphaArg2 & 0xf) == GFX_TA_TEXTURE;
    if (st.alphaOp == GFX_TOP_MODULATE && (texture1 || texture2))
        u.cutout[0] = 2.0f;
    else if ((st.alphaOp == GFX_TOP_SELECTARG1 && texture1) || (st.alphaOp == GFX_TOP_SELECTARG2 && texture2) ||
             billboard)
        u.cutout[0] = 1.0f;
    else
        return;
    float ref = s.alphaTest && (s.alphaFunc == GFX_CMP_GREATER || s.alphaFunc == GFX_CMP_GREATEREQUAL)
                    ? (s.alphaRef & 0xff) / 255.0f
                    : 0.0f;
    // Blended edges (foliage) count as surface where mostly opaque.
    u.cutout[1] = s.alphaBlend || billboard ? std::max(ref, 0.5f) : std::max(ref, 1.0f / 255.0f);
}

// A sphere around vertices: the middle of their box, half its diagonal.
std::array<float, 4> BoundsOf(const GfxVertex *v, uint32_t count)
{
    float lo[3] = { v[0].x, v[0].y, v[0].z }, hi[3] = { v[0].x, v[0].y, v[0].z };
    for (uint32_t i = 1; i < count; i++) {
        const float p[3] = { v[i].x, v[i].y, v[i].z };
        for (int k = 0; k < 3; k++) {
            lo[k] = std::min(lo[k], p[k]);
            hi[k] = std::max(hi[k], p[k]);
        }
    }
    float dx = hi[0] - lo[0], dy = hi[1] - lo[1], dz = hi[2] - lo[2];
    return { (lo[0] + hi[0]) * 0.5f, (lo[1] + hi[1]) * 0.5f, (lo[2] + hi[2]) * 0.5f,
             0.5f * sqrtf(dx * dx + dy * dy + dz * dz) };
}

} // namespace

void RemasterSettings::Load()
{
    enabled = SDL_strcasecmp(Platform_GetSetting("video.graphics", "original"), "remaster") == 0;
    bloom = std::max(0, std::min(100, Platform_GetSettingInt("video.bloom", 6))) / 100.0f;
    exposure = std::max(25, std::min(400, Platform_GetSettingInt("video.exposure", 100))) / 100.0f;
    tonemap = Platform_GetSettingInt("video.tonemap", 1) != 0;
    shadows = Platform_GetSettingInt("video.shadows", 1) != 0;
    shadowSize = std::max(512, std::min(8192, Platform_GetSettingInt("video.shadow_size", 2048)));
    sun = std::max(0, std::min(400, Platform_GetSettingInt("video.sun", 100))) / 100.0f;
    sky = std::max(0, std::min(400, Platform_GetSettingInt("video.sky", 100))) / 100.0f;
}

Remaster::~Remaster()
{
    if (gpu.device == NULL)
        return;
    ReleaseTargets();
    SDL_ReleaseGPUTexture(gpu.device, shadowMaps);
    for (auto &p : pipelines)
        SDL_ReleaseGPUGraphicsPipeline(gpu.device, p.second);
    SDL_ReleaseGPUSampler(gpu.device, linearClamp);
    SDL_ReleaseGPUSampler(gpu.device, shadowSampler);
    for (SDL_GPUShader *shader : { vsFullscreen, fsBloomDown, fsBloomUp, fsTonemap, vsShadow, fsShadow })
        SDL_ReleaseGPUShader(gpu.device, shader);
}

bool Remaster::Create()
{
    settings.Load();
    vsFullscreen = gpu.Shader(SHADER_CODE(fullscreen_vert), SDL_GPU_SHADERSTAGE_VERTEX, 0, 0);
    fsBloomDown = gpu.Shader(SHADER_CODE(bloom_down_frag), SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 1);
    fsBloomUp = gpu.Shader(SHADER_CODE(bloom_up_frag), SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 1);
    fsTonemap = gpu.Shader(SHADER_CODE(tonemap_frag), SDL_GPU_SHADERSTAGE_FRAGMENT, 2, 1);
    vsShadow = gpu.Shader(SHADER_CODE(shadow_vert), SDL_GPU_SHADERSTAGE_VERTEX, 0, 1);
    fsShadow = gpu.Shader(SHADER_CODE(shadow_frag), SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 1);
    SDL_GPUSamplerCreateInfo info = {};
    info.min_filter = info.mag_filter = SDL_GPU_FILTER_LINEAR;
    info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    info.address_mode_u = info.address_mode_v = info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    linearClamp = SDL_CreateGPUSampler(gpu.device, &info);
    info.enable_compare = true;
    info.compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL;
    shadowSampler = SDL_CreateGPUSampler(gpu.device, &info);
    if (!(vsFullscreen && fsBloomDown && fsBloomUp && fsTonemap && vsShadow && fsShadow && linearClamp &&
          shadowSampler))
        return false;

    if (!CreateShadowMaps(settings.enabled && settings.shadows ? settings.shadowSize : 1))
        return false;
    Sys_Log("gfx: remaster (%s): sun %.2f, sky %.2f, shadows %s (%d), bloom %.2f, exposure %.2f, tone mapping %s",
            settings.enabled ? "on" : "off, Ctrl+G", settings.sun, settings.sky, settings.shadows ? "on" : "off",
            settings.shadowSize, settings.bloom, settings.exposure, settings.tonemap ? "on" : "off");
    return true;
}

// The shadow maps, cleared (one texel stands in while there are no shadows:
// the scene's shader always has them bound).
bool Remaster::CreateShadowMaps(int size)
{
    if (shadowMaps != NULL && shadowMapSize == size)
        return true;
    if (shadowMaps != NULL) {
        SDL_WaitForGPUIdle(gpu.device);
        SDL_ReleaseGPUTexture(gpu.device, shadowMaps);
    }
    shadowMaps = gpu.CreateTexture(SDL_GPU_TEXTURETYPE_2D_ARRAY, SDL_GPU_TEXTUREFORMAT_D32_FLOAT, size, size, kLayers,
                                   1, SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
    shadowMapSize = shadowMaps != NULL ? size : 0;
    if (shadowMaps == NULL)
        return false;
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(gpu.device);
    for (int layer = 0; layer < kLayers; layer++)
        SDL_EndGPURenderPass(BeginShadowPass(cmd, layer));
    SDL_SubmitGPUCommandBuffer(cmd);
    InvalidateShadows();
    return true;
}

bool Remaster::SetEnabled(bool enabled)
{
    settings.enabled = enabled;
    sunOn = day = castersWanted = false;
    framesWithoutScene = 0;
    InvalidateShadows();
    Sys_Log("gfx: remaster %s", enabled ? "on" : "off");
    if (!enabled) {
        // Its targets and shadow maps go until it is back.
        SDL_WaitForGPUIdle(gpu.device);
        ReleaseTargets();
        return CreateShadowMaps(1);
    }
    return !settings.shadows || CreateShadowMaps(settings.shadowSize);
}

void Remaster::ReleaseTargets()
{
    SDL_ReleaseGPUTexture(gpu.device, sceneColor);
    SDL_ReleaseGPUTexture(gpu.device, sceneMsaa);
    sceneColor = sceneMsaa = NULL;
    for (Level &l : bloom)
        SDL_ReleaseGPUTexture(gpu.device, l.texture);
    bloom.clear();
}

bool Remaster::EnsureTargets(int w, int h, SDL_GPUSampleCount wantedSamples)
{
    if (sceneColor != NULL && width == w && height == h && samples == wantedSamples)
        return true;
    SDL_WaitForGPUIdle(gpu.device);
    ReleaseTargets();
    width = w;
    height = h;
    samples = wantedSamples;
    sceneColor = gpu.CreateTexture(SDL_GPU_TEXTURETYPE_2D, sceneFormat, w, h, 1, 1,
                                   SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
    if (samples != SDL_GPU_SAMPLECOUNT_1)
        sceneMsaa = gpu.CreateTexture(SDL_GPU_TEXTURETYPE_2D, sceneFormat, w, h, 1, 1,
                                      SDL_GPU_TEXTUREUSAGE_COLOR_TARGET, samples);
    int lw = w, lh = h;
    for (int i = 0; i < kBloomLevels; i++) {
        lw = std::max(1, lw / 2);
        lh = std::max(1, lh / 2);
        Level l;
        l.width = lw;
        l.height = lh;
        l.texture = gpu.CreateTexture(SDL_GPU_TEXTURETYPE_2D, sceneFormat, lw, lh, 1, 1,
                                      SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
        if (l.texture == NULL)
            return false;
        bloom.push_back(l);
    }
    return sceneColor != NULL && (samples == SDL_GPU_SAMPLECOUNT_1 || sceneMsaa != NULL);
}

// ---- the scene ----

void Remaster::FindScene(const GfxFrame &frame, RemasterScene &scene) const
{
    scene = RemasterScene();
    GfxTexture *target = NULL;
    float area = -1.0f, brightest = -1.0f;
    bool stage = false;
    for (size_t i = 0; i < frame.commands.size(); i++) {
        const GfxCommand &c = frame.commands[i];
        if (c.type == GFX_CMD_TARGET)
            target = c.target;
        if (target != NULL || c.type != GFX_CMD_DRAW || c.transformed)
            continue;
        scene.last = i;
        const GfxDrawState &s = frame.states[c.state];
        stage |= (s.shadowFlags & GFX_SHADOW_STATIC) != 0;
        if (SolidSurface(c, s)) {
            scene.casters.push_back(i);
            // The main camera: the largest viewport's, the last of them (the
            // scene ends with the cars). Smaller views (a mirror) get their
            // own uniforms; the shadows are fitted to the main one.
            float a = (float)s.viewport.width * s.viewport.height;
            if (a >= area) {
                area = a;
                scene.view.projection[0] = s.projection._11;
                scene.view.projection[1] = s.projection._22;
                scene.view.projection[2] = s.projection._31;
                scene.view.projection[3] = s.projection._32;
                scene.view.view = s.view;
                scene.hasView = true;
            }
        }
        // The sun: the brightest directional light the draws are lit with
        // (the cars').
        if (!s.lighting)
            continue;
        for (int k = 0; k < 8; k++) {
            const GfxLight &l = s.lights[k];
            if (!s.lightEnabled[k] || l.type != GFX_LIGHT_DIRECTIONAL)
                continue;
            float b = std::max(l.diffuse.r, std::max(l.diffuse.g, l.diffuse.b));
            if (b <= brightest)
                continue;
            brightest = b;
            scene.sun.direction[0] = l.direction.x;
            scene.sun.direction[1] = l.direction.y;
            scene.sun.direction[2] = l.direction.z;
            // At night the game keeps the light, black.
            scene.sun.bright = b > 0.02f;
            scene.hasSun = true;
        }
    }
    // Only a stage is remastered: the menus' 3D (the cars shown) is drawn
    // as the game draws it.
    if (!stage)
        scene = RemasterScene();
}

void Remaster::BeginScene(SDL_GPUCommandBuffer *cmd, const GfxFrame &frame, const RemasterScene &scene,
                          bool complete)
{
    memset(&uniforms, 0, sizeof(uniforms));
    uniformsValid = false;
    sunOn = false;
    framesWithoutScene = 0;
    // Whether the game drew the stage it culls (it was asked to), with all
    // of the frame here.
    bool full = castersWanted && complete;
    castersWanted = false;
    // No draw lit by the sun (no car in view) keeps the last frame's.
    if (scene.hasSun) {
        lastSun = scene.sun;
        day |= scene.sun.bright;
    }
    if (!scene.hasView || !day)
        return;
    PlaceSun(lastSun);
    if (!sunOn)
        return;
    DWORD shade, light;
    if (Gfx_BakedLightRamp(shade, light)) {
        Gpu_Color(shade, uniforms.bakedShade);
        Gpu_Color(light, uniforms.bakedLight);
        // A ramp of one colour carries no light to take out.
        uniforms.bakedLight[3] = memcmp(uniforms.bakedShade, uniforms.bakedLight, 3 * sizeof(float)) != 0;
    }
    if (settings.shadows)
        UpdateShadows(cmd, frame, scene, full);
}

void Remaster::FrameWithoutScene()
{
    // A few such frames happen within a stage (a flushed frame, a pause):
    // only a run of them (a loading screen, the menus) ends it.
    if (++framesWithoutScene > 30 && day) {
        day = false;
        InvalidateShadows();
    }
}

void Remaster::PlaceSun(const SunLight &sun)
{
    float L[3] = { sun.direction[0], sun.direction[1], sun.direction[2] };
    float len = sqrtf(Dot(L, L));
    if (len < 1e-6f)
        return;
    for (float &c : L)
        c /= len;
    sunOn = true;
    // The static shadow maps were drawn along the old light.
    if (fabsf(Dot(L, lightAxes[2]) - 1.0f) > 1e-6f)
        InvalidateShadows();

    for (int k = 0; k < 3; k++)
        toSun[k] = -L[k];
    const float sunColor[3] = { 1.00f, 0.94f, 0.84f }, skyColor[3] = { 0.46f, 0.50f, 0.56f },
                groundColor[3] = { 0.30f, 0.27f, 0.23f };
    for (int k = 0; k < 3; k++) {
        uniforms.sunColor[k] = sunColor[k] * 0.95f * settings.sun;
        uniforms.skyColor[k] = skyColor[k] * settings.sky;
        uniforms.groundColor[k] = groundColor[k] * settings.sky;
    }
    uniforms.sunDirection[3] = 1.0f;
    float across = sqrtf(toSun[0] * toSun[0] + toSun[2] * toSun[2]);
    sunAcross[0] = across > 0.05f ? toSun[0] / across : 0.0f;
    sunAcross[1] = 0.0f;
    sunAcross[2] = across > 0.05f ? toSun[2] / across : 0.0f;

    // The light's basis: r and u across it, L along it.
    float upAxis[3] = { 0.0f, 1.0f, 0.0f };
    if (fabsf(L[1]) > 0.99f) {
        upAxis[0] = 1.0f;
        upAxis[1] = 0.0f;
    }
    float *r = lightAxes[0], *u = lightAxes[1];
    r[0] = upAxis[1] * L[2] - upAxis[2] * L[1];
    r[1] = upAxis[2] * L[0] - upAxis[0] * L[2];
    r[2] = upAxis[0] * L[1] - upAxis[1] * L[0];
    float rl = sqrtf(Dot(r, r));
    for (int k = 0; k < 3; k++)
        r[k] /= rl;
    u[0] = L[1] * r[2] - L[2] * r[1];
    u[1] = L[2] * r[0] - L[0] * r[2];
    u[2] = L[0] * r[1] - L[1] * r[0];
    memcpy(lightAxes[2], L, sizeof(L));
}

// A shadow map's box around a world sphere: its centre on whole texels, so
// that boxes drawn at different times rasterise the world alike and their
// shadows do not crawl.
void Remaster::FitBox(ShadowBox &box, const float centre[3], float radius) const
{
    const float *r = lightAxes[0], *u = lightAxes[1], *L = lightAxes[2];
    float texel = 2.0f * radius / settings.shadowSize;
    box.radius = radius;
    box.centre[0] = floorf(Dot(centre, r) / texel) * texel;
    box.centre[1] = floorf(Dot(centre, u) / texel) * texel;
    box.zMin = Dot(centre, L) - radius - kCasterReach;
    box.depth = 2.0f * radius + kCasterReach;
    GfxMatrix &m = box.worldToClip;
    memset(&m, 0, sizeof(m));
    m._11 = r[0] / radius;
    m._21 = r[1] / radius;
    m._31 = r[2] / radius;
    m._41 = -box.centre[0] / radius;
    m._12 = u[0] / radius;
    m._22 = u[1] / radius;
    m._32 = u[2] / radius;
    m._42 = -box.centre[1] / radius;
    m._13 = L[0] / box.depth;
    m._23 = L[1] / box.depth;
    m._33 = L[2] / box.depth;
    m._43 = -box.zMin / box.depth;
    m._44 = 1.0f;
    GfxMatrix bias = Identity();
    bias._11 = 0.5f;
    bias._22 = -0.5f;
    bias._41 = 0.5f;
    bias._42 = 0.5f;
    box.worldToShadow = Multiply(m, bias);
}

void Remaster::InvalidateShadows()
{
    for (StaticCascade &c : cascades)
        c.valid = false;
}

void Remaster::UpdateShadows(SDL_GPUCommandBuffer *cmd, const GfxFrame &frame, const RemasterScene &scene,
                             bool full)
{
    const SceneView &view = scene.view;
    GfxMatrix eyeToWorld;
    if (!Invert(view.view, eyeToWorld))
        return;
    FindCasters(frame, scene);
    bool wanted = false;
    uniforms.sunColor[3] = 1.0f;
    uniforms.cascadeEnds[3] = 1.0f / settings.shadowSize;

    for (int c = 0; c < kCascades; c++) {
        StaticCascade &cascade = cascades[c];
        uniforms.cascadeEnds[c] = kCascadeEnds[c];
        // The cascade's part of the view, and a sphere around it, its size
        // in steps so that it seldom changes.
        float n = c == 0 ? 0.1f : kCascadeEnds[c - 1] * 0.85f, f = kCascadeEnds[c];
        float corners[8][3], centre[3] = { 0, 0, 0 };
        for (int i = 0; i < 8; i++) {
            float z = (i & 4) ? f : n;
            float nx = (i & 1) ? 1.0f : -1.0f, ny = (i & 2) ? 1.0f : -1.0f;
            float eye[3] = { (nx - view.projection[2]) * z / view.projection[0],
                             (ny - view.projection[3]) * z / view.projection[1], z };
            Transform(eye, 1.0f, eyeToWorld, corners[i]);
            for (int k = 0; k < 3; k++)
                centre[k] += corners[i][k] / 8.0f;
        }
        float radius = 0.0f;
        for (int i = 0; i < 8; i++) {
            float d[3] = { corners[i][0] - centre[0], corners[i][1] - centre[1], corners[i][2] - centre[2] };
            radius = std::max(radius, sqrtf(Dot(d, d)));
        }
        radius = ceilf(radius / 4.0f) * 4.0f;
        float drawn = radius * kCascadeMargin;
        // How far the sphere is from the edge of the box drawn, across and
        // along the light (negative: past it).
        float slack = -1.0f;
        if (cascade.valid && cascade.box.radius == drawn) {
            const ShadowBox &b = cascade.box;
            float dx = fabsf(Dot(centre, lightAxes[0]) - b.centre[0]);
            float dy = fabsf(Dot(centre, lightAxes[1]) - b.centre[1]);
            float dz = fabsf(Dot(centre, lightAxes[2]) - (b.zMin + kCasterReach + b.radius));
            slack = drawn - radius - std::max(dx, std::max(dy, dz));
        }
        unsigned refresh = kCascadeRefresh + (unsigned)c * 40;
        bool due = cascade.age >= refresh || slack < kCascadeSlackDue * radius;
        // Drawn again when due and this frame has the whole stage around
        // the camera; at once, with what there is, when no longer covered.
        if (slack < 0.0f || (due && full)) {
            FitBox(cascade.box, centre, drawn);
            DrawCasters(cmd, frame, c, cascade.box, true);
            cascade.valid = true;
            // Without the stage off screen: again soon, with it.
            cascade.age = full ? 0 : refresh;
        } else {
            cascade.age++;
        }
        wanted |= cascade.age + 1 >= refresh || (slack >= 0.0f && slack < kCascadeSlackWanted * radius);
    }
    castersWanted = wanted;

    // What moves, around the camera.
    float camera[3] = { eyeToWorld._41, eyeToWorld._42, eyeToWorld._43 };
    FitBox(dynamicBox, camera, kDynamicReach);
    DrawCasters(cmd, frame, kDynamic, dynamicBox, false);
}

void Remaster::FindCasters(const GfxFrame &frame, const RemasterScene &scene)
{
    casters.clear();
    if (rangeBounds.size() > 65536)
        rangeBounds.clear();
    for (size_t i : scene.casters) {
        const GfxCommand &c = frame.commands[i];
        const GfxDrawState &s = frame.states[c.state];
        std::array<float, 4> b;
        if (c.buffer != NULL) {
            RangeKey key = { c.buffer, c.buffer->version, c.firstVertexByte, c.vertexCount };
            auto it = rangeBounds.find(key);
            if (it == rangeBounds.end()) {
                const GfxVertex *v = c.buffer->vertices.data() + c.firstVertexByte / sizeof(GfxVertex);
                it = rangeBounds.emplace(key, BoundsOf(v, c.vertexCount)).first;
            }
            b = it->second;
        } else {
            b = BoundsOf((const GfxVertex *)(frame.vertices.data() + c.firstVertexByte), c.vertexCount);
        }
        Caster k;
        k.command = i;
        k.world = s.world;
        bool billboard = Billboard(c, s);
        if (billboard && (sunAcross[0] != 0.0f || sunAcross[2] != 0.0f)) {
            // Turned to the sun (as the game turns them to the camera: local
            // z towards it, x = (z.z, 0, -z.x)), so that their shadows keep
            // their width whichever way the camera looks.
            GfxMatrix &w = k.world;
            float sx = sqrtf(w._11 * w._11 + w._13 * w._13), sz = sqrtf(w._31 * w._31 + w._33 * w._33);
            w._11 = sunAcross[2] * sx;
            w._13 = -sunAcross[0] * sx;
            w._31 = sunAcross[0] * sz;
            w._33 = sunAcross[2] * sz;
        }
        k.isStatic = (s.shadowFlags & GFX_SHADOW_STATIC) != 0;
        const float centre[3] = { b[0], b[1], b[2] };
        Transform(centre, 1.0f, k.world, k.centre);
        float scale = 0.0f;
        for (const float *row : { &k.world._11, &k.world._21, &k.world._31 })
            scale = std::max(scale, sqrtf(Dot(row, row)));
        k.radius = b[3] * scale;
        casters.push_back(k);
    }
}

void Remaster::DrawCasters(SDL_GPUCommandBuffer *cmd, const GfxFrame &frame, int layer, const ShadowBox &box,
                           bool statics)
{
    SDL_GPURenderPass *pass = BeginShadowPass(cmd, layer);
    int size = settings.shadowSize;
    SDL_GPUViewport vp = { 0.0f, 0.0f, (float)size, (float)size, 0.0f, 1.0f };
    SDL_SetGPUViewport(pass, &vp);
    SDL_Rect scissor = { 0, 0, size, size };
    SDL_SetGPUScissor(pass, &scissor);
    // Most draws share their matrix, cut-out and texture with the one
    // before (the stage's): those are pushed and bound when they change.
    SDL_GPUGraphicsPipeline *bound = NULL;
    SDL_GPUBuffer *boundVertices = NULL;
    uint32_t boundOffset = ~0u;
    bool indicesBound = false;
    const GfxMatrix *pushedWorld = NULL;
    CutoutUniforms pushedCutout;
    bool cutoutPushed = false;
    SDL_GPUTextureSamplerBinding boundTexture = { NULL, NULL };
    // The box in clip space: x, y within -1..1, z within 0..1.
    const GfxMatrix &m = box.worldToClip;
    float toClipXY = 1.0f / box.radius, toClipZ = 1.0f / box.depth;
    for (const Caster &k : casters) {
        if (k.isStatic != statics)
            continue;
        float p[3];
        Transform(k.centre, 1.0f, m, p);
        float rxy = k.radius * toClipXY, rz = k.radius * toClipZ;
        if (fabsf(p[0]) > 1.0f + rxy || fabsf(p[1]) > 1.0f + rxy || p[2] < -rz || p[2] > 1.0f + rz)
            continue;
        const GfxCommand &c = frame.commands[k.command];
        const GfxDrawState &s = frame.states[c.state];
        SDL_GPUGraphicsPipeline *pipeline = ShadowPipeline(c.primitive);
        if (pipeline == NULL)
            continue;
        if (pipeline != bound) {
            SDL_BindGPUGraphicsPipeline(pass, pipeline);
            bound = pipeline;
        }
        if (pushedWorld == NULL || memcmp(&k.world, pushedWorld, sizeof(k.world)) != 0) {
            float wm[16];
            Gpu_Multiply(k.world, m, wm);
            SDL_PushGPUVertexUniformData(cmd, 0, wm, sizeof(wm));
            pushedWorld = &k.world;
        }
        CutoutUniforms cu;
        FillCutout(c, s, cu);
        if (!cutoutPushed || memcmp(&cu, &pushedCutout, sizeof(cu)) != 0) {
            SDL_PushGPUFragmentUniformData(cmd, 0, &cu, sizeof(cu));
            pushedCutout = cu;
            cutoutPushed = true;
        }
        // Only cut-outs read their texture.
        SDL_GPUTextureSamplerBinding tb = { gpu.white2D, boundTexture.sampler };
        if (cu.cutout[0] != 0.0f) {
            GfxTexture *t = s.stages[0].texture;
            if (t != NULL && t->cube != NULL)
                t = t->cube;
            GpuTexture *g = t != NULL && !(t->flags & GFX_TEXTURE_CUBE) ? gpu.Ensure(t) : NULL;
            tb = { g != NULL ? g->texture : gpu.white2D, gpu.Sampler(s.stages[0]) };
        } else if (tb.sampler == NULL) {
            tb.sampler = gpu.Sampler(s.stages[0]);
        }
        if (tb.texture != boundTexture.texture || tb.sampler != boundTexture.sampler) {
            SDL_BindGPUFragmentSamplers(pass, 0, &tb, 1);
            boundTexture = tb;
        }
        SDL_GPUBuffer *vertices = c.buffer != NULL ? gpu.VertexBufferOf(c.buffer) : gpu.frameVertices;
        if (vertices != boundVertices || c.firstVertexByte != boundOffset) {
            SDL_GPUBufferBinding vb = { vertices, c.firstVertexByte };
            SDL_BindGPUVertexBuffers(pass, 0, &vb, 1);
            boundVertices = vertices;
            boundOffset = c.firstVertexByte;
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
    SDL_EndGPURenderPass(pass);
}

SDL_GPURenderPass *Remaster::BeginShadowPass(SDL_GPUCommandBuffer *cmd, int layer)
{
    SDL_GPUDepthStencilTargetInfo depth = {};
    depth.texture = shadowMaps;
    depth.load_op = SDL_GPU_LOADOP_CLEAR;
    depth.store_op = SDL_GPU_STOREOP_STORE;
    depth.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
    depth.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;
    depth.clear_depth = 1.0f;
    depth.layer = (Uint8)layer;
    return SDL_BeginGPURenderPass(cmd, NULL, 0, &depth);
}

SDL_GPUGraphicsPipeline *Remaster::ShadowPipeline(int primitive)
{
    uint64_t key = 0x100000000ull | (uint64_t)(primitive & 7);
    auto it = pipelines.find(key);
    if (it != pipelines.end())
        return it->second;
    SDL_GPUVertexBufferDescription vb = {};
    vb.slot = 0;
    vb.pitch = sizeof(GfxVertex);
    vb.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
    SDL_GPUVertexAttribute attrs[3] = {
        { 0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0 },
        { 2, 0, SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM, 24 },
        { 4, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, 32 },
    };
    SDL_GPUGraphicsPipelineCreateInfo info = {};
    info.vertex_shader = vsShadow;
    info.fragment_shader = fsShadow;
    info.vertex_input_state.vertex_buffer_descriptions = &vb;
    info.vertex_input_state.num_vertex_buffers = 1;
    info.vertex_input_state.vertex_attributes = attrs;
    info.vertex_input_state.num_vertex_attributes = 3;
    info.primitive_type = Gpu_PrimitiveType(primitive);
    info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    info.rasterizer_state.enable_depth_bias = true;
    info.rasterizer_state.depth_bias_constant_factor = 2.0f;
    info.rasterizer_state.depth_bias_slope_factor = 2.0f;
    info.rasterizer_state.enable_depth_clip = true;
    info.depth_stencil_state.enable_depth_test = true;
    info.depth_stencil_state.enable_depth_write = true;
    info.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL;
    info.target_info.num_color_targets = 0;
    info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    info.target_info.has_depth_stencil_target = true;
    SDL_GPUGraphicsPipeline *p = SDL_CreateGPUGraphicsPipeline(gpu.device, &info);
    if (p == NULL)
        Sys_Log("gfx: shadow pipeline: %s", SDL_GetError());
    pipelines[key] = p;
    return p;
}

RemasterSurface Remaster::Surface(const GfxCommand &c, const GfxDrawState &s) const
{
    if (c.transformed || !settings.enabled)
        return REMASTER_NONE;
    const GfxStageState &st = s.stages[0];
    if ((st.texCoordIndex & 0xffff0000) == GFX_TCI_CAMERASPACEREFLECTIONVECTOR && s.alphaBlend &&
        s.destBlend == GFX_BLEND_ONE)
        return REMASTER_REFLECTION;
    if (!sunOn || !s.depthTest)
        return REMASTER_NONE;      // night, or the sky
    if (s.alphaBlend && s.destBlend == GFX_BLEND_ONE)
        return REMASTER_NONE;      // light the game adds (glows, flares)
    bool sunLit = false;
    for (int i = 0; i < 8; i++)
        sunLit |= s.lighting && s.lightEnabled[i] && s.lights[i].type == GFX_LIGHT_DIRECTIONAL;
    if (sunLit)
        return REMASTER_LIT;
    if (Billboard(c, s))
        return REMASTER_FOLIAGE;
    bool tinted = s.lighting ? s.colorVertex && s.emissiveSource == GFX_MCS_COLOR1 : s.colorVertex;
    return tinted ? REMASTER_BAKED : REMASTER_NONE;
}

const RemasterUniforms &Remaster::Uniforms(const GfxMatrix &view)
{
    if (!sunOn || (uniformsValid && memcmp(&view, &uniformsView, sizeof(view)) == 0))
        return uniforms;
    uniformsView = view;
    uniformsValid = true;
    const float up[3] = { 0.0f, 1.0f, 0.0f };
    Transform(toSun, 0.0f, view, uniforms.sunDirection);
    Transform(up, 0.0f, view, uniforms.upDirection);
    GfxMatrix eyeToWorld;
    if (settings.shadows && Invert(view, eyeToWorld)) {
        for (int c = 0; c < kCascades; c++) {
            GfxMatrix m = Multiply(eyeToWorld, cascades[c].box.worldToShadow);
            memcpy(uniforms.eyeToShadow[c], &m, sizeof(m));
        }
        GfxMatrix m = Multiply(eyeToWorld, dynamicBox.worldToShadow);
        memcpy(uniforms.eyeToShadow[kDynamic], &m, sizeof(m));
    }
    return uniforms;
}

void Remaster::BindSceneTextures(SDL_GPURenderPass *pass)
{
    SDL_GPUTextureSamplerBinding binding = { shadowMaps, shadowSampler };
    SDL_BindGPUFragmentSamplers(pass, 4, &binding, 1);
}

// ---- the picture ----

SDL_GPUGraphicsPipeline *Remaster::FullscreenPipeline(SDL_GPUShader *fragment, SDL_GPUTextureFormat format,
                                                      SDL_GPUSampleCount sampleCount, bool additive)
{
    uint64_t key = (uint64_t)(uintptr_t)fragment << 16 | (uint64_t)format << 4 | (uint64_t)sampleCount << 1 |
                   (additive ? 1u : 0u);
    auto it = pipelines.find(key);
    if (it != pipelines.end())
        return it->second;
    SDL_GPUColorTargetDescription color = {};
    color.format = format;
    if (additive) {
        color.blend_state.enable_blend = true;
        color.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
        color.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
        color.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
        color.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
        color.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
        color.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
    }
    SDL_GPUGraphicsPipelineCreateInfo info = {};
    info.vertex_shader = vsFullscreen;
    info.fragment_shader = fragment;
    info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    info.multisample_state.sample_count = sampleCount;
    info.target_info.color_target_descriptions = &color;
    info.target_info.num_color_targets = 1;
    SDL_GPUGraphicsPipeline *p = SDL_CreateGPUGraphicsPipeline(gpu.device, &info);
    if (p == NULL)
        Sys_Log("gfx: effect pipeline: %s", SDL_GetError());
    pipelines[key] = p;
    return p;
}

void Remaster::Fullscreen(SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *target, bool clear,
                          SDL_GPUGraphicsPipeline *pipeline, SDL_GPUTexture *const *sources, int sourceCount,
                          const void *uniformData, size_t uniformSize)
{
    if (pipeline == NULL)
        return;
    SDL_GPUColorTargetInfo color = {};
    color.texture = target;
    color.load_op = clear ? SDL_GPU_LOADOP_CLEAR : SDL_GPU_LOADOP_LOAD;
    color.store_op = SDL_GPU_STOREOP_STORE;
    SDL_GPURenderPass *pass = SDL_BeginGPURenderPass(cmd, &color, 1, NULL);
    SDL_BindGPUGraphicsPipeline(pass, pipeline);
    SDL_GPUTextureSamplerBinding bindings[4];
    for (int i = 0; i < sourceCount; i++)
        bindings[i] = { sources[i], linearClamp };
    if (sourceCount > 0)
        SDL_BindGPUFragmentSamplers(pass, 0, bindings, sourceCount);
    SDL_PushGPUFragmentUniformData(cmd, 0, uniformData, (Uint32)uniformSize);
    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
}

void Remaster::Bloom(SDL_GPUCommandBuffer *cmd)
{
    // Down: scene -> level 0 (thresholded) -> level 1 -> ...
    SDL_GPUGraphicsPipeline *down = FullscreenPipeline(fsBloomDown, sceneFormat, SDL_GPU_SAMPLECOUNT_1, false);
    for (size_t i = 0; i < bloom.size(); i++) {
        SDL_GPUTexture *source = i == 0 ? sceneColor : bloom[i - 1].texture;
        int sw = i == 0 ? width : bloom[i - 1].width, sh = i == 0 ? height : bloom[i - 1].height;
        BloomDownUniforms u = {};
        u.texel[0] = 1.0f / sw;
        u.texel[1] = 1.0f / sh;
        u.texel[2] = i == 0 ? 1.0f : 0.0f;
        u.threshold[0] = 1.0f;
        u.threshold[1] = 0.5f;
        Fullscreen(cmd, bloom[i].texture, true, down, &source, 1, &u, sizeof(u));
    }
    // Up: each level adds the tent-filtered level below it.
    SDL_GPUGraphicsPipeline *up = FullscreenPipeline(fsBloomUp, sceneFormat, SDL_GPU_SAMPLECOUNT_1, true);
    for (size_t i = bloom.size() - 1; i > 0; i--) {
        BloomUpUniforms u = {};
        u.texel[0] = 1.0f / bloom[i].width;
        u.texel[1] = 1.0f / bloom[i].height;
        u.texel[2] = 1.0f;
        Fullscreen(cmd, bloom[i - 1].texture, false, up, &bloom[i].texture, 1, &u, sizeof(u));
    }
}

void Remaster::Finish(SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *output, SDL_GPUSampleCount outputSamples)
{
    if (sceneMsaa != NULL) {
        SDL_GPUColorTargetInfo resolve = {};
        resolve.texture = sceneMsaa;
        resolve.load_op = SDL_GPU_LOADOP_LOAD;
        resolve.store_op = SDL_GPU_STOREOP_RESOLVE;
        resolve.resolve_texture = sceneColor;
        SDL_EndGPURenderPass(SDL_BeginGPURenderPass(cmd, &resolve, 1, NULL));
    }
    bool bloomOn = settings.bloom > 0.0f;
    if (bloomOn)
        Bloom(cmd);
    TonemapUniforms u = {};
    u.params[0] = bloomOn ? settings.bloom : 0.0f;
    u.params[1] = settings.exposure;
    u.params[2] = settings.tonemap ? 1.0f : 0.0f;
    SDL_GPUTexture *sources[2] = { sceneColor, bloomOn ? bloom[0].texture : gpu.white2D };
    Fullscreen(cmd, output, false, FullscreenPipeline(fsTonemap, gpu.colorFormat, outputSamples, false), sources, 2,
               &u, sizeof(u));
}
