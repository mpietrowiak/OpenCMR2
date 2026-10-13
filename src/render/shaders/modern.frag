#version 450
#extension GL_GOOGLE_include_directive : require
#define LIGHTING_SET 3
#include "modern_lighting.glsl"

layout(location = 0) in vec4 inDiffuse;
layout(location = 1) in vec4 inSpecular;
layout(location = 2) in vec4 inTex0;
layout(location = 3) in vec4 inTex1;
layout(location = 4) in float inFog;
layout(location = 5) in float inEyeDepth;
layout(location = 6) in vec3 inEyePosition;
layout(location = 7) in vec4 inNormal;

layout(location = 0) out vec4 outColor;

layout(set = 2, binding = 0) uniform sampler2D tex0;
layout(set = 2, binding = 1) uniform samplerCube cube0;
layout(set = 2, binding = 2) uniform sampler2D tex1;
layout(set = 2, binding = 3) uniform samplerCube cube1;
// The remaster's sun shadow maps (gfx_remaster.h): three static cascades,
// then the dynamic map.
layout(set = 2, binding = 4) uniform sampler2DArrayShadow shadowMaps;

layout(std140, set = 3, binding = 0) uniform FragmentUniforms {
    ivec4 stage0;       // colour op, colour arg1, colour arg2, alpha op
    ivec4 stage0b;      // alpha arg1, alpha arg2, texture kind (0 none, 1 2D, 2 cube, 3 bump), -
    ivec4 stage1;
    ivec4 stage1b;
    vec4 textureFactor;
    ivec4 alphaTest;    // enabled, compare function, alpha to coverage, per-pixel lighting
    vec4 alphaRef;      // reference / 255
    vec4 fogColor;
    ivec4 fogMode;      // fog enabled, table mode (3 linear), specular enabled, -
    vec4 fogParams;     // start, end
    vec4 bumpMatrix;    // m00, m01, m10, m11 (of the bump stage)
    vec4 bumpLuminance; // scale, offset
    ivec4 remaster;     // surface (RemasterSurface in gfx_remaster.h), -, -, -
};

// The remaster's lighting of the frame (gfx_remaster.h, RemasterUniforms).
layout(std140, set = 3, binding = 2) uniform Remaster {
    mat4 eyeToShadow[4];    // to each shadow map's uv and depth (3: dynamic)
    vec4 cascadeEnds;   // eye depth where each static cascade ends; w: shadow map texel
    vec4 sunDirection;  // eye space, towards the sun; w: sun on
    vec4 sunColor;      // w: shadows on
    vec4 skyColor;      // light from above
    vec4 groundColor;   // light from below
    vec4 upDirection;   // eye space
    vec4 bakedShade;    // the game's baked light ramp: full shade
    vec4 bakedLight;    // full light; w: ramp known
};

const int REMASTER_BAKED = 1, REMASTER_LIT = 2, REMASTER_REFLECTION = 3, REMASTER_FOLIAGE = 4;

const int DYNAMIC_MAP = 3;

// One shadow map at eye-space point p: 3x3 hardware-filtered taps, soft
// edges that hold still from frame to frame. Outside it, lit.
float MapShadow(int layer, vec3 p)
{
    vec4 s = eyeToShadow[layer] * vec4(p, 1.0);
    if (any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0))) || s.z > 1.0)
        return 1.0;
    float sum = 0.0;
    for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++)
            sum += texture(shadowMaps, vec4(s.xy + vec2(x, y) * cascadeEnds.w, float(layer), s.z));
    return sum / 9.0;
}

// How much sun reaches eye-space point p of a surface facing n.
float SunShadow(vec3 p, vec3 n)
{
    if (sunColor.w == 0.0)
        return 1.0;
    float z = p.z;
    // Off the surface, more far away where the texels are bigger and more
    // where the sun grazes it (the shadow map's depth changes fastest there,
    // and the surface would shadow itself in patches that crawl as the view
    // moves).
    float grazing = 1.0 - clamp(dot(n, sunDirection.xyz), 0.0, 1.0);
    vec3 q = p + n * (0.05 + z * 0.004) * (1.0 + 2.0 * grazing);
    // The stage's, from the static cascade at this depth (blended into the
    // next over their overlap), and what moves, from the dynamic map.
    float stage = 1.0;
    for (int c = 0; c < 3; c++) {
        if (z < cascadeEnds[c]) {
            stage = MapShadow(c, q);
            float blendStart = cascadeEnds[c] * 0.85;
            if (z > blendStart)
                stage = mix(stage, c < 2 ? MapShadow(c + 1, q) : 1.0,
                            (z - blendStart) / (cascadeEnds[c] - blendStart));
            break;
        }
    }
    return min(stage, MapShadow(DYNAMIC_MAP, q));
}

// A stage colour without the light the game baked into it: colours on its
// ramp (ambient + light x level, the level from the vertex normals and the
// stage's baked shadows) are the stage in full light. Others are kept.
vec3 Unbaked(vec3 c)
{
    if (bakedLight.w == 0.0)
        return c;
    vec3 range = bakedLight.rgb - bakedShade.rgb;
    float t = clamp(dot(c - bakedShade.rgb, range) / dot(range, range), 0.0, 1.0);
    vec3 off = abs(c - (bakedShade.rgb + range * t));
    return max(off.r, max(off.g, off.b)) < 3.0 / 255.0 ? bakedLight.rgb : c;
}

// Light from the sky: its colour from above, the ground's from below.
vec3 SkyLight(vec3 n)
{
    return mix(groundColor.rgb, skyColor.rgb, dot(n, upDirection.xyz) * 0.5 + 0.5);
}

const int TOP_DISABLE = 1, TOP_SELECTARG1 = 2, TOP_SELECTARG2 = 3, TOP_MODULATE = 4, TOP_MODULATE2X = 5,
          TOP_MODULATE4X = 6, TOP_ADD = 7, TOP_ADDSIGNED = 8, TOP_ADDSIGNED2X = 9, TOP_SUBTRACT = 10,
          TOP_BUMPENVMAP = 22, TOP_BUMPENVMAPLUMINANCE = 23;

// The diffuse and specular colours: interpolated, or lit here.
vec4 diffuse, specular;

vec4 Arg(int arg, vec4 current, vec4 texel)
{
    vec4 v;
    switch (arg & 0xf) {
    case 0: v = diffuse; break;
    case 1: v = current; break;
    case 2: v = texel; break;
    case 3: v = textureFactor; break;
    case 4: v = specular; break;
    default: v = current; break;
    }
    if ((arg & 0x10) != 0)
        v = vec4(1.0) - v;
    if ((arg & 0x20) != 0)
        v = v.aaaa;
    return v;
}

vec4 Op(int op, vec4 a, vec4 b)
{
    switch (op) {
    case TOP_SELECTARG1: return a;
    case TOP_SELECTARG2: return b;
    case TOP_MODULATE: return a * b;
    case TOP_MODULATE2X: return clamp(a * b * 2.0, 0.0, 1.0);
    case TOP_MODULATE4X: return clamp(a * b * 4.0, 0.0, 1.0);
    case TOP_ADD: return clamp(a + b, 0.0, 1.0);
    case TOP_ADDSIGNED: return clamp(a + b - 0.5, 0.0, 1.0);
    case TOP_ADDSIGNED2X: return clamp((a + b - 0.5) * 2.0, 0.0, 1.0);
    case TOP_SUBTRACT: return clamp(a - b, 0.0, 1.0);
    default: return a;
    }
}

vec4 Sample(int kind, sampler2D t2, samplerCube tc, vec4 coord)
{
    if (kind == 2)
        return texture(tc, coord.xyz);
    if (kind == 0)
        return vec4(1.0);
    return texture(t2, coord.xy);
}

float SignedByte(float v)
{
    float b = floor(v * 255.0 + 0.5);
    return (b >= 128.0 ? b - 256.0 : b) / 128.0;
}

bool Compare(int func, float a, float b)
{
    switch (func) {
    case 1: return false;
    case 2: return a < b;
    case 3: return a == b;
    case 4: return a <= b;
    case 5: return a > b;
    case 6: return a != b;
    case 7: return a >= b;
    default: return true;
    }
}

void main()
{
    diffuse = inDiffuse;
    specular = inSpecular;
    if (alphaTest.w != 0) {
        // The interpolated normal, at the length the vertices gave it (the
        // fixed-function lighting does not normalise unless asked to).
        vec3 n = dot(inNormal.xyz, inNormal.xyz) > 0.0 ? normalize(inNormal.xyz) * inNormal.w : vec3(0.0);
        ComputeLighting(inEyePosition, n, inDiffuse, inSpecular, diffuse, specular);
    }

    // The remaster's sun and sky.
    int surface = remaster.x;
    bool addSpecular = fogMode.z != 0;
    vec3 n = dot(inNormal.xyz, inNormal.xyz) > 0.0 ? normalize(inNormal.xyz) : upDirection.xyz;
    if (surface == REMASTER_BAKED) {
        // The stages' normals are smoothed across creases: the slope of a
        // bank bends the normals of the road along it. Where the normal
        // strays far from the triangle's own, the triangle's is taken.
        vec3 face = cross(dFdx(inEyePosition), dFdy(inEyePosition));
        if (dot(face, face) > 1e-12) {
            face = normalize(face);
            if (dot(face, n) < 0.0)
                face = -face;
            n = normalize(mix(face, n, smoothstep(0.80, 0.95, dot(face, n))));
        }
    }
    // Guarded: a NaN here would spread over the screen through the bloom.
    vec3 toEye = dot(inEyePosition, inEyePosition) > 0.0 ? normalize(-inEyePosition) : vec3(0.0, 0.0, -1.0);
    // The back of a two-sided surface is lit from its side. The winding is
    // not enough to tell: some of the stages' ground is wound the other way,
    // its normals still up; only a normal facing away is turned.
    if (!gl_FrontFacing && surface != REMASTER_FOLIAGE && dot(n, toEye) < 0.0)
        n = -n;
    if (surface == REMASTER_BAKED || surface == REMASTER_FOLIAGE) {
        // The vertex colour is the surface's tint. What the game's lights
        // add to it (headlights) is kept.
        vec3 light;
        if (surface == REMASTER_FOLIAGE) {
            // Billboards face the camera: lit as a loose volume of leaves.
            // Looked up towards the sun, past its own shadow (a quad facing the sun).
            light = SkyLight(upDirection.xyz) +
                    sunColor.rgb * 0.6 * SunShadow(inEyePosition + sunDirection.xyz * 2.5, sunDirection.xyz);
        } else {
            float nDotL = max(dot(n, sunDirection.xyz), 0.0);
            float sun = nDotL > 0.0 ? nDotL * SunShadow(inEyePosition, n) : 0.0;
            light = SkyLight(n) + sunColor.rgb * sun;
        }
        diffuse.rgb = Unbaked(inDiffuse.rgb) * light + max(diffuse.rgb - inDiffuse.rgb, vec3(0.0));
    } else if (surface == REMASTER_LIT) {
        // Paint: the material lit by sun and sky, a sharp highlight of the sun.
        vec4 base = MaterialColor(sources.x, materialDiffuse, inDiffuse, inSpecular);
        vec4 emissive = MaterialColor(sources.w, materialEmissive, inDiffuse, inSpecular);
        float nDotL = max(dot(n, sunDirection.xyz), 0.0);
        float shadow = nDotL > 0.0 ? SunShadow(inEyePosition, n) : 0.0;
        diffuse = vec4(emissive.rgb + base.rgb * (SkyLight(n) + sunColor.rgb * nDotL * shadow), base.a);
        vec3 h = sunDirection.xyz + toEye;
        h = dot(h, h) > 1e-8 ? normalize(h) : n;
        float fresnel = 0.04 + 0.96 * pow(clamp(1.0 - dot(h, toEye), 0.0, 1.0), 5.0);
        float nDotH = clamp(dot(n, h), 0.0, 1.0);
        float highlight = pow(nDotH, 80.0) * 6.0 + pow(nDotH, 12.0) * 0.25;
        // Capped: at grazing angles the highlight would reach hundreds of
        // times white in linear light, and one pixel of it flashes the bloom.
        specular = vec4(min(sunColor.rgb * highlight * fresnel * 4.0 * nDotL * shadow, vec3(3.0)), 1.0);
        addSpecular = true;
    }

    vec4 current = diffuse;
    vec2 bumpOffset = vec2(0.0);
    float luminance = 1.0;

    // Stage 0.
    if (stage0.x != TOP_DISABLE) {
        if (stage0.x == TOP_BUMPENVMAP || stage0.x == TOP_BUMPENVMAPLUMINANCE) {
            vec4 bump = texture(tex0, inTex0.xy);
            float du = SignedByte(bump.r), dv = SignedByte(bump.g);
            bumpOffset = vec2(du * bumpMatrix.x + dv * bumpMatrix.z, du * bumpMatrix.y + dv * bumpMatrix.w);
            if (stage0.x == TOP_BUMPENVMAPLUMINANCE)
                luminance = clamp(bump.b * bumpLuminance.x + bumpLuminance.y, 0.0, 1.0);
        } else {
            vec4 texel = Sample(stage0b.z, tex0, cube0, inTex0);
            vec4 color = Op(stage0.x, Arg(stage0.y, current, texel), Arg(stage0.z, current, texel));
            float alpha = stage0.w == TOP_DISABLE ? current.a
                                                  : Op(stage0.w, Arg(stage0b.x, current, texel), Arg(stage0b.y, current, texel)).a;
            current = vec4(color.rgb, alpha);
        }
        // Stage 1.
        if (stage1.x != TOP_DISABLE) {
            vec4 coord = inTex1 + vec4(bumpOffset, 0.0, 0.0);
            vec4 texel = Sample(stage1b.z, tex1, cube1, coord);
            texel.rgb *= luminance;
            vec4 color = Op(stage1.x, Arg(stage1.y, current, texel), Arg(stage1.z, current, texel));
            float alpha = stage1.w == TOP_DISABLE ? current.a
                                                  : Op(stage1.w, Arg(stage1b.x, current, texel), Arg(stage1b.y, current, texel)).a;
            current = vec4(color.rgb, alpha);
        }
    }

    if (surface == REMASTER_REFLECTION) {
        // The environment shows on the paint at grazing angles (Fresnel), and
        // only faintly where it faces the camera.
        float facing = clamp(dot(n, toEye), 0.0, 1.0);
        current.a *= mix(0.12, 1.0, pow(1.0 - facing, 4.0));
    }

    if (addSpecular)
        current.rgb = surface != 0 ? current.rgb + specular.rgb : clamp(current.rgb + specular.rgb, 0.0, 1.0);

    if (alphaTest.x != 0) {
        // With alpha to coverage, a "greater" test becomes a ramp of coverage
        // one pixel wide across the edge instead of a hard cut.
        if (alphaTest.z != 0 && (alphaTest.y == 5 || alphaTest.y == 7)) {
            float edge = max(fwidth(current.a), 1.0 / 255.0);
            current.a = clamp((current.a - alphaRef.x) / edge + 0.5, 0.0, 1.0);
        } else if (!Compare(alphaTest.y, floor(current.a * 255.0 + 0.5), floor(alphaRef.x * 255.0 + 0.5))) {
            discard;
        }
    }

    if (fogMode.x != 0) {
        float f = inFog;
        if (fogMode.y == 3) {
            float range = fogParams.y - fogParams.x;
            f = range != 0.0 ? clamp((fogParams.y - inEyeDepth) / range, 0.0, 1.0) : 1.0;
        }
        current.rgb = mix(fogColor.rgb, current.rgb, f);
    }
    outColor = current;
}
