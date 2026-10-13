#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"
// The scene as shown: bloom added in linear light, then the Khronos PBR
// Neutral tone mapper, which leaves colours up to about 0.8 (linear) as they
// are and rolls brighter ones off smoothly, so the original picture keeps
// its colours and the remaster's highlights do not clip.

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(set = 2, binding = 0) uniform sampler2D scene;
layout(set = 2, binding = 1) uniform sampler2D bloom;

layout(std140, set = 3, binding = 0) uniform Tonemap {
    vec4 params;        // bloom intensity, exposure, tone mapping on, -
};

vec3 Neutral(vec3 color)
{
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression)
        return color + offset;
    const float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3(newPeak), g) + offset;
}

void main()
{
    vec3 c = ToLinear(Finite(texture(scene, inUv).rgb)) * params.y;
    c += Finite(texture(bloom, inUv).rgb) * params.x;
    if (params.z != 0.0)
        c = Neutral(c);
    outColor = vec4(ToGamma(c), 1.0);
}
