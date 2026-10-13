#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"
// One step down the bloom chain: a 13-tap filter of the level above (Jimenez,
// "Next generation post processing in Call of Duty: Advanced Warfare"). The
// first step reads the scene, takes it to linear light and keeps only what
// is above the threshold, weighting each tap by 1 / (1 + its brightness)
// (Karis): a lone very bright pixel (a sun highlight on the paint) is then
// one pixel's worth of glow, not a flash.

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(set = 2, binding = 0) uniform sampler2D source;

layout(std140, set = 3, binding = 0) uniform BloomDown {
    vec4 texel;         // 1 / source size; first step, -
    vec4 threshold;     // threshold, soft knee, -, -
};

vec3 Fetch(vec2 uv)
{
    vec3 c = Finite(texture(source, uv).rgb);
    if (texel.z != 0.0) {
        c = ToLinear(c);
        float brightness = max(c.r, max(c.g, c.b));
        float knee = threshold.x * threshold.y;
        float soft = clamp(brightness - threshold.x + knee, 0.0, 2.0 * knee);
        soft = soft * soft / (4.0 * knee + 1e-5);
        float contribution = max(soft, brightness - threshold.x) / max(brightness, 1e-5);
        c *= contribution;
    }
    return c;
}

void main()
{
    vec2 t = texel.xy;
    vec3 a = Fetch(inUv + t * vec2(-2.0, -2.0));
    vec3 b = Fetch(inUv + t * vec2(0.0, -2.0));
    vec3 c = Fetch(inUv + t * vec2(2.0, -2.0));
    vec3 d = Fetch(inUv + t * vec2(-2.0, 0.0));
    vec3 e = Fetch(inUv);
    vec3 f = Fetch(inUv + t * vec2(2.0, 0.0));
    vec3 g = Fetch(inUv + t * vec2(-2.0, 2.0));
    vec3 h = Fetch(inUv + t * vec2(0.0, 2.0));
    vec3 i = Fetch(inUv + t * vec2(2.0, 2.0));
    vec3 j = Fetch(inUv + t * vec2(-1.0, -1.0));
    vec3 k = Fetch(inUv + t * vec2(1.0, -1.0));
    vec3 l = Fetch(inUv + t * vec2(-1.0, 1.0));
    vec3 m = Fetch(inUv + t * vec2(1.0, 1.0));
    if (texel.z == 0.0) {
        vec3 color = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
        outColor = vec4(color, 1.0);
        return;
    }
    // The same filter as five 2x2 boxes, each weighted by its brightness.
    vec3 boxes[5] = vec3[](
        (j + k + l + m) * 0.25, (a + b + d + e) * 0.25, (b + c + e + f) * 0.25, (d + e + g + h) * 0.25,
        (e + f + h + i) * 0.25);
    const float weights[5] = float[](0.5, 0.125, 0.125, 0.125, 0.125);
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int n = 0; n < 5; n++) {
        float w = weights[n] / (1.0 + Luminance(boxes[n]));
        sum += boxes[n] * w;
        total += w;
    }
    outColor = vec4(sum / total, 1.0);
}
