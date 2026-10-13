#version 450
// One step up the bloom chain: a 3x3 tent filter of the level below, added
// to this level (the pipeline blends additively).

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(set = 2, binding = 0) uniform sampler2D source;

layout(std140, set = 3, binding = 0) uniform BloomUp {
    vec4 texel;         // 1 / source size, radius, -
};

void main()
{
    vec2 t = texel.xy * texel.z;
    vec3 c = texture(source, inUv).rgb * 4.0;
    c += (texture(source, inUv + vec2(-t.x, 0.0)).rgb + texture(source, inUv + vec2(t.x, 0.0)).rgb +
          texture(source, inUv + vec2(0.0, -t.y)).rgb + texture(source, inUv + vec2(0.0, t.y)).rgb) * 2.0;
    c += texture(source, inUv - t).rgb + texture(source, inUv + t).rgb +
         texture(source, inUv + vec2(-t.x, t.y)).rgb + texture(source, inUv + vec2(t.x, -t.y)).rgb;
    outColor = vec4(c / 16.0, 1.0);
}
