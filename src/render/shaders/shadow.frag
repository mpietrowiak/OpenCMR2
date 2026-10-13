#version 450
// Shadow maps: depth only, cut-outs (foliage, fences) cut as the scene cuts
// them.

layout(location = 0) in vec2 inTex0;
layout(location = 1) in float inAlpha;

layout(set = 2, binding = 0) uniform sampler2D tex0;

layout(std140, set = 3, binding = 0) uniform Cutout {
    vec4 cutout;    // alpha mode (0 none, 1 texture, 2 texture * vertex), reference, -, -
};

void main()
{
    if (cutout.x != 0.0) {
        float a = texture(tex0, inTex0).a;
        if (cutout.x == 2.0)
            a *= inAlpha;
        if (a < cutout.y)
            discard;
    }
}
