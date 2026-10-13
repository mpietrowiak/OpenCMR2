#version 450
// The remaster's sun shadow maps (gfx_remaster.cpp): the scene's solid draws
// again, transformed into a shadow map.

layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec4 inDiffuse;    // B, G, R, A bytes
layout(location = 4) in vec2 inTex0;

layout(location = 0) out vec2 outTex0;
layout(location = 1) out float outAlpha;

layout(std140, set = 1, binding = 0) uniform Shadow {
    mat4 worldToCascade;    // the draw's world matrix times the cascade's
};

void main()
{
    outTex0 = inTex0;
    outAlpha = inDiffuse.w;
    gl_Position = worldToCascade * vec4(inPosition, 1.0);
}
