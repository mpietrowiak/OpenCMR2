// Shared by the modern renderer's vertex shaders (gfx_modern.cpp). The
// uniforms are split by how often they change, and each block is pushed
// only when it does. Matrices are Direct3D's row-vector matrices uploaded as
// they are, so in GLSL they act transposed: v * M (D3D) == M_glsl * v.

// Every draw: where it goes.
layout(std140, set = 1, binding = 0) uniform Transform {
    mat4 worldView;
    mat4 projection;
    vec4 target;        // game's render target width, height; viewport half-pixel offset x, y (NDC)
};

#define LIGHTING_SET 1
#include "modern_lighting.glsl"

// Texture coordinate generation.
layout(std140, set = 1, binding = 2) uniform TexGen {
    mat4 textureMatrix[2];
    ivec4 texCoordIndex;    // stage 0, stage 1, texture transform 0, texture transform 1
};
