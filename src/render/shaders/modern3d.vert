#version 450
#extension GL_GOOGLE_include_directive : require
#include "modern_common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inDiffuse;    // B, G, R, A bytes
layout(location = 3) in vec4 inSpecular;
layout(location = 4) in vec2 inTex0;
layout(location = 5) in vec2 inTex1;

layout(location = 0) out vec4 outDiffuse;
layout(location = 1) out vec4 outSpecular;
layout(location = 2) out vec4 outTex0;
layout(location = 3) out vec4 outTex1;
layout(location = 4) out float outFog;
layout(location = 5) out float outEyeDepth;
layout(location = 6) out vec3 outEyePosition;
layout(location = 7) out vec4 outNormal;     // eye space; w: its length (per-pixel lighting)

vec4 TexCoord(int stage, vec3 eyePosition, vec3 eyeNormal)
{
    int index = texCoordIndex[stage];
    int transform = texCoordIndex[stage + 2];
    vec4 tc;
    if ((index & 0xffff0000) == 0x30000) {
        // Camera-space reflection vector.
        vec3 e = normalize(-eyePosition);
        tc = vec4(2.0 * dot(e, eyeNormal) * eyeNormal - e, 1.0);
    } else {
        vec2 uv = (index & 0xffff) == 1 ? inTex1 : inTex0;
        tc = vec4(uv, 1.0, 0.0);
    }
    if (transform != 0) {
        vec4 t = textureMatrix[stage] * vec4(tc.xyz, 1.0);
        tc = vec4(t.xyz, 1.0);
    }
    return tc;
}

void main()
{
    vec4 color1 = inDiffuse.zyxw;
    vec4 color2 = inSpecular.zyxw;
    vec4 eye = worldView * vec4(inPosition, 1.0);
    vec3 n = mat3(worldView) * inNormal;
    if (fogFlags.y != 0 && dot(n, n) > 0.0)
        n = normalize(n);

    // Per-pixel lighting (fogFlags.z) gets the vertex colours as they are.
    vec4 diffuse = color1;
    vec4 specular = color2;
    if (lightFlags.x != 0 && fogFlags.z == 0)
        ComputeLighting(eye.xyz, n, color1, color2, diffuse, specular);

    outDiffuse = diffuse;
    outSpecular = specular;
    outTex0 = TexCoord(0, eye.xyz, n);
    outTex1 = TexCoord(1, eye.xyz, n);
    outEyeDepth = eye.z;
    outEyePosition = eye.xyz;
    outNormal = vec4(n, length(n));
    outFog = 1.0;
    if (fogFlags.x == 3) {
        float range = fogRange.y - fogRange.x;
        outFog = range != 0.0 ? clamp((fogRange.y - eye.z) / range, 0.0, 1.0) : 1.0;
    }

    vec4 clip = projection * eye;
    clip.xy += target.zw * clip.w;
    gl_Position = clip;
}
