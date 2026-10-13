#version 450
// A triangle covering the target; uv (0, 0) at the top left.

layout(location = 0) out vec2 outUv;

void main()
{
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    outUv = p;
    gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
}
