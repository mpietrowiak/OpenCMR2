// Direct3D 7's lighting, shared by the modern renderer's shaders: the vertex
// shader evaluates it per vertex (video.lighting = vertex, as the original),
// the fragment shader per pixel (video.lighting = pixel). The includer
// defines LIGHTING_SET: 1 in a vertex shader, 3 in a fragment shader.

struct Light {
    vec4 diffuse;
    vec4 specular;
    vec4 ambient;
    vec4 position;      // eye space; w: type (1 point, 2 spot, 3 directional)
    vec4 direction;     // eye space; w: range
    vec4 attenuation;   // a0, a1, a2, falloff
    vec4 spot;          // cos(theta / 2), cos(phi / 2), enabled, -
};

// Materials and lights: changes with the object drawn.
layout(std140, set = LIGHTING_SET, binding = 1) uniform Lighting {
    vec4 materialDiffuse;
    vec4 materialAmbient;
    vec4 materialSpecular;
    vec4 materialEmissive;
    vec4 globalAmbient;     // w: material power
    Light lights[8];
    ivec4 lightFlags;       // lighting, colour vertex, specular, local viewer
    ivec4 sources;          // diffuse, specular, ambient, emissive (0 material, 1 colour 1, 2 colour 2)
    ivec4 fogFlags;         // vertex fog mode (3 linear) when fog is on, normalise normals, per-pixel, -
    vec4 fogRange;          // vertex fog start, end, -, -
};

vec4 MaterialColor(int source, vec4 material, vec4 color1, vec4 color2)
{
    if (lightFlags.y != 0) {
        if (source == 1)
            return color1;
        if (source == 2)
            return color2;
    }
    return material;
}

// The lit diffuse and specular colours at eye-space position eye with normal
// n, from the vertex colours color1 (diffuse) and color2 (specular).
void ComputeLighting(vec3 eye, vec3 n, vec4 color1, vec4 color2, out vec4 diffuse, out vec4 specular)
{
    vec4 matDiffuse = MaterialColor(sources.x, materialDiffuse, color1, color2);
    vec4 matSpecular = MaterialColor(sources.y, materialSpecular, color1, color2);
    vec4 matAmbient = MaterialColor(sources.z, materialAmbient, color1, color2);
    vec4 matEmissive = MaterialColor(sources.w, materialEmissive, color1, color2);
    vec3 ambient = globalAmbient.rgb;
    vec3 diffuseSum = vec3(0.0);
    vec3 specularSum = vec3(0.0);
    vec3 toEye = lightFlags.w != 0 && dot(eye, eye) > 0.0 ? normalize(-eye) : vec3(0.0, 0.0, -1.0);
    for (int i = 0; i < 8; i++) {
        Light l = lights[i];
        if (l.spot.z == 0.0)
            continue;
        int type = int(l.position.w);
        vec3 L;
        float atten = 1.0;
        float spotFactor = 1.0;
        if (type == 3) {
            L = normalize(-l.direction.xyz);
        } else {
            vec3 toLight = l.position.xyz - eye;
            float d = length(toLight);
            if (d > l.direction.w)
                continue;
            L = d > 0.0 ? toLight / d : vec3(0.0, 0.0, 1.0);
            float denom = l.attenuation.x + l.attenuation.y * d + l.attenuation.z * d * d;
            atten = denom > 0.0 ? 1.0 / denom : 1.0;
            if (type == 2) {
                float rho = dot(-L, normalize(l.direction.xyz));
                if (rho <= l.spot.y)
                    spotFactor = 0.0;
                else if (rho < l.spot.x)
                    spotFactor = pow((rho - l.spot.y) / max(l.spot.x - l.spot.y, 1e-6), l.attenuation.w);
            }
        }
        float scale = atten * spotFactor;
        ambient += l.ambient.rgb * scale;
        float nDotL = max(dot(n, L), 0.0);
        diffuseSum += l.diffuse.rgb * nDotL * scale;
        if (lightFlags.z != 0 && nDotL > 0.0) {
            // pow(0, 0) and a zero half vector are undefined (NaN on some GPUs).
            vec3 h = toEye + L;
            float nDotH = dot(h, h) > 1e-8 ? max(dot(n, normalize(h)), 0.0) : 0.0;
            specularSum += l.specular.rgb * (globalAmbient.w > 0.0 ? pow(nDotH, globalAmbient.w) : 1.0) * scale;
        }
    }
    diffuse = vec4(clamp(matEmissive.rgb + ambient * matAmbient.rgb + diffuseSum * matDiffuse.rgb, 0.0, 1.0),
                   matDiffuse.a);
    specular = vec4(clamp(specularSum * matSpecular.rgb, 0.0, 1.0), color2.a);
}
