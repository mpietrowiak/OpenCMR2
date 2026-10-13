// Shared by the remaster post-processing shaders (gfx_remaster.cpp). The
// scene target holds the game's colours as it made them (gamma encoded, 1.0
// is the brightest it knew) and, from the remaster lighting, brighter ones.

vec3 ToLinear(vec3 c)
{
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

vec3 ToGamma(vec3 c)
{
    return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2));
}

float Luminance(vec3 c)
{
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// NaN and infinity (a broken pixel of the scene) as black, and the rest
// within what half floats hold.
vec3 Finite(vec3 c)
{
    bvec3 bad = bvec3(isnan(c.r) || isinf(c.r), isnan(c.g) || isinf(c.g), isnan(c.b) || isinf(c.b));
    return clamp(mix(c, vec3(0.0), bad), vec3(-65504.0), vec3(65504.0));
}
