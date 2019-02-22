#include "interpolation.h"

// Trilinearly interpolate the values between a and b to a point
// which c = a(1 - t) + bt, 0 <= t <= 1
vec3 trilinear_interpolate(vec3 a, vec3 b, float t){

    // Interpolate x
}

// 4 radiance volumes:
// Interpolate between the first two and the last two
// Interpolate between these values to give a colour value between the four of them