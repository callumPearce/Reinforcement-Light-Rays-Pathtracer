#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <vector>
#include <glm/glm.hpp>
#include "radiance_volume.h"
#include "ray.h"
#include "radiance_tree.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

// Trilinearly interpolate the values between a and b to a point
// which c = a(1 - t) + bt, 0 <= t <= 1
vec3 trilinear_interpolate(vec3 a, vec3 b, float t);

#endif