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

// Compute the barycentric coordinates for point P in the triangle defined by
// v0, v1, v2 vertices and fill in the two barycentric coordinates to u, v
// return true if the point lies within the triangle
bool compute_barycentric(vec4 v0, vec4 v1, vec4 v2, vec4 P, float& u, float& v);

#endif