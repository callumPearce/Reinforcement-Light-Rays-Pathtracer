#ifndef HEMISPHERE_HELPERS_H
#define HEMISPHERE_HELPERS_H

#include <glm/glm.hpp>

using namespace std;
using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

// Create a coordinate system for a unit hemisphere at the given normal
void create_normal_coordinate_system(vec3& normal, vec3& normal_T, vec3& normal_B);

// Sample a direction on a unit hemisphere
vec3 uniform_hemisphere_sample(float r1, float r2);

// Create the transformation matrix for a unit hemisphere
mat4 create_transformation_matrix(vec3 normal, vec4 position);

#endif 