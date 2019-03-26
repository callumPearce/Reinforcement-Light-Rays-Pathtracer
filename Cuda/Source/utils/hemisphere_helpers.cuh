#ifndef HEMISPHERE_HELPERS_H
#define HEMISPHERE_HELPERS_H

#include <glm/glm.hpp>
#include "ray.cuh"
//cuRand
#include <curand.h>
#include <curand_kernel.h>
#include "radiance_volumes_settings.h"

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

// Create a coordinate system for a unit hemisphere at the given normal
__device__
void create_normal_coordinate_system(vec3& normal, vec3& normal_T, vec3& normal_B);

// Sample a direction on a unit hemisphere
__device__ __host__
vec3 uniform_hemisphere_sample(float r1, float r2);

// // Create the transformation matrix for a unit hemisphere
__host__ __device__
mat4 create_transformation_matrix(vec3 normal, vec4 position);

// Sample a random direction in a unit hemisphere around an intersection point
__device__
vec4 sample_random_direction_around_intersection(curandState* d_rand_state, const vec3& norm, float& cos_theta);

// Convert the x and y coordinates (not normalised) to a direction of ray
__host__ __device__
vec3 convert_grid_pos_to_direction(float x, float y, vec3 position, mat4& transformation_matrix);

/*
* This function takes a point in the unit square,
* and maps it to a point on the unit hemisphere.
*
* Copyright 1994 Kenneth Chiu
*
* This code may be freely distributed and used
* for any purpose, commercial or non-commercial,
* as long as attribution is maintained.
*/
__host__ __device__
void map(float x, float y, float& x_ret, float& y_ret, float& z_ret);

#endif 