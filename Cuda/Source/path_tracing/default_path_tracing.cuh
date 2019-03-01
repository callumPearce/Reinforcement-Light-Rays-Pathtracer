#ifndef DEFAULT_PATH_TRACING_H
#define DEFAULT_PATH_TRACING_H

#include <glm/glm.hpp>
#include <string>
#include <memory>

#include "ray.cuh"
#include "surface.cuh"
#include "area_light.cuh"
#include "monte_carlo_settings.h"
#include "hemisphere_helpers.cuh"
#include "camera.cuh"
#include "image_settings.h"
#include "printing.h"

#include <curand.h>
#include <curand_kernel.h>

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Traces the path of a ray following monte carlo path tracing algorithm:
    https://en.wikipedia.org/wiki/Path_tracing
*/

__global__
void draw_default_path_tracing(vec3* device_buffer, curandState* d_rand_state, Camera camera, AreaLight* light_planes, Surface* surfaces, int light_plane_count, int surfaces_count);

__device__
vec3 path_trace(curandState* d_rand_state, Camera camera, int pixel_x, int pixel_y, Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count);

__device__
vec3 path_trace_recursive(curandState* d_rand_state, Ray ray, Surface* surfaces, AreaLight* light_planes, int bounces, int light_plane_count, int surfaces_count);

__device__
vec3 indirect_irradiance(curandState* d_rand_state, const Ray incident_ray, Surface* surfaces, AreaLight* light_planes, int bounces, int light_plane_count, int surfaces_count);

#endif