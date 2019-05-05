#ifndef DEFAULT_PATH_TRACING_H
#define DEFAULT_PATH_TRACING_H

#include <glm/glm.hpp>
#include <string>
#include <memory>

#include "ray.cuh"
#include "scene.cuh"
#include "monte_carlo_settings.h"
#include "hemisphere_helpers.cuh"
#include "camera.cuh"
#include "image_settings.h"
#include "printing.h"
#include "radiance_volumes_settings.h"
#include "hemisphere_helpers.cuh"
#include "radiance_volume.cuh"

#include <curand.h>
#include <curand_kernel.h>

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

class RadianceVolume;

/*
    Traces the path of a ray following monte carlo path tracing algorithm:
    https://en.wikipedia.org/wiki/Path_tracing
*/

__global__
void draw_default_path_tracing(vec3* device_buffer, curandState* d_rand_state, Camera* camera, Scene* scene, int* device_path_lengths);

__device__
vec3 path_trace(curandState* d_rand_state, Camera* camera, int pixel_x, int pixel_y, Scene* scene, int* device_path_lengths);

__device__
vec3 path_trace_iterative(curandState* d_rand_state, Ray ray, Scene* scene, int& path_length);

// Given a radiance volume fill the true radiance volume values inside
__global__
void path_trace_radiance_volume(curandState* d_rand_state, RadianceVolume* rv, Scene* scene);

__device__
float path_trace_volume(curandState* d_rand_state, RadianceVolume* rv, int pixel_x, int pixel_y, Scene* scene);

#endif