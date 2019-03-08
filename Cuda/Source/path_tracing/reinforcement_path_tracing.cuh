#ifndef REINFORCEMENT_PATH_TRACING_H
#define REINFORCEMENT_PATH_TRACING_H

#include <glm/glm.hpp>
#include <string>
#include <memory>

#include "ray.cuh"
#include "surface.cuh"
#include "area_light.cuh"
#include "monte_carlo_settings.h"
#include "hemisphere_helpers.cuh"
#include "radiance_map.cuh"
#include "image_settings.h"
#include "sdl_screen.h"
#include "scene.cuh"

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Traces the path of a ray via importance sampling over the distribution of
    Q(x, omega), where Q is the valuation (irradiance) at point x from direction
    omega. Q is progressively learned via an adaptation of Expected Sarsa which
    is a reinforcement learning strategy. See the full paper for details:
    https://arxiv.org/abs/1701.07403
*/

__global__
void update_radiance_volume_distributions(RadianceMap* radiance_map);

__global__
void draw_reinforcement_path_tracing(vec3* device_buffer, curandState* d_rand_state, RadianceMap* radiance_map, Camera* camera, Scene* scene);

__device__
vec3 path_trace_reinforcement(curandState* d_rand_state, RadianceMap* radiance_map, Camera* camera, int pixel_x, int pixel_y, Scene* scene);

__device__
vec3 path_trace_reinforcement_iterative(int pixel_x, int pixel_y, Camera* camera, curandState* d_rand_state, RadianceMap* radiance_map, Scene* scene);

#endif
