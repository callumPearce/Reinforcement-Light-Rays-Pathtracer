#ifndef VORONOI_TRACE_H
#define VORONOI_TRACE_H

#include <glm/glm.hpp>
#include <string>
#include <memory>

#include "ray.cuh"
#include "surface.cuh"
#include "area_light.cuh"
#include "monte_carlo_settings.h"
#include "radiance_map.cuh"
#include "image_settings.h"
#include "sdl_screen.h"
#include "scene.cuh"

//cuRand
#include <curand_kernel.h>
#include <curand.h>

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

__global__
void draw_voronoi_trace(vec3* device_buffer, curandState* d_rand_state, RadianceMap* radiance_map, Camera* camera, Scene* scene);

__device__
vec3 voronoi_trace(curandState* d_rand_state, Camera* camera, RadianceMap* radiance_map, int pixel_x, int pixel_y, Scene* scene);

#endif