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

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

void draw_voronoi_trace(vec3* host_buffer, RadianceMap* radiance_map, Camera& camera, AreaLight* light_planes, Surface* surfaces, int light_plane_count, int surfaces_count);

vec3 voronoi_trace(Camera& camera, RadianceMap* radiance_map, int pixel_x, int pixel_y, Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count);

#endif