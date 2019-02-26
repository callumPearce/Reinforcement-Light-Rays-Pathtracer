#ifndef VORONOI_TRACE_H
#define VORONOI_TRACE_H

#include <glm/glm.hpp>
#include <string>
#include <memory>

#include "ray.h"
#include "surface.h"
#include "area_light_plane.h"
#include "monte_carlo_settings.h"
#include "hemisphere_helpers.h"
#include "radiance_map.h"
#include "image_settings.h"
#include "sdl_screen.h"

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

void draw_voronoi_trace(SDLScreen screen, RadianceMap& radiance_map, Camera& camera, std::vector<AreaLightPlane *> light_planes, std::vector<Surface *> surfaces);

vec3 voronoi_trace(Camera& camera, RadianceMap& radiance_map, int pixel_x, int pixel_y, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes);

#endif