#ifndef PRECOMPUTED_IRRADIANCE_PATH_TRACING_H
#define PRECOMPUTED_IRRADIANCE_PATH_TRACING_H

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

/*
    Traces the path of a ray to the first intersection point of that ray,
    it then finds the closest x RadianceVolumes and average their radiance
    values to find the irradiance of that point in the scene.
*/

void draw_radiance_map_path_tracing(SDLScreen screen, Camera& camera, RadianceMap& radiance_map, std::vector<AreaLightPlane *> light_planes, std::vector<Surface *> surfaces);

vec3 path_trace_radiance_map(Camera& camera, int pixel_x, int pixel_y, RadianceMap& radiance_map, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes);

#endif