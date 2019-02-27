#ifndef DEFAULT_PATH_TRACING_H
#define DEFAULT_PATH_TRACING_H

#include <glm/glm.hpp>
#include <string>
#include <memory>

#include "ray.h"
#include "surface.h"
#include "area_light_plane.h"
#include "monte_carlo_settings.h"
#include "hemisphere_helpers.h"
#include "camera.h"
#include "image_settings.h"
#include "printing.h"
#include "sdl_screen.h"

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Traces the path of a ray following monte carlo path tracing algorithm:
    https://en.wikipedia.org/wiki/Path_tracing
*/
void draw_default_path_tracing(SDLScreen screen, Camera& camera, std::vector<AreaLightPlane *> light_planes, std::vector<Surface *> surfaces);

vec3 path_trace(Camera& camera, int pixel_x, int pixel_y, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes);

vec3 path_trace_recursive(Ray ray, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces);

vec3 indirect_irradiance(const Intersection& intersection, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces);

#endif