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

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Traces the path of a ray following monte carlo path tracing algorithm:
    https://en.wikipedia.org/wiki/Path_tracing
*/

vec3 path_trace(Ray ray, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces);

vec3 indirect_radiance(const Intersection& intersection, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces);

#endif