#ifndef IMPORTANCE_SAMPLING_PATH_TRACING_H
#define IMPORTANCE_SAMPLING_PATH_TRACING_H

#include <glm/glm.hpp>
#include <string>
#include <memory>

#include "ray.h"
#include "surface.h"
#include "area_light_plane.h"
#include "monte_carlo_settings.h"
#include "hemisphere_helpers.h"
#include "radiance_map.h"

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Traces the path of a ray using importance sampling on the bounce
    direction of the ray when it intersects with a diffuse surface. The
    higher the precomputed irradiance value for a given sector of the
    RadianceVolume, the more likely the ray is to be sampled in that
    direction, finding the light faster.
*/

vec3 path_trace_importance_sampling(RadianceMap& radiance_map, Ray ray, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces);

vec3 importance_sample_ray(const Intersection& intersection, RadianceMap& radiance_map, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces);

#endif