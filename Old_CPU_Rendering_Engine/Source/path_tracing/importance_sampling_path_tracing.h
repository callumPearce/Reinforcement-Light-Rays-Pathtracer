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
#include "camera.h"
#include "image_settings.h"
#include "sdl_screen.h"

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
void draw_importance_sampling_path_tracing(SDLScreen screen, Camera& camera, RadianceMap& radiance_map, std::vector<AreaLightPlane *> light_planes, std::vector<Surface *> surfaces);

vec3 path_trace_importance_sampling(RadianceMap& radiance_map, Camera& camera, int pixel_x, int pixel_y, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes);

vec3 path_trace_importance_sampling_recursive(RadianceMap& radiance_map, Ray ray, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces);

vec3 importance_sample_ray(const Intersection& intersection, RadianceMap& radiance_map, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces);

#endif