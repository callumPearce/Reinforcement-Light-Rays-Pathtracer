#ifndef PATH_TRACING_H
#define PATH_TRACING_H

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>

#include "ray.h"
#include "surface.h"
#include "area_light_plane.h"
#include "monte_carlo_settings.h"
#include "hemisphere_helpers.h"
#include "radiance_map.h"

using namespace std;
using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Path tracing functionality
*/

// Traces the path of a ray following monte carlo path tracer in order to estimate the radiance for a ray shot
// from its angle and starting position
vec3 path_trace(Ray ray, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes, int bounces);

vec3 path_trace_radiance_map(RadianceMap& radiance_map, Ray ray, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes);

vec3 indirect_radiance(const Intersection& intersection, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes, int bounces);

#endif