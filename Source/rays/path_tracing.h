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

using namespace std;
using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Path tracing functionality
*/

void create_normal_coordinate_system(vec3& normal, vec3& normal_T, vec3& normal_B);

vec3 uniform_hemisphere_sample(float r1, float r2);

vec3 path_trace(Ray ray, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes, int bounces);

vec3 indirect_radiance(const Intersection& intersection, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes, int bounces);

#endif