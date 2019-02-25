#ifndef REINFORCEMENT_PATH_TRACING_H
#define REINFORCEMENT_PATH_TRACING_H

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

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Traces the path of a ray via importance sampling over the distribution of
    Q(x, omega), where Q is the valuation (irradiance) at point x from direction
    omega. Q is progressively learned via an adaptation of Expected Sarsa which
    is a reinforcement learning strategy. See the full paper for details:
    https://arxiv.org/abs/1701.07403
*/

vec3 path_trace_reinforcement(Camera& camera, int pixel_x, int pixel_y, RadianceMap& radiance_map, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes);

#endif