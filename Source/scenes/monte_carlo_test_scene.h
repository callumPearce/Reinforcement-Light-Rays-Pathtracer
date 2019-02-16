#ifndef MONTE_CARLO_TEST_SCENE_h
#define MONTE_CARLO_TEST_SCENE_h

#include <iostream>
#include <glm/glm.hpp>
#include <vector>

#include "area_light_plane.h"
#include "surface.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

void get_monte_carlo_shapes(vector<Surface>& surfaces, std::vector<AreaLightPlane>& area_light_planes);

#endif