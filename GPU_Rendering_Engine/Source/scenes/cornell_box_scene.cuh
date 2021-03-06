#ifndef CORNELL_BOX_SCENE_H
#define CORNELL_BOX_SCENE_H

#include <iostream>
#include <glm/glm.hpp>

#include "surface.cuh"
#include "area_light.cuh"

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

__host__
void get_cornell_shapes(std::vector<Surface>& surfaces, std::vector<AreaLight>& light_planes, std::vector<float>& vertices);

#endif