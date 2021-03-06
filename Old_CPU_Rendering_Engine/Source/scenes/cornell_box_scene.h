#ifndef CORNELL_BOX_SCENE_H
#define CORNELL_BOX_SCENE_H

#include <iostream>
#include <glm/glm.hpp>

#include "surface.h"
#include "area_light_plane.h"

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

void get_cornell_shapes(std::vector<Surface>& surfaces, std::vector<AreaLightPlane>& area_light_planes);

#endif