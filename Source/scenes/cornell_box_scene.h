#ifndef CORNELL_BOX_SCENE_H
#define CORNELL_BOX_SCENE_H

#include <iostream>
#include <glm/glm.hpp>
#include <vector>

#include "triangle.h"
#include "shape.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

void get_cornell_shapes(vector<Triangle>& triangles);

#endif