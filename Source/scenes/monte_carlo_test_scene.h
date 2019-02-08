#ifndef MONTE_CARLO_TEST_SCENE_h
#define MONTE_CARLO_TEST_SCENE_h

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

void get_monte_carlo_shapes(vector<Triangle>& triangles);

#endif