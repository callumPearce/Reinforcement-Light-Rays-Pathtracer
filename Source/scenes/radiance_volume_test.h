#ifndef RADIANCE_VOLUME_TEST_H
#define RADIANCE_VOLUME_TEST_H

#include <glm/glm.hpp>
#include <vector>
#include <surface.h>

using namespace std;
using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

void get_radiance_volume_shapes(vector<Surface>& surfaces);

#endif