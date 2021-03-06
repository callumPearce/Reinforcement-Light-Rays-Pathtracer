#ifndef PRINTING_H
#define PRINTING_H

#include <iostream>
#include <glm/glm.hpp>

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

void print_vec3(std::string name, vec3 v);

void print_vec4(std::string name, vec4 v);

#endif