#ifndef OBJECT_IMPORTER_H
#define OBJECT_IMPORTER_H

#include <glm/glm.hpp>
#include <vector>

#include "surface.h"

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

bool load_scene(const char* path, std::vector<Surface>& surfaces);

void build_surfaces(std::vector<Surface>& surfaces, std::vector<vec3>& vertex_indices, std::vector<vec3>& temp_vertices);

void split_string(std::vector<std::string>& sub_strs, std::string search_string, std::string delimiter);

#endif