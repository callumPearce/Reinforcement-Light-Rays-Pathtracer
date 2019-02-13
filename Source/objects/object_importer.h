#ifndef OBJECT_IMPORTER_H
#define OBJECT_IMPORTER_H

#include <glm/glm.hpp>

#include "surface.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

bool load_scene(const char* path, vector<Surface>& surfaces);

void build_surfaces(vector<Surface>& surfaces, vector<vec3>& vertex_indices, vector<vec3>& temp_vertices);

void split_string(vector<string>& sub_strs, string search_string, string delimiter);

#endif