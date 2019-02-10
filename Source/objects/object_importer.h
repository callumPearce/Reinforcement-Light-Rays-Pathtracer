#ifndef OBJECT_IMPORTER_H
#define OBJECT_IMPORTER_H

#include <glm/glm.hpp>

#include "triangle.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

bool load_scene(const char* path, vector<Triangle>& triangles);

void build_triangles(vector<Triangle>& triangles, vector<vec3>& vertex_indices, vector<vec3>& temp_vertices);

void split_string(vector<string>& sub_strs, string search_string, string delimiter);

#endif