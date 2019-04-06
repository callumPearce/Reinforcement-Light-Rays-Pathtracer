#ifndef OBJECT_IMPORTER_H
#define OBJECT_IMPORTER_H

#include <glm/glm.hpp>
#include <vector>

#include "surface.cuh"
#include "area_light.cuh"

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

__host__
bool load_scene(const char* path, std::vector<Surface>& surfaces, std::vector<AreaLight>& area_lights, std::vector<float>& vertices);

__host__
void build_surfaces(std::vector<Surface>& surfaces, std::vector<float>& vertices, std::vector<vec3>& vertex_indices, std::vector<vec3>& temp_vertices);

__host__
void split_string(std::vector<std::string>& sub_strs, std::string search_string, std::string delimiter);

__host__
void build_area_lights(std::vector<AreaLight>& area_lights, std::vector<float>& vertices);

#endif