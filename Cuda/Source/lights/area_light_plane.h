#ifndef AREA_LIGHT_PLANE_H
#define AREA_LIGHT_PLANE_H

#include <glm/glm.hpp>
#include "area_light.h"
#include "ray.h"
#include <vector>

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;


/*
    Defines a plane of area lights i.e. multiple triangular area lights
    combined together to form a plane
*/

class AreaLightPlane{

    private:
        std::vector<AreaLight> area_lights;
        vec3 diffuse_p;

        // Generate the area lights from the vertices (fan triangulation)
        void generate_area_lights(std::vector<vec4> vertices);

    public:
        // Constructor
        AreaLightPlane(std::vector<vec4> vertices, vec3 diffuse_p);

        // Check if a given ray intersects with the light plane closer then the current intersction
        bool light_plane_intersects(Ray * ray, Intersection& intersction, int index);

        // Getters
        std::vector<AreaLight> get_area_lights();
        vec3 get_diffuse_p();

        // Setters
        void set_diffuse_p(vec3 diffuse_p);
};

#endif