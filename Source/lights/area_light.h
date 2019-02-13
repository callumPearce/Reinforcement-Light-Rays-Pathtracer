#ifndef AREA_LIGHT_H
#define AREA_LIGHT_H

#include <glm/glm.hpp>
#include "triangle.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Defines a triangle area light in the scene, the entire surface is treated as
    a light when intersected by the path tracer
*/

class AreaLight : public Triangle{

    private:
        vec3 diffuse_p;

    public:
        // Constructor
        AreaLight(vec4 v0, vec4 v1, vec4 v2, vec3 diffuse_p);

        // Getters
        vec3 get_diffuse_p();

        // Setters
        void set_diffuse_p();
};

#endif