#ifndef SURFACE_H
#define SURFACE_H

#include <glm/glm.hpp>
#include "material.cuh"
#include "triangle.cuh"

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Defines a triangle surface in the scene, this is essentially any triangle in the scene
    which is not a light, therefore it has a material associated with it
*/

class Surface : public Triangle{

    private:
        Material material = Material(vec3(0));

    public:
        // Default constructor
        Surface();

        // Constructor
        Surface(vec4 v0, vec4 v1, vec4 v2, Material material);

        // Getters
        __device__
        Material get_material();

        // Setters
        void set_material(Material material);
};

#endif