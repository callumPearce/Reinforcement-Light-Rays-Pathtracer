#ifndef SHAPE_H
#define SHAPE_H

#include <glm/glm.hpp>
#include "material.h"
#include "ray.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Defines the generic Shape class, a shape will have a material
    associated with it.
*/
class Shape {

    private:
        Material material = Material(vec3(0));

    public:
        // Constructor
        Shape(Material material);

        // Tests whether the shape intersects a ray
        virtual bool intersects(Ray * ray, Intersection & intersection, int index)=0;

        // Getters
        Material get_material();

        // Setters
        void set_material(Material material);
};

#endif