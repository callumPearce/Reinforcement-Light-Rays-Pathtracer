#ifndef SHAPE_H
#define SHAPE_H

#include <glm/glm.hpp>
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

    public:
        // Constructor
        Shape();

        // Tests whether the shape intersects a ray
        virtual bool intersects(Ray * ray, Intersection & intersection, int index)=0;
};

#endif