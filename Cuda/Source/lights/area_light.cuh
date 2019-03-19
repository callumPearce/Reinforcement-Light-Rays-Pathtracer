#ifndef AREA_LIGHT_H
#define AREA_LIGHT_H

#include <glm/glm.hpp>
#include "triangle.cuh"

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Path Tracer Light
    -----------------
    Defines a triangle area light in the scene, the entire surface is treated as
    a light when intersected by the path tracer. Rays continually bounce until they
    intersect the area light, then and only then does any kind of illumination returned
    for that ray.
*/

class AreaLight : public Triangle{

    private:

    public:
        vec3 diffuse_p;
        float luminance;
        
        // Default Constructor
        __host__
        AreaLight();

        // Constructor
        __host__
        AreaLight(vec4 v0, vec4 v1, vec4 v2, vec3 diffuse_p);
};

#endif