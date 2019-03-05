#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "shape.cuh"
using glm::dot;

/*
    Defines a Triangle shape consisting of three vertices and
    the surface normal computed based off the vertices supplied
*/
class Triangle : public Shape {

    private:

    public:
        vec4 v0;
        vec4 v1;
        vec4 v2;
        vec4 normal;

        // Constructor
        __host__
        Triangle(vec4 v0, vec4 v1, vec4 v2);

        // Gets the surface area of a triangle
        __host__ __device__
        float compute_area();

        // Samples a position on the triangle plane
        __host__
        vec4 sample_position_on_plane();

        __host__
        void compute_and_set_normal();

};
#endif
