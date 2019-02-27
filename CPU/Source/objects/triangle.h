#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "shape.h"
using glm::dot;

/*
    Defines a Triangle shape consisting of three vertices and
    the surface normal computed based off the vertices supplied
*/
class Triangle : public Shape {

    private:
        vec4 v0;
        vec4 v1;
        vec4 v2;
        vec4 normal;

        // Cramers Rule: Solve a 3x3 linear equation system 
        bool cramer(mat3 A, vec3 b, vec3& solution);

    public:
        // Constructor
        Triangle(vec4 v0, vec4 v1, vec4 v2);

        // Tests whether a triangle intersects a ray
        bool intersects(Ray* ray, Intersection& intersection, int index);

        // Gets the surface area of a triangle
        float compute_area();

        // Samples a position on the triangle plane
        vec4 sample_position_on_plane();

        // Getters
        vec4 getV0();
        vec4 getV1();
        vec4 getV2();
        vec4 getNormal();

        // Setters
        void setV0(vec4 v0);
        void setV1(vec4 v1);
        void setV2(vec4 v2);
        void compute_and_set_normal();

};
#endif
