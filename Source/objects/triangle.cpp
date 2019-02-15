#include "triangle.h"
#include "image_settings.h"

#include <iostream>

Triangle::Triangle(vec4 v0, vec4 v1, vec4 v2){
    setV0(v0);
    setV1(v1);
    setV2(v2);
    compute_and_set_normal();
}

// Tests whether the triangle intersects a ray, closer to the current closest intersection
bool Triangle::intersects(Ray * ray, Intersection& intersection, int index) {
    bool returnVal = false;
    vec4 start = ray->get_start();
    vec4 dir = ray->get_direction();

    vec4 v0 = getV0();
    vec4 v1 = getV1();
    vec4 v2 = getV2();

    vec3 e1 = vec3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    vec3 e2 = vec3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    vec3 b = vec3(start.x - v0.x, start.y - v0.y, start.z - v0.z);

    dir = vec4(vec3(dir) * (float)SCREEN_HEIGHT, 1);

    mat3 A(vec3(-dir), e1, e2);

    // solution.x = t: Scalar position of intersection along raw
    // solution.y = u: Scalar position along vecotr (v1- v0)
    // solution.z = v: Scalar position along vecotr (v2- v0)
    vec3 solution;
    bool crmr = cramer(A, b, solution);

    if (crmr && solution.x >= 0.0f && solution.y >= 0.0f && solution.z >= 0.0f && solution.y + solution.z <= 1.0f) {
        if (solution.x < intersection.distance + EPS && solution.x > EPS) {
            intersection.position = start + solution.x * dir;
            intersection.position[3] = 1;
            intersection.distance = solution.x;
            intersection.normal = normal;
            intersection.index = index;
            returnVal = true;
        }
    }
    return returnVal;
}

// Cramers Rule: Solve a 3x3 linear equation system 
bool Triangle::cramer(mat3 A, vec3 b, vec3& solution) {
    bool ret = false;
    // Initialise the solution output
    solution = vec3(0,0,0);
    float detA = determinant(A);
    if (detA != 0) {
        ret = true;
        // Temp variable to hold the value of A
        mat3 temp = A;

        A[0] = b;
        solution.x = determinant(A) / detA;
        A = temp;

        A[1] = b;
        solution.y = determinant(A) / detA;
        A = temp;

        A[2] = b;
        solution.z = determinant(A) / detA;
        A = temp;
    } else {
        ret = false;
    }
    return ret;
}

// Getters
vec4 Triangle::getV0() {
    return v0;
}

vec4 Triangle::getV1() {
    return v1;
}

vec4 Triangle::getV2() {
    return v2;
}

vec4 Triangle::getNormal() {
    return normal;
}

// Setters
void Triangle::setV0(vec4 v0) {
    this->v0 = v0;
}

void Triangle::setV1(vec4 v1) {
    this->v1 = v1;
}

void Triangle::setV2(vec4 v2) {
    this->v2 = v2;
}

void Triangle::compute_and_set_normal() {
    vec3 e1 = vec3(v1.x-v0.x,v1.y-v0.y,v1.z-v0.z);
    vec3 e2 = vec3(v2.x-v0.x,v2.y-v0.y,v2.z-v0.z);
    vec3 normal3 = normalize(cross(e2, e1));
    normal.x = normal3.x;
    normal.y = normal3.y;
    normal.z = normal3.z;
    normal.w = 1.0;
    this->normal = normal;
}
