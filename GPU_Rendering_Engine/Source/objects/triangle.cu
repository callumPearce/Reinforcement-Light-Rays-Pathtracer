#include "triangle.cuh"
#include "image_settings.h"
#include "printing.h"
#include <iostream>

__host__
Triangle::Triangle(vec4 v0, vec4 v1, vec4 v2){
    this->v0 = v0;
    this->v1 = v1;
    this->v2 = v2;
    compute_and_set_normal();
}


// Computes and returns the surface area of this triangle
__host__ __device__
float Triangle::compute_area(){
    // A = 1/2 * |AB||AC|sin(theta)
    vec3 v0_3 = vec3(v0.x, v0.y, v0.z);
    vec3 v1_3 = vec3(v1.x, v1.y, v1.z);
    vec3 v2_3 = vec3(v2.x, v2.y, v2.z);
    float e01_e02 = length(v1_3 - v0_3) * length(v2_3 - v0_3);
    float cos_theta = dot(v1_3 - v0_3, v2_3 - v0_3)/e01_e02;
    float sin_theta = sqrt(1 - pow(cos_theta,2));
    return 0.5f * e01_e02 * sin_theta;
}

// Sample a position on the triangles plane
__host__ 
vec4 Triangle::sample_position_on_plane(){
    // http://mathworld.wolfram.com/TrianglePointPicking.html
    // https://math.stackexchange.com/questions/538458/triangle-point-picking-in-3d
    // x =  v0 + a_1*(v1-v0) + a_2*(v2-v0) 
    float a1 = 1.f;
    float a2 = 1.f;
    vec4 pos = vec4(0);
    do{
        a1 = ((float) rand() / (RAND_MAX));
        a2 = ((float) rand() / (RAND_MAX));
        pos = this->v0 + a1*(this->v1 - this->v0)  + a2*(this->v2 - this->v0);
    }
    while(a1 + a2 > 1.f);
    pos.w = 1.f;
    return pos;
}

// Sample a position on the triangles plane
__device__ 
vec4 Triangle::sample_position_on_plane(curandState* d_rand_state, int i){
    // http://mathworld.wolfram.com/TrianglePointPicking.html
    // https://math.stackexchange.com/questions/538458/triangle-point-picking-in-3d
    // x =  v0 + a_1*(v1-v0) + a_2*(v2-v0) 
    float a1 = 1.f;
    float a2 = 1.f;
    vec4 pos = vec4(0);
    do{
        a1 = (curand_uniform(&d_rand_state[ i ]));
        a2 = (curand_uniform(&d_rand_state[ i ]));
        pos = this->v0 + a1*(this->v1 - this->v0)  + a2*(this->v2 - this->v0);
    }
    while(a1 + a2 > 1.f);
    pos.w = 1.f;
    return pos;
}

__host__
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
