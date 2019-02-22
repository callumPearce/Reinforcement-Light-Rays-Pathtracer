#include "interpolation.h"

// Compute the barycentric coordinates for point P in the triangle defined by
// v0, v1, v2 vertices and fill in the two barycentric coordinates to u, v
// return true if the point lies within the triangle
bool compute_barycentric(vec4 v0, vec4 v1, vec4 v2, vec4 P, float& u, float& v){
    float area = length(cross(vec3(v1 - v0), vec3(v2 - v0)));
    float u_area = length(cross(vec3(v0 - v2), vec3(P - v2))); //(CA, CP)
    float v_area = length(cross(vec3(v1 - v0), vec3(P - v0))); //(AB, AP)
    float w_area = length(cross(vec3(v2 - v1), vec3(P - v1))); //(BC, BP)
    u = u_area/area;
    v = v_area/area;
    float w = w_area/area;
    if (u + v + w == 1.f){
        return true;
    } else{
        return false;
    }
}