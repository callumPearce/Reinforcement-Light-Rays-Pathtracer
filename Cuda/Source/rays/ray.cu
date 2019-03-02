#include "ray.cuh"
#include "surface.cuh"
#include "area_light.cuh"
#include <limits>

__device__
Ray::Ray(vec4 start, vec4 direction) {
    this->start = start;
    vec3 dir3 = normalize(vec3(direction));
    this->direction = vec4(dir3, 1);
    Intersection i;
    i.intersection_type = NOTHING;
    this->intersection = i;
}

__device__
void Ray::closest_intersection(Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count) {
    
    this->intersection.distance = 999999.f;

    // Find intersection with surface
    for (int i = 0; i < surfaces_count; i++) {
        bool return_val = this->intersects(i, surfaces);
        if(return_val){
            this->intersection.intersection_type = SURFACE;
        }
    }
    
    // Find intersection with area lights
    for (int i = 0; i < light_plane_count; i++) { //TODO: Enum on type of closest intersection
        bool return_val = this->intersects(i, light_planes);
        if(return_val){
            this->intersection.intersection_type = AREA_LIGHT;
        }
    }
}

// Tests whether the triangle intersects a ray, closer to the current closest intersection
__device__
bool Ray::intersects(int index, Surface* surfaces) {
    bool returnVal = false;
    vec4 start = this->start;
    vec4 dir = this->direction;

    vec4 v0 = surfaces[index].v0;
    vec4 v1 = surfaces[index].v1;
    vec4 v2 = surfaces[index].v2;

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
        if (solution.x < this->intersection.distance + EPS && solution.x > EPS) {
            this->intersection.position = start + solution.x * dir;
            this->intersection.position[3] = 1;
            this->intersection.distance = solution.x;
            this->intersection.normal = surfaces[index].normal;
            this->intersection.index = index;
            returnVal = true;
        }
    }
    return returnVal;
}

// Tests whether the triangle intersects a ray, closer to the current closest intersection
__device__
bool Ray::intersects(int index, AreaLight* area_lights) {
    bool returnVal = false;
    vec4 start = this->start;
    vec4 dir = this->direction;

    vec4 v0 = area_lights[index].v0;
    vec4 v1 = area_lights[index].v1;
    vec4 v2 = area_lights[index].v2;

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
        if (solution.x < this->intersection.distance + EPS && solution.x > EPS) {
            this->intersection.position = start + solution.x * dir;
            this->intersection.position[3] = 1;
            this->intersection.distance = solution.x;
            this->intersection.normal = area_lights[index].normal;
            this->intersection.index = index;
            returnVal = true;
        }
    }
    return returnVal;
}

// Cramers Rule: Solve a 3x3 linear equation system
__device__ 
bool Ray::cramer(mat3 A, vec3 b, vec3& solution) {
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


// Rotate a ray by "yaw"
__device__
void Ray::rotate_ray(float yaw) {
    mat4 R = mat4(1.0);
    R[0] = vec4(cos(yaw), 0, sin(yaw), 0);
    R[2] = vec4(-sin(yaw), 0, cos(yaw), 0);
    this->direction = (R * this->direction);
}

