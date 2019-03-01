#include "ray.cuh"
#include "surface.cuh"
#include "area_light.cuh"
#include <limits>

__device__
Ray::Ray(vec4 start, vec4 direction) {
    set_start(start);
    vec3 dir3 = normalize(vec3(direction));
    set_direction(vec4(dir3, 1));
    Intersection i;
    i.intersection_type = NOTHING;
    this->intersection = i;
}

// Find (if there is) the closest intersection with a given ray and a shape
// __device__
// void Ray::closest_intersection(Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count) {
//     this->intersection.distance = 999999.f;

//     // vec3 colour = surfaces[i].get_material().get_diffuse_c();
//     // printf("(%.3f, %.3f, %.3f)\n", colour.x, colour.y, colour.z);

//     // Find intersection with surface
//     for (int i = 0; i < surfaces_count; i++) {
//         if (surfaces[i].intersects(this, i)) {
//             this->intersection.intersection_type = SURFACE;
//         }
//     }
//     // Find intersection with area lights
//     for (int i = 0; i < light_plane_count; i++) { //TODO: Enum on type of closest intersection
//         if (light_planes[i].intersects(this, i)) {
//             this->intersection.intersection_type = AREA_LIGHT;
//         }
//     }
// }

__device__
void Ray::closest_intersection(Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count) {
    this->intersection.distance = 999999.f;

    // vec3 colour = surfaces[i].get_material().get_diffuse_c();
    // printf("(%.3f, %.3f, %.3f)\n", colour.x, colour.y, colour.z);

    // Find intersection with surface
    for (int i = 0; i < surfaces_count; i++) {

        bool returnVal = false;
        vec4 start = this->get_start();
        vec4 dir = this->get_direction();
    
        vec4 v0 = surfaces[i].getV0();
        vec4 v1 = surfaces[i].getV1();
        vec4 v2 = surfaces[i].getV2();
    
        vec3 e1 = vec3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        vec3 e2 = vec3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        vec3 b = vec3(start.x - v0.x, start.y - v0.y, start.z - v0.z);
    
        dir = vec4(vec3(dir) * (float)SCREEN_HEIGHT, 1);
    
        mat3 A(vec3(-dir), e1, e2);
    
        // solution.x = t: Scalar position of intersection along raw
        // solution.y = u: Scalar position along vecotr (v1- v0)
        // solution.z = v: Scalar position along vecotr (v2- v0)
        vec3 solution;
        bool crmr = surfaces[i].cramer(A, b, solution);
    
        if (crmr && solution.x >= 0.0f && solution.y >= 0.0f && solution.z >= 0.0f && solution.y + solution.z <= 1.0f) {
            if (solution.x < this->intersection.distance + EPS && solution.x > EPS) {
                this->intersection.position = start + solution.x * dir;
                this->intersection.position[3] = 1;
                this->intersection.distance = solution.x;
                this->intersection.normal = surfaces[i].getNormal();
                this->intersection.index = i;
                returnVal = true;
            }
        }
        if(returnVal){
            this->intersection.intersection_type = SURFACE;
        }
    }
    
    // Find intersection with area lights
    for (int i = 0; i < light_plane_count; i++) { //TODO: Enum on type of closest intersection
        bool returnVal = false;
        vec4 start = this->get_start();
        vec4 dir = this->get_direction();
    
        vec4 v0 = light_planes[i].getV0();
        vec4 v1 = light_planes[i].getV1();
        vec4 v2 = light_planes[i].getV2();
    
        vec3 e1 = vec3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        vec3 e2 = vec3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        vec3 b = vec3(start.x - v0.x, start.y - v0.y, start.z - v0.z);
    
        dir = vec4(vec3(dir) * (float)SCREEN_HEIGHT, 1);
    
        mat3 A(vec3(-dir), e1, e2);
    
        // solution.x = t: Scalar position of intersection along raw
        // solution.y = u: Scalar position along vecotr (v1- v0)
        // solution.z = v: Scalar position along vecotr (v2- v0)
        vec3 solution;
        bool crmr = light_planes[i].cramer(A, b, solution);
    
        if (crmr && solution.x >= 0.0f && solution.y >= 0.0f && solution.z >= 0.0f && solution.y + solution.z <= 1.0f) {
            if (solution.x < this->intersection.distance + EPS && solution.x > EPS) {
                this->intersection.position = start + solution.x * dir;
                this->intersection.position[3] = 1;
                this->intersection.distance = solution.x;
                this->intersection.normal = light_planes[i].getNormal();
                this->intersection.index = i;
                returnVal = true;
            }
        }
        if(returnVal){
            this->intersection.intersection_type = AREA_LIGHT;
        }
    }
}


// Rotate a ray by "yaw"
__device__
void Ray::rotate_ray(float yaw) {
    mat4 R = mat4(1.0);
    R[0] = vec4(cos(yaw), 0, sin(yaw), 0);
    R[2] = vec4(-sin(yaw), 0, cos(yaw), 0);
    set_direction(R * get_direction());
}

// Getters
__device__
vec4 Ray::get_start() {
    return start;
}

__device__
vec4 Ray::get_direction() {
    return direction;
}

// Setters
__device__
void Ray::set_start(vec4 start) {
    this->start = start;
}

__device__
void Ray::set_direction(vec4 dir) {
    this->direction = dir;
}
