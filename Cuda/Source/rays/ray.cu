#include "ray.cuh"
#include "surface.cuh"
#include "area_light.cuh"
#include <limits>

__device__
Ray::Ray(vec4 start, vec4 direction) {
    set_start(start);
    vec3 dir3 = normalize(vec3(direction));
    set_direction(vec4(dir3, 1));
}

// Find (if there is) the closest intersection with a given ray and a shape
__device__
void Ray::closest_intersection(Surface* surfaces, AreaLight* light_planes, Intersection& closest_intersection, int light_plane_count, int surfaces_count) {
    closest_intersection.distance = 9999999999.f;
    // Find intersection with surface
    for (int i = 0; i < surfaces_count; i++) {
        if (surfaces[i].intersects(this, closest_intersection, i)) {
            closest_intersection.intersection_type = SURFACE;
        }
    }
    // Find intersection with area lights
    for (int i = 0; i < light_plane_count; i++) { //TODO: Enum on type of closest intersction
        if (light_planes[i].intersects(this, closest_intersection, i)) {
            closest_intersection.intersection_type = AREA_LIGHT;
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
