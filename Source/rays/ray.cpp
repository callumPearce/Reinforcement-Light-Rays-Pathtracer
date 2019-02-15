#include "ray.h"
#include "surface.h"
#include "area_light_plane.h"
#include <iostream>
#include <limits>

Ray::Ray(vec4 start, vec4 direction) {
    set_start(start);
    vec3 dir3 = normalize(vec3(direction));
    set_direction(vec4(dir3, 1));
}

// Find (if there is) the closest intersection with a given ray and a shape
void Ray::closest_intersection(vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes, Intersection& closest_intersection) {
    closest_intersection.distance = numeric_limits<float>::max();
    // Find intersection with surface
    for (int i = 0; i < surfaces.size(); i++) {
        if (surfaces[i]->intersects(this, closest_intersection, i)) {
            closest_intersection.intersection_type = SURFACE;
        }
    }
    // Find intersection with area lights
    for (int i = 0; i < light_planes.size(); i++) { //TODO: Enum on type of closest intersction
        if (light_planes[i]->light_plane_intersects(this, closest_intersection, i)) {
            closest_intersection.intersection_type = AREA_LIGHT_PLANE;
        }
    }
}

// Rotate a ray by "yaw"
void Ray::rotate_ray(float yaw) {
    mat4 R = mat4(1.0);
    R[0] = vec4(cos(yaw), 0, sin(yaw), 0);
    R[2] = vec4(-sin(yaw), 0, cos(yaw), 0);
    set_direction(R * get_direction());
}

// Getters
vec4 Ray::get_start() {
    return start;
}

vec4 Ray::get_direction() {
    return direction;
}

// Setters
void Ray::set_start(vec4 start) {
    this->start = start;
}

void Ray::set_direction(vec4 dir) {
    this->direction = dir;
}
