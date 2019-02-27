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
void Ray::closest_intersection(std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, Intersection& closest_intersection) {
    closest_intersection.distance = std::numeric_limits<float>::max();
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

// Sample a ray which passes through the pixel at the specified coordinates from the camera
Ray Ray::sample_ray_through_pixel(Camera& camera, int pixel_x, int pixel_y){
    // Generate the random point within a pixel for the ray to pass through
    float x = (float)pixel_x + ((float) rand() / (RAND_MAX));
    float y = (float)pixel_y + ((float) rand() / (RAND_MAX));

    // Set direction to pass through pixel (pixel space -> Camera space)
    vec4 dir((x - (float)SCREEN_WIDTH / 2.f) , (y - (float)SCREEN_HEIGHT / 2.f) , (float)FOCAL_LENGTH , 1);
    
    // Create a ray that we will change the direction for below
    Ray ray(camera.get_position(), dir);
    ray.rotate_ray(camera.get_yaw());

    return ray;
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
