#include "ray.h"
#include "shape.h"
#include <iostream>
#include <limits>

Ray::Ray(vec4 start, vec4 direction) {
    set_start(start);
    vec3 dir3 = normalize(vec3(direction));
    set_direction(vec4(dir3, 1));
}

bool Ray::closest_intersection(vector<Shape *> shapes, Intersection& closest_intersection) {
    closest_intersection.distance = numeric_limits<float>::max();
    bool returnVal = false;
    for (int i = 0; i < shapes.size(); i++) {
        if (shapes[i]->intersects(this, closest_intersection, i)) {
            returnVal = true;
        }
    }
    return returnVal;
}

// Getters
vec4 Ray::get_start() {
    return start;
}

vec4 Ray::get_direction() {
    float lngth = length(vec3(direction));
    return direction;
}

// Setters
void Ray::set_start(vec4 start) {
    this->start = start;
}

void Ray::set_direction(vec4 dir) {
    this->direction = dir;
}
