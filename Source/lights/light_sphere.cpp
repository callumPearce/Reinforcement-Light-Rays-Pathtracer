#include "light_sphere.h"
#include <iostream>

LightSphere::LightSphere(vec4 centre, float radius, int num_lights, vec3 diffuse_p, vec3 ambient_p){
    set_centre(centre);
    set_radius(radius);
    set_diffuse_p(diffuse_p);
    set_ambient_p(ambient_p);
    create_lights(num_lights);
}

// Populate the list of lights by sampling light locations within the speher
void LightSphere::create_lights(int sample_count){
    vec4 c = this->centre;
    for (int i = 0 ; i < sample_count ; i++) {
        bool not_found = true;
        // rejection sampling
        while (not_found) {
            float randx = ((float) rand() / (RAND_MAX)) * radius - radius / 2;
            float randy = ((float) rand() / (RAND_MAX)) * radius - radius / 2;
            float randz = ((float) rand() / (RAND_MAX)) * radius - radius / 2;
            vec4 p(c.x + randx, c.y + randy, c.z + randz, 1);
            if (contained_in_sphere(p)) {
                PointLight light(p, this->diffuse_p / (float) sample_count, this->ambient_p/ (float) sample_count);
                point_lights.push_back(light);
                not_found = false;
            }
        }
    }
}

// Check if a given point is within the lightsphere
bool LightSphere::contained_in_sphere(vec4 point){
    return distance(point, centre) <= radius;
}

// Return the light for a given interesection point contributed to by the lightsphere
vec3 LightSphere::get_intersection_radiance(Intersection& intersection, vector<Surface *> surfaces){
    vec3 colour(0,0,0);
    float size = (float)point_lights.size();
    for (int i = 0 ; i < point_lights.size() ; i++) {
        PointLight l = point_lights[i];
        vec3 l_light = l.get_intersection_radiance(intersection, surfaces, 0);
        colour = vec3(colour.x + l_light.x, colour.y + l_light.y, colour.z + l_light.z);
    }
    return colour;
}

// Movement 
void LightSphere::translate_left(float distance) {
    set_centre(get_centre() - vec4(distance, 0, 0, 0));
    for (int i = 0 ; i < point_lights.size() ; i++) {
        point_lights[i].translate_left(distance);
    }
}

void LightSphere::translate_right(float distance) {
    set_centre(get_centre() - vec4(distance, 0, 0, 0));
    for (int i = 0 ; i < point_lights.size() ; i++) {
        point_lights[i].translate_right(distance);
    }
}

void LightSphere::translate_forwards(float distance) {
    set_centre(get_centre() - vec4(distance, 0, 0, 0));
    for (int i = 0 ; i < point_lights.size() ; i++) {
        point_lights[i].translate_forwards(distance);
    }
}

void LightSphere::translate_backwards(float distance) {
    set_centre(get_centre() - vec4(distance, 0, 0, 0));
    for (int i = 0 ; i < point_lights.size() ; i++) {
        point_lights[i].translate_backwards(distance);
    }
}

void LightSphere::translate_up(float distance) {
    set_centre(get_centre() - vec4(distance, 0, 0, 0));
    for (int i = 0 ; i < point_lights.size() ; i++) {
        point_lights[i].translate_up(distance);
    }
}

void LightSphere::translate_down(float distance) {
    set_centre(get_centre() - vec4(distance, 0, 0, 0));
    vector<PointLight> newLights;
    for (int i = 0 ; i < point_lights.size() ; i++) {
        point_lights[i].translate_down(distance);
    }
}

// Getters
vec4 LightSphere::get_centre(){
    return centre;
}

float LightSphere::get_radius(){
    return radius;
}

vec3 LightSphere::get_diffuse_p(){
    return diffuse_p;
}

vector<PointLight> LightSphere::get_point_lights(){
    return point_lights;
}

vec3 LightSphere::get_ambient_p(){
    return ambient_p;
}

// Setters
void LightSphere::set_centre(vec4 centre){
    this->centre = centre;
}

void LightSphere::set_radius(float r){
    this->radius = r;
}

void LightSphere::set_diffuse_p(vec3 diffuse_p){
    this->diffuse_p = diffuse_p; 
}

void LightSphere::set_point_lights(vector<PointLight> point_lights){
    this->point_lights = point_lights;
}

void LightSphere::set_ambient_p(vec3 l_amb){
    this->ambient_p = l_amb;
}