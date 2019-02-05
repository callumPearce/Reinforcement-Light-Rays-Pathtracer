#include "light.h"
#include "printing.h"

Light::Light(vec4 position, vec3 diffuse_p){
    set_position(position);
    set_diffuse_p(diffuse_p);
}

// For a given intersection point, return the diffuse illumination
// for the surface intersected with
vec3 Light::direct_light(const Intersection& i, vector<Shape *> shapes){

    float distance_to_light = distance(i.position, this->position);

    // We must first check if the surface receives illumination from the light
    // source by checking if there is an opaque light object blocking the light
    // from reached the intersection point, in which case, no contribution of light
    // is given to the surface
    vec3 direction_point_to_light(
        this->position.x - i.position.x,
        this->position.y - i.position.y,
        this->position.z - i.position.z
    );

    direction_point_to_light = normalize(direction_point_to_light);

    Ray point_to_light(
        i.position + 0.001f * vec4(direction_point_to_light, 1),
        vec4(direction_point_to_light, 1)
        );

    Intersection closest_intersection;

    if (point_to_light.closest_intersection(shapes, closest_intersection)){
        float distance_to_shape = distance(i.position, closest_intersection.position);
        if (distance_to_shape < distance_to_light){
           return vec3(0,0,0);
        }
    }

    // Compute the diffuse illumination
    vec3 point_normal(i.normal.x, i.normal.y, i.normal.z);
    float power_per_surface_area = (max(dot(direction_point_to_light, point_normal), 0.f)) / (4.f * M_PI * pow(distance_to_light,2));

    vec3 diffuse_c = this->get_diffuse_p();
    vec3 diffuse_illumination = vec3(
        diffuse_c.x * power_per_surface_area, 
        diffuse_c.y * power_per_surface_area, 
        diffuse_c.z * power_per_surface_area
        );

    diffuse_illumination = diffuse_illumination;

    return diffuse_illumination * shapes[i.index]->get_material().get_diffuse_c();
}

vec3 Light::ambient_light(const Intersection& i, vector<Shape *> shapes, vec3 l_ambient){
    return shapes[i.index]->get_material().get_diffuse_c() * l_ambient;
}

// Movement functions
void Light::translate_left(float distance) {
    set_position(get_position() - vec4(distance, 0, 0, 0));
}

void Light::translate_right(float distance) {
    set_position(get_position() + vec4(distance, 0, 0, 0));
}

void Light::translate_forwards(float distance) {
    set_position(get_position() + vec4(0, 0, distance, 0));
}

void Light::translate_backwards(float distance) {
    set_position(get_position() - vec4(0, 0, distance, 0));
}

void Light::translate_up(float distance) {
    set_position(get_position() + vec4(0, distance, 0, 0));
}

void Light::translate_down(float distance) {
   set_position(get_position() - vec4(0, distance, 0, 0));
}

// Getters
vec4 Light::get_position(){
    return position;
}

vec3 Light::get_diffuse_p(){
    return diffuse_p;
}

// Setters
void Light::set_position(vec4 position){
    this->position = position;
}

void Light::set_diffuse_p(vec3 diffuse_p){
    this->diffuse_p = diffuse_p;
}