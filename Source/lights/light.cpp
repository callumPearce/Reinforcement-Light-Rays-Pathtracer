#include "light.h"
#include "printing.h"
#include "monte_carlo_settings.h"

Light::Light(vec4 position, vec3 diffuse_p, vec3 ambient_p){
    set_position(position);
    set_diffuse_p(diffuse_p);
    set_ambient_p(ambient_p);
}

// Get the outgoing radiance (L_O(w_O)) for a given intersection point
vec3 Light::get_intersection_radiance(const Intersection& i, vector<Shape *> shapes, Ray incident_ray){
    return this->direct_light(i, shapes) + this->indirect_light(i, shapes, incident_ray, 0) + this->ambient_light(i, shapes);
}

// For a given intersection point, return the radiance of the surface directly
// resulting from this light source
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

// Get the ambient light radiance
vec3 Light::ambient_light(const Intersection& i, vector<Shape *> shapes){
    return shapes[i.index]->get_material().get_diffuse_c() * this->ambient_p;
}

// For a given intersection point, return the radiance of the surface resulting
// from indirect illumination (i.e. other shapes in the scene) via the Monte Carlo Raytracing
vec3 Light::indirect_light(const Intersection& i, vector<Shape *> shapes, Ray incident_ray, int bounces){

    // Sample SAMPLES_PER_BOUNCE angles uniformly in a hemisphere around the normal of the interesection
    vector<Ray> sampled_rays = uniform_sample_hemisphere_rays(i);

    // Sum up all samples (note we assume the surface does not emit light itself i.e. omit L_e)

    // Divide the sum by the number of samples (Monte Carlo)

    return vec3(0);
    
}

// Sample n rays within a hemisphere
vector<Ray> Light::uniform_sample_hemisphere_rays(const Intersection& intersection){
    
    vector<Ray> sampled_rays;

    for (int i = 0; i < SAMPLES_PER_BOUNCE; i++){
        vec3 dir = random_hemisphere_direction(intersection.normal);
        sampled_rays.push_back(Ray(intersection.position, vec4(dir.x, dir.y, dir.z, 1)));
        print_vec3("direction",dir);
    }
    
    return sampled_rays;
}


float Light::uniform_random(float a, float b) {
  return a + drand48() * (b - a);
}

vec3 Light::random_hemisphere_direction(vec3 normal) {

    // Make an orthogonal basis whose third vector is along `direction'
    vec3 b3 = normal;
    vec3 different = (abs(b3.x) < 0.5f) ? vec3(1.0f, 0.0f, 0.0f) : vec3(0.0f, 1.0f, 0.0f);
    vec3 b1 = normalize(cross(b3, different));
    vec3 b2 = cross(b1, b3);
    
    // Pick (x,y,z) randomly around (0,0,1)
    float z = uniform_random(cos(0.5 * M_PI), 1);
    float r = sqrt(1.0f - z * z);
    float theta = uniform_random(-M_PI, M_PI);
    float x = r * cos(theta);
    float y = r * sin(theta);
    
    // Construct the vector that has coordinates (x,y,z) in the basis formed by b1, b2, b3
    return x * b1 + y * b2 + z * b3;
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

vec3 Light::get_ambient_p(){
    return ambient_p;
}

// Setters
void Light::set_position(vec4 position){
    this->position = position;
}

void Light::set_diffuse_p(vec3 diffuse_p){
    this->diffuse_p = diffuse_p;
}

void Light::set_ambient_p(vec3 ambient_p){
    this->ambient_p = ambient_p;
}