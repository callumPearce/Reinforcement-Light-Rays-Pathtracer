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

    // Sum up all samples (note we assume the surface does not emit light itself i.e. omit L_e)

    // Divide the sum by the number of samples (Monte Carlo)

    return vec3(0);
    
}

// Sample n rays within a hemisphere
vector<Ray> Light::uniform_sample_hemisphere_rays(vector<Shape *> shapes, Intersection& intersection){
    
    vec4 c = intersection.position;
    int radius = 1.f;
    for (int i = 0 ; i < SAMPLES_PER_BOUNCE ; i++) {
        bool not_found = true;
        // rejection sampling
        while (not_found) {
            float randx = ((float) rand() / (RAND_MAX)) * radius - radius / 2;
            float randy = ((float) rand() / (RAND_MAX)) * radius - radius / 2;
            float randz = ((float) rand() / (RAND_MAX)) * radius - radius / 2;
            vec4 p(c.x + randx, c.y + randy, c.z + randz, 1);
            if (contained_in_hemisphere(p, c, radius)) {
                
                vec3 position = vec3(p.x, p.y, p.z);
                vec3 centre = vec3(c.x, c.y, c.z);
                vec3 norm = vec3(
                    intersection.normal.x, 
                    intersection.normal.y, 
                    intersection.normal.z
                    );

                // Angle to rotate to reach normal
                float angle = dot(position - centre, norm);
                not_found = false;
            }
        }
    }
    return vector<Ray>();
}

// Check if a given point is within the lightsphere
bool Light::contained_in_hemisphere(vec4 point, vec4 centre, float radius){
    return distance(point, centre) <= radius && point.z  > centre.z;
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