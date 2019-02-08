#include "light.h"
#include "printing.h"
#include "monte_carlo_settings.h"
#include "image_settings.h"

Light::Light(vec4 position, vec3 diffuse_p, vec3 ambient_p){
    set_position(position);
    set_diffuse_p(diffuse_p);
    set_ambient_p(ambient_p);
}

// Get the outgoing radiance (L_O(w_O)) for a given intersection point
vec3 Light::get_intersection_radiance(const Intersection& i, vector<Shape *> shapes, int bounces){
    if (bounces == MAX_RAY_BOUNCES){
        return this->direct_light(i, shapes) + this->ambient_light(i, shapes);
    } else{
        return this->direct_light(i, shapes) + this->indirect_light(i, shapes, bounces) + this->ambient_light(i, shapes);
    }
}

// Blends the colour of the two intersected indices
vec3 Light::blend_intersection_diffuse_c(const Intersection& i, vector<Shape *> shapes){
    vec3 diff_1 = shapes[i.indices[0]]->get_material().get_diffuse_c();
    // vec3 diff_2 = shapes[i.indices[1]]->get_material().get_diffuse_c();
    // vec3 mixed = (diff_1 + diff_2)/2.f;
    // mixed.x = glm::mix(diff_1.x, diff_2.x, 0.5f * fabs(i.distances[0] - i.distances[1]) / EPS);
    // mixed.y = glm::mix(diff_1.y, diff_2.y, 0.5f * fabs(i.distances[0] - i.distances[1]) / EPS);
    // mixed.z = glm::mix(diff_1.z, diff_2.z, 0.5f * fabs(i.distances[0] - i.distances[1]) / EPS);
    return diff_1;
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

    vec4 start = i.position + 0.001f * vec4(direction_point_to_light, 1);
    start[3] = 1.f;
    Ray point_to_light(
        start,
        vec4(direction_point_to_light, 1)
        );

    Intersection closest_intersection;

    if (point_to_light.closest_intersection(shapes, closest_intersection)){
        float distance_to_shape = distance(i.position, closest_intersection.position);
        if (distance_to_shape < distance_to_light && abs(distance_to_light - distance_to_shape) > EPS){
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

    return diffuse_illumination * blend_intersection_diffuse_c(i, shapes);
}

// Get the ambient light radiance
vec3 Light::ambient_light(const Intersection& i, vector<Shape *> shapes){
    return blend_intersection_diffuse_c(i, shapes) * this->ambient_p;
}


// For a given intersection point, return the radiance of the surface resulting
// from indirect illumination (i.e. other shapes in the scene) via the Monte Carlo Raytracing
vec3 Light::indirect_light(const Intersection& intersection, vector<Shape *> shapes, int bounces){

    // 1) Create new coordinate system (tranformation matrix)
    vec3 normal = vec3(intersection.normal.x, intersection.normal.y, intersection.normal.z);
    vec3 normal_T = vec3(0);
    vec3 normal_B = vec3(0);
    create_normal_coordinate_system(normal, normal_T, normal_B);

    // Calculate the estiamted total radiance estimate
    // (\int L(w_i) * roh / pi * cos(w_i, N) dw)
    vec3 total_radiance = vec3(0);
    for (int i = 0; i < SAMPLES_PER_BOUNCE; i++){
        
        // Generate random number for monte carlo sampling of theta and phi
        float cos_theta = ((float) rand() / (RAND_MAX)); //r1
        float r2 = ((float) rand() / (RAND_MAX));

        // 2) Sample uniformly coordinates on unit hemisphere
        vec3 sample = uniform_hemisphere_sample(cos_theta, r2);

        // 3) Transform random sampled position into the world coordinate system
        vec3 sampled_direction = vec3(
            sample.x * normal_B.x + sample.y * normal.x + sample.z * normal_T.x, 
            sample.x * normal_B.y + sample.y * normal.y + sample.z * normal_T.y, 
            sample.x * normal_B.z + sample.y * normal.z + sample.z * normal_T.z
        );

        // Create the new bounced ray
        vec4 start = intersection.position + (0.001f * vec4(sampled_direction, 1));
        start[3] = 1.f;
        Ray ray = Ray(start, vec4(sampled_direction, 1));

        // 4) Get the radiance contribution for this ray and add to the sum
        vec3 radiance = vec3(0);
        Intersection intersection;
        if (ray.closest_intersection(shapes, intersection)) {
            radiance = this->get_intersection_radiance(intersection, shapes, bounces+1);
        } 
        // Note: we can multiply the BRDF to the final sum because it is 
        // constant for diffuse surfaces, so we omit it here
        total_radiance += cos_theta * radiance;
    }

    // Divide the sum by the number of samples (Monte Carlo) and apply BRDF
    // Note: 1/2pi comes from PDF being constant for sampled ray directions (Monte Carlo) 
    total_radiance /= ((float)SAMPLES_PER_BOUNCE * (1 / (2 * M_PI))); 
    total_radiance *= blend_intersection_diffuse_c(intersection, shapes);

    return total_radiance;
}

// Generate a random point on a unit hemisphere
vec3 Light::uniform_hemisphere_sample(float r1, float r2){

    // cos(theta) = 1 - r1 same as just doing r1
    float y = r1; 
    
    // theta = cos^-1 (1 - r1 - r1)
    float sin_theta = sqrt(1 - r1 * r1);

    // phi = 2*pi * r2
    float phi = 2 * M_PI * r2;

    // x = sin(theta) * cos(phi)
    float x = sin_theta * cosf(phi);
    // z = sin(theta) * sin(phi)
    float z = sin_theta * sinf(phi);

    return vec3(x,y,z);
}

// Create the new coordinate system based on the normal being the y-axis unit vector.
// In other words, create a **basis** set of vectors which any vector in the 3D space
// can be created with by taking a linear combination of these 3 vectors
void Light::create_normal_coordinate_system(vec3& normal, vec3& normal_T, vec3& normal_B){
    // normal_T is found by setting either x or y to 0
    // i.e. the two define a plane
    if (fabs(normal.x) > fabs(normal.y)){
        // N_x * x = -N_z * z
        normal_T = normalize(vec3(normal.z, 0, -normal.x));
    } else{
        //N_y * y = -N_z * z
        normal_T = normalize(vec3(0, -normal.z, normal.y));
    }
    // The cross product between the two vectors creates another  
    // perpendicular to the plane formed by normal, normal_T
    normal_B = cross(normal, normal_T);
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