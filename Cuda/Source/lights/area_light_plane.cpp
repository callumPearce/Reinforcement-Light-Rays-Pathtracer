#include <stdexcept>
#include "area_light_plane.h"

AreaLightPlane::AreaLightPlane(std::vector<vec4> vertices, vec3 diffuse_p){
    if (vertices.size() < 3){
        printf("At 3 vertices must be provided to define a plane in 3D!\n");
    } else{
        set_diffuse_p(diffuse_p);
        generate_area_lights(vertices);
    }
}

// Generate the area lights from the vertices (fan triangulation)
void AreaLightPlane::generate_area_lights(std::vector<vec4> vertices){
    vec4 initial_v = vertices[0];
    // Fan triangulation to create area lights
    for (int i = 1; i < vertices.size()-1; i++){
        this->area_lights.push_back(
            AreaLight(initial_v, vertices[i], vertices[i+1], this->diffuse_p)
            );
    }
}

// Check if a given ray intersects with the light plane closer then the current intersction
bool AreaLightPlane::light_plane_intersects(Ray * ray, Intersection& intersction, int index){
    bool returnVal = false;
    for (int i = 0; i < this->area_lights.size(); i++) { 
        if (this->area_lights[i].intersects(ray, intersction, index)) {
            returnVal = true;
        }
    }
    return returnVal;
}

// Getters
std::vector<AreaLight> AreaLightPlane::get_area_lights(){
    return this->area_lights;
}

vec3 AreaLightPlane::get_diffuse_p(){
    return this->diffuse_p;
}

// Setters
void AreaLightPlane::set_diffuse_p(vec3 diffuse_p){
    this->diffuse_p = diffuse_p;
}