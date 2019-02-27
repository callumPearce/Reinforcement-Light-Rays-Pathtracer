#include "area_light.h"
#include "printing.h"
#include "monte_carlo_settings.h"
#include "image_settings.h"

AreaLight::AreaLight() : Triangle(vec4(0), vec4(0), vec4(0)){
    set_diffuse_p(vec3(0));
}

AreaLight::AreaLight(vec4 v0, vec4 v1, vec4 v2, vec3 diffuse_p) : Triangle(v0, v1, v2){
    set_diffuse_p(diffuse_p);
}

// Getters
vec3 AreaLight::get_diffuse_p(){
    return this->diffuse_p;
}

// Setters
void AreaLight::set_diffuse_p(vec3 diffuse_p){
    this->diffuse_p = diffuse_p;
}