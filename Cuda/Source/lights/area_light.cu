#include "area_light.cuh"
#include "printing.h"
#include "monte_carlo_settings.h"
#include "image_settings.h"

__host__
AreaLight::AreaLight() : Triangle(vec4(0), vec4(0), vec4(0)){
    this->diffuse_p = vec3(0);
}

__host__
AreaLight::AreaLight(vec4 v0, vec4 v1, vec4 v2, vec3 diffuse_p) : Triangle(v0, v1, v2){
    this->diffuse_p = diffuse_p;
}