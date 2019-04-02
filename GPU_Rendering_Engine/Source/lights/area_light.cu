#include "area_light.cuh"
#include "printing.h"
#include "monte_carlo_settings.h"
#include "image_settings.h"

__host__
AreaLight::AreaLight() : Triangle(vec4(0), vec4(0), vec4(0)){
    this->diffuse_p = vec3(0);
    this->luminance = 0.f;
}

__host__
AreaLight::AreaLight(vec4 v0, vec4 v1, vec4 v2, vec3 diffuse_p) : Triangle(v0, v1, v2){
    this->diffuse_p = diffuse_p;
    
    float max_rgb = max(diffuse_p.x, diffuse_p.y);
    max_rgb = max(diffuse_p.z, max_rgb);

    float min_rgb = min(diffuse_p.x, diffuse_p.y);
    min_rgb = min(diffuse_p.z, min_rgb);

    this->luminance = 0.5f * (max_rgb + min_rgb);
}