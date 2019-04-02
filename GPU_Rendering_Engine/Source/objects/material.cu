#include "material.cuh"

__host__
Material::Material(vec3 diffuse_c){
    this->diffuse_c = diffuse_c;

    float max_rgb = max(diffuse_c.x, diffuse_c.y);
    max_rgb = max(diffuse_c.z, max_rgb);

    float min_rgb = min(diffuse_c.x, diffuse_c.y);
    min_rgb = min(diffuse_c.z, min_rgb);

    this->luminance = 0.5f * (max_rgb + min_rgb);
}
