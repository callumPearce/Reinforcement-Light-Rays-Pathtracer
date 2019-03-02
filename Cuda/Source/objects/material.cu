#include "material.cuh"

__host__
Material::Material(vec3 diffuse_c){
    this->diffuse_c = diffuse_c;
}
