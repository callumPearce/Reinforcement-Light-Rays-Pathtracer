#include "material.h"

Material::Material(vec3 diffuse_c){
    set_diffuse_c(diffuse_c);
}

//Getters
vec3 Material::get_diffuse_c(){
    return this->diffuse_c;
}

// Setters
void Material::set_diffuse_c(vec3 diffuse_c){
    this->diffuse_c = diffuse_c;
}