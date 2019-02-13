#include "surface.h"
#include <iostream>

Surface::Surface(vec4 v0, vec4 v1, vec4 v2, Material material) : Triangle(v0, v1, v2){
    set_material(material);
}

// Getters
Material Surface::get_material(){
    return this->material;
}

// Setters
void Surface::set_material(Material material){
    this->material = material;
}