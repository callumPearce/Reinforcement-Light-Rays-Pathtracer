#include "surface.cuh"
#include <iostream>

Surface::Surface() : Triangle(vec4(0.f), vec4(0.f), vec4(0.f)){
    this->material = Material(vec3(0.f));
}

Surface::Surface(vec4 v0, vec4 v1, vec4 v2, Material material) : Triangle(v0, v1, v2){
    this->material = material;
}
