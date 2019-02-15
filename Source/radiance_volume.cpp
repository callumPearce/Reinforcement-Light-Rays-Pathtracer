#include "radiance_volume.h"

RadianceVolume::RadianceVolume(vec4 position, vec4 normal){
    set_position(position);
    set_normal(normal);
}

// Getters
vec4 RadianceVolume::get_position(){
    return this->position;
}

vec4 RadianceVolume::get_normal(){
    return this->normal;
}

// Setters
void RadianceVolume::set_position(vec4 position){
    this->position = position;
}

void RadianceVolume::set_normal(vec4 normal){
    this->normal = normal;
}