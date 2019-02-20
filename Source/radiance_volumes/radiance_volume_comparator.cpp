#include "radiance_volume_comparator.h"

RadianceVolumeComparator::RadianceVolumeComparator(RadianceVolume* radiance_volume, float distance){
    this->radiance_volume = radiance_volume;
    this->distance = distance;
}

bool operator<(const RadianceVolumeComparator & lhs, const RadianceVolumeComparator & rhs){
    return lhs.distance < rhs.distance;
}

// Getters
RadianceVolume* RadianceVolumeComparator::get_radiance_volume(){
    return this->radiance_volume;
}

float RadianceVolumeComparator::get_distance(){
    return this->distance;
}