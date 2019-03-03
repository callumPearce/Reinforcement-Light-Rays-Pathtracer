#include "radiance_volume_comparator.cuh"

RadianceVolumeComparator::RadianceVolumeComparator(RadianceVolume* radiance_volume, float distance){
    this->radiance_volume = radiance_volume;
    this->distance = distance;
}

bool operator<(const RadianceVolumeComparator & lhs, const RadianceVolumeComparator & rhs){
    return lhs.distance < rhs.distance;
}
