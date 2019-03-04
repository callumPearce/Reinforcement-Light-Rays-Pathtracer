#include "radiance_volume_comparator.cuh"

__device__
RadianceVolumeComparator::RadianceVolumeComparator(RadianceVolume* radiance_volume, float distance){
    this->radiance_volume = radiance_volume;
    this->distance = distance;
}

__device__
bool operator<(const RadianceVolumeComparator & lhs, const RadianceVolumeComparator & rhs){
    return lhs.distance < rhs.distance;
}
