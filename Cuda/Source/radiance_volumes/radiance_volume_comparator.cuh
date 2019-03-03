#ifndef RADIANCE_VOLUME_COMPARATOR_H
#define RADIANCE_VOLUME_COMPARATOR_H

#include <glm/glm.hpp>
#include "radiance_volume.cuh"

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

class RadianceVolumeComparator{

    private:

    public:

        float distance;
        RadianceVolume* radiance_volume;
        
        // Constructor
        RadianceVolumeComparator(RadianceVolume* radiance_volume, float distance);

        // Weak ordering
        friend bool operator<(const RadianceVolumeComparator& lhs, const RadianceVolumeComparator& rhs);

};

#endif