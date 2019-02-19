#ifndef RADIANCE_VOLUME_COMPARATOR_H
#define RADIANCE_VOLUME_COMPARATOR_H

#include <glm/glm.hpp>
#include "radiance_volume.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

class RadianceVolumeComparator{

    private:
        float distance;
        RadianceVolume radiance_volume;

    public:
        
        // Constructor
        RadianceVolumeComparator(RadianceVolume radiance_volume, float distance);

        // Weak ordering
        friend bool operator<(const RadianceVolumeComparator& lhs, const RadianceVolumeComparator& rhs);

        // Getters
        RadianceVolume get_radiance_volume();
        float get_distance();
};

#endif