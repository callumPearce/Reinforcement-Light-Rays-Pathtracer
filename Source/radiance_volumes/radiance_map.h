#ifndef RADIANCE_MAP_H
#define RADIANCE_MAP_H

#include <glm/glm.hpp>
#include "radiance_volume.h"
#include "ray.h"
#include "radiance_tree.h"
#include "printing.h"
#include "radiance_volumes_settings.h"
#include "interpolation.h"

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Defines a single radiance volume in the scene as described in:
    https://www.researchgate.net/publication/3208677_The_Irradiance_Volume
*/
class RadianceMap{

    private:
        std::vector<RadianceVolume*> radiance_volumes;
        std::unique_ptr<RadianceTree> radiance_tree;
        void uniformly_sample_radiance_volumes(Surface surface);

    public:

        // Constructor
        RadianceMap(std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, std::vector<Surface>& surfaces_builder);

        // Builds all RadianceVolumes which are part of the RadianceMap into the scene
        void build_radiance_map_shapes(std::vector<Surface>& surfaces);

        // Get the radiance estimate for every radiance volume in the RadianceMap
        void get_radiance_estimates(std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes);

        // Get radiance estimate at the intersection point
        vec3 get_irradiance_estimate(const Intersection& intersection, std::vector<Surface *> surfaces);

        // Calculate the Gaussian filter for radiance contribution
        float calculate_gaussian_filter(float volume_distance, float furthest_volume_distance);
};

#endif