#ifndef RADIANCE_MAP_H
#define RADIANCE_MAP_H

#include <glm/glm.hpp>
#include "radiance_volume.h"
#include "ray.h"
#include "radiance_tree.h"
#include "printing.h"
#include "radiance_volumes_settings.h"
#include "interpolation.h"
#include "hemisphere_helpers.h"

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

        // Given an intersection point, importance sample a ray direction according to the
        // cumulative distribution formed by the closest RadianceVolume's radiance_map
        RadianceVolume* importance_sample_ray_direction(const Intersection& intersection, int& sector_x, int& sector_y, vec4& sampled_direction);

        // Normalizes all RadianceVolumes radiance values i.e. their grid values
        // all sum to 1 (taking the length of each vec3)
        void update_radiance_distributions();

        // Performs the temporal difference update for the radiance volume passed in given the sampled ray direction lead to the intersection
        void temporal_difference_update_radiance_volume_sector(RadianceVolume* current_radiance_volume, int current_sector_x, int current_sector_y, Intersection& intersection, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes);
};

#endif