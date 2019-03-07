#ifndef RADIANCE_MAP_H
#define RADIANCE_MAP_H

#include <glm/glm.hpp>
#include "radiance_volume.cuh"
#include "ray.cuh"
#include "radiance_tree.cuh"
#include "printing.h"
#include "radiance_volumes_settings.h"
#include "hemisphere_helpers.cuh"
//cuRand
#include <curand.h>
#include <curand_kernel.h>

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

        __host__
        void get_radiance_volumes_count(Surface* surfaces, int surfaces_count);

        __host__
        void uniformly_sample_radiance_volumes(Surface* surfaces, int surfaces_count, std::vector<RadianceVolume>& temp_rvs);

    public:

        // Pointer to a list of pointers
        RadianceVolume* radiance_volumes;
        // RadianceTree* radiance_tree;
        int radiance_volumes_count = 0;

        // Constructor
        __host__
        RadianceMap(Surface* surfaces, int surfaces_count, std::vector<RadianceVolume>& temp_rvs);

        // Get the radiance estimate for every radiance volume in the RadianceMap
        __device__
        void get_radiance_estimates(curandState* volume_rand_state, Surface* surfaces, AreaLight* light_planes);

        // Get radiance estimate at the intersection point
        // vec3 get_irradiance_estimate(const Intersection& intersection, Surface* surfaces);

        // Calculate the Gaussian filter for radiance contribution
        __device__
        float calculate_gaussian_filter(float volume_distance, float furthest_volume_distance);

        // Given an intersection point, importance sample a ray direction according to the
        // cumulative distribution formed by the closest RadianceVolume's radiance_map
        __device__
        RadianceVolume* importance_sample_ray_direction(curandState* d_rand_state, const Intersection& intersection, int& sector_x, int& sector_y, int x, int y, vec4& sampled_direction);

        // Performs the temporal difference update for the radiance volume passed in given the sampled ray direction lead to the intersection
        __device__
        void temporal_difference_update_radiance_volume_sector(RadianceVolume* current_radiance_volume, int current_sector_x, int current_sector_y, Intersection& intersection, Surface* surfaces, AreaLight* light_planes);

        // Set the voronoi colours of all radiance volumes in the scene in the first entry of the radiance_grid[0][0]
        __host__
        void set_voronoi_colours(std::vector<RadianceVolume>& temp_rvs);

        // Get the voronoi colour of the interesection point from the closest radiance volume
        __device__
        vec3 get_voronoi_colour(const Intersection& intersection);

        // Find the closest radiance volume in linear time by traversing the list of radiance volumes
        __device__
        RadianceVolume* get_closest_radiance_volume_linear(float max_dist, vec4 position, vec4 normal);

};

#endif