#ifndef RADIANCE_VOLUME_H
#define RADIANCE_VOLUME_H

#include <glm/glm.hpp>
#include "surface.cuh"
#include "radiance_volumes_settings.h"
#include "hemisphere_helpers.cuh"
#include "default_path_tracing.cuh"
#include "printing.h"

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Defines a single radiance hemisphere in the scene as described in:
    https://www.researchgate.net/publication/3208677_The_Irradiance_Volume
    Note: I have decided to call each hemisphere a radiance volume and all
    of the hemishperes together form the RadianceMap of the scene
*/
class RadianceVolume{

    private:

        /*
        * This function takes a point in the unit square,
        * and maps it to a point on the unit hemisphere.
        *
        * Copyright 1994 Kenneth Chiu
        *
        * This code may be freely distributed and used
        * for any purpose, commercial or non-commercial,
        * as long as attribution is maintained.
        */
       __device__
        static void map(float x, float y, float& x_ret, float& y_ret, float& z_ret);

        // Initialises square grid as a 2D std::vector of vec3s (radiance stores for each angle)
        __host__
        void initialise_radiance_grid();

        // Initialises square grid as a 2D vector of floats representing the radiance distribution
        // for the radiance volume
        __host__
        void initialise_radiance_distribution();

        // Initialises thge alpha-value for all state-action pairs to be 1.f
        __host__
        void initialise_visits();

    public:
        bool initialised = false;
        vec4 position;
        float radiance_grid [GRID_RESOLUTION * GRID_RESOLUTION * 3];
        float radiance_distribution [GRID_RESOLUTION * GRID_RESOLUTION];
        unsigned int visits [GRID_RESOLUTION * GRID_RESOLUTION];
        // For coordinate system
        vec3 normal;
        mat4 transformation_matrix;

        // Default Constructor
        __host__ __device__
        RadianceVolume();
        
        // Constructor
        __host__
        RadianceVolume(vec4 position, vec4 normal);

        // Update the transformation matrix with the current normal and position values
        __host__
        void update_transformation_matrix();

        // Get vertices of radiance volume in world space
        __device__
        vec4* get_vertices();

        // // Gets the incoming radiance values from all grid samples and
        // // populates radiance_grid with the estimates
        // // NOTE: This should be called before any radiance_volumes are instantiated
        // // in the scene by surfaces or these surfaces will be taken into account
        // __device__
        // void get_radiance_estimate_per_sector(Surface* surfaces, AreaLight* light_planes);

        // // Builds a radiance volume out of Surfaces, where each surfaces colour
        // // represents the incoming radiance at that position from that angle
        // __device__
        // void build_radiance_volume_shapes(std::vector<Surface>& surfaces);

        // // Builds a radiance volume out of Surfaces, where each surfaces colour
        // // represents the magnitude of incoming radiance compared to the other directions
        // __device__
        // void build_radiance_magnitude_volume_shapes(std::vector<Surface>& surfaces);

        // Gets the irradiance for an intersection point by solving the rendering equations (summing up 
        // radiance from all directions whilst multiplying by BRDF and cos(theta))
        __device__
        vec3 get_irradiance(const Intersection& intersection, Surface* surfaces);

        // Normalizes this RadianceVolume so that all radiance values 
        // i.e. their grid values all sum to 1 (taking the length of each vec3)
        __device__
        void update_radiance_distribution();

        // Samples a direction from the radiance_distribution of this radiance
        // volume
        __device__
        vec4 sample_direction_from_radiance_distribution(curandState* d_rand_state, int x, int y, int& sector_x, int& sector_y);

        // Performs a temporal difference update for the current radiance volume for the incident
        // radiance in the sector specified with the intersection surfaces irradiance value
        __device__
        void temporal_difference_update(vec3 next_irradiance, int sector_x, int sector_y);

        // Sets a voronoi colour for the radiance volume (random colour) in the first entry of its radiance grid
        __host__
        void set_voronoi_colour();

        // Gets the voronoi colour of the radiance volume
        __device__
        vec3 get_voronoi_colour();
};

#endif