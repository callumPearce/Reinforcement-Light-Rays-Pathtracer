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

        // Initialises square grid as a 2D std::vector of vec3s (radiance stores for each angle)
        __host__
        void initialise_radiance_grid(Surface* surfaces);

        // Initialises square grid as a 2D vector of floats representing the radiance distribution
        // for the radiance volume
        __host__
        void initialise_radiance_distribution();

        // Initialises thge alpha-value for all state-action pairs to be 1.f
        __host__
        void initialise_visits();

    public:
        vec4 position;
        float radiance_grid [GRID_RESOLUTION * GRID_RESOLUTION ];
        float radiance_distribution [GRID_RESOLUTION * GRID_RESOLUTION];
        unsigned int visits [GRID_RESOLUTION * GRID_RESOLUTION];
        // For coordinate system
        float irradiance_accum;
        unsigned int surface_index;
        vec3 normal;
        mat4 transformation_matrix;
        int index = -1;

        // Default Constructor
        __host__ __device__
        RadianceVolume();
        
        // Constructor
        __host__
        RadianceVolume(Surface* surfaces, vec4 position, vec4 normal, unsigned int surface_index, int idx);

        // Constructor for radiance volume which will be rendered
        // Constructor for loading radiance volume to render
        __host__
        RadianceVolume(vec4 position, vec3 normal, std::vector<float>& q_vals);

        // Update the transformation matrix with the current normal and position values
        __host__
        void update_transformation_matrix();

        // Returns a list of vertices for the generated radiance volume
        __host__
        std::vector<vec4> get_vertices();

        __device__
        void expected_sarsa_irradiance(Surface* surfaces, const float update, const int sector_x, const int sector_y);

        // __device__
        // float q_learning_irradiance(Surface* surfaces);

        // Updates the current irradiance estimate
        __device__
        void update_irradiance(Surface* surfaces, const float update, const int sector_x, const int sector_y);

        // Normalizes this RadianceVolume so that all radiance values 
        // i.e. their grid values all sum to 1 (taking the length of each vec3)
        __device__
        void update_radiance_distribution();

        // Samples a direction from the radiance_distribution of this radiance
        // volume
        __device__
        vec4 sample_direction_from_radiance_distribution(curandState* d_rand_state, int pixel_x, int pixel_y, int& sector_x, int& sector_y, float& pdf);

        // Samples a direction from the radiance volume using binary search for the sector
        __device__
        vec4 sample_max_direction_from_radiance_distribution(curandState* d_rand_state, int pixel_x, int pixel_y, int& sector_x, int& sector_y);

        // Performs a temporal difference update for the current radiance volume for the incident
        // radiance in the sector specified with the intersection surfaces irradiance value
        __device__
        void temporal_difference_update(float sector_irradiance, int sector_x, int sector_y, Surface* surface);

        // Gets the current irradiance estimate for the radiance volume
        __device__
        float get_irradiance_estimate();

        // Sets a voronoi colour for the radiance volume (random colour) in the first entry of its radiance grid
        __host__
        void set_voronoi_colour();

        // Gets the voronoi colour of the radiance volume
        __device__
        vec3 get_voronoi_colour();

        // Conver the radiance volumes cumulative distribution to a distribution
        __host__
        void convert_radiance_distribution();

        // Write the radiance volumes Q-values out to a file
        __host__
        void write_volume_to_file(std::string filename);

        // Set the radiance distribution of the radiance volume to the
        // supplied q_vals
        void set_q_vals(std::vector<float>& q_vals);

        // Build the radiance volumes surfaces and add it to the list based
        // on the radiance distribution values
        void build_surfaces(std::vector<Surface>& surfaces);

        // Read radiance volumes in from a file and populate the list rvs
        // rvs with them
        __host__
        static void read_radiance_volumes_from_file(
            std::string fname, 
            std::vector<RadianceVolume>& rvs
        );

        // Read the list of radiance volumes from a file and build surfaces of them
        __host__
        static void read_radiance_volumes_to_surfaces(
            std::string fname,
            std::vector<Surface>& surfaces
        );
};

#endif
