#ifndef RADIANCE_VOLUME_H
#define RADIANCE_VOLUME_H

#include <glm/glm.hpp>
#include "surface.h"
#include "radiance_volumes_settings.h"
#include "hemisphere_helpers.h"
#include "default_path_tracing.h"
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
        vec4 position;
        std::vector<std::vector<vec3>> radiance_grid;
        // For coordinate system
        vec3 normal;
        mat4 transformation_matrix;

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
        static void map(float x, float y, float& x_ret, float& y_ret, float& z_ret);

        // Initialises square grid as a 2D std::vector of vec3s (radiance stores for each angle)
        void initialise_radiance_grid();

    public:
        // Default Constructor
        RadianceVolume();
        
        // Constructor
        RadianceVolume(vec4 position, vec4 normal);

        // Update the transformation matrix with the current normal and position values
        void update_transformation_matrix();

        // Get vertices of radiance volume in world space
        void get_vertices(std::vector<std::vector<vec4>>& vertices);

        // Gets the incoming radiance values from all grid samples and
        // populates radiance_grid with the estimates
        // NOTE: This should be called before any radiance_volumes are instantiated
        // in the scene by surfaces or these surfaces will be taken into account
        void get_radiance_estimate(std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes);

        // Builds a radiance volume out of Surfaces, where each surfaces colour
        // represents the incoming radiance at that position from that angle
        void build_radiance_volume_shapes(std::vector<Surface>& surfaces);

        // Builds a radiance volume out of Surfaces, where each surfaces colour
        // represents the magnitude of incoming radiance compared to the other directions
        void build_radiance_magnitude_volume_shapes(std::vector<Surface>& surfaces);

        // Gets the irradiance for an intersection point by solving the rendering equations (summing up 
        // radiance from all directions whilst multiplying by BRDF and cos(theta))
        vec3 get_irradiance(const Intersection& intersection, std::vector<Surface *> surfaces);

        // Normalizes this RadianceVolume so that all radiance values 
        // i.e. their grid values all sum to 1 (taking the length of each vec3)
        void RadianceVolume::normalize_radiance_volume();

        // Getters
        vec4 get_position();
        vec3 get_normal();

        // Setters
        void set_position(vec4 position);
        void set_normal(vec3 normal);
};

#endif