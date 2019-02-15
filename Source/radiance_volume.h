#ifndef RADIANCE_VOLUME_H
#define RADIANCE_VOLUME_H

#include <vector>
#include <glm/glm.hpp>

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Defines a single radiance volume in the scene as described in:
    https://www.researchgate.net/publication/3208677_The_Irradiance_Volume
*/
class RadianceVolume{

    private:
        vec4 position;
        vector<vector<vec3>> radiance_grid;
        // For coordinate system
        vec3 normal;
        vec3 normal_T;
        vec3 normal_B;

        // Initialises square grid as a 2D vector of vec3s (radiance stores for each angle)
        void initialise_radiance_grid();

    public:
        // Constructor
        RadianceVolume(vec4 position, vec4 normal);

        // Radiance Volume functions

        // Get vertices of radiance volume in world space
        void get_vertices(vector<vector<vec4>>& vertices);

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
        void map(float x, float y, float& x_ret, float& y_ret, float& z_ret);

        // Getters
        vec4 get_position();
        vec3 get_normal();

        // Setters
        void set_position(vec4 position);
        void set_normal(vec3 normal);
};

#endif