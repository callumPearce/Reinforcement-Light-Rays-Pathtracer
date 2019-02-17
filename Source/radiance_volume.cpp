#include "radiance_volume.h"
#include "radiance_volumes_settings.h"
#include "hemisphere_helpers.h"
#include <iostream>

RadianceVolume::RadianceVolume(vec4 position, vec4 normal){
    initialise_radiance_grid();
    set_position(position);
    set_normal(vec3(normal.x, normal.y, normal.z));

    // Create the transformation matrix for this hemisphere: local->world
    this->transformation_matrix = create_transformation_matrix(normal, position);
}

// Updates the transformation matrix with the current set values of the normal and position
void RadianceVolume::update_transformation_matrix(){
    this->transformation_matrix = create_transformation_matrix(normal, position);
} 

// Intialises a 2D grid to store radiance values at each grid point
void RadianceVolume::initialise_radiance_grid(){
    for (int x = 0; x < GRID_RESOLUTION; x++){
        vector<vec3> grid_row;
        for (int y = 0; y < GRID_RESOLUTION; y++){
            grid_row.push_back(vec3(0));
        }
        this->radiance_grid.push_back(grid_row);
    }
}

// Returns a list of vertices for the generated radiance volume
void RadianceVolume::get_vertices(vector<vector<vec4>>& vertices){
    // For every grid coordinate, add the corresponding 3D world coordinate
    for (int x = 0; x <= GRID_RESOLUTION; x++){
        vector<vec4> vertices_row;
        for (int y = 0; y <= GRID_RESOLUTION; y++){
            // Get the coordinates on the unit hemisphere
            float x_h, y_h, z_h;
            map(x/(float)GRID_RESOLUTION, y/(float)GRID_RESOLUTION, x_h, y_h, z_h);
            // Scale to the correct diameter desired of the hemisphere
            x_h *= DIAMETER;
            y_h *= DIAMETER;
            z_h *= DIAMETER;
            // Convert to world space
            vec4 world_position = this->transformation_matrix * vec4(x_h, y_h, z_h, 1.f);
            // Add the point to vertices_row
            vertices_row.push_back(world_position);
        }
        vertices.push_back(vertices_row);
    }
}

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
void RadianceVolume::map(float x, float y, float& x_ret, float& y_ret, float& z_ret) {
    float xx, yy, offset, theta, phi;
    x = 2*x - 1;
    y = 2*y - 1;
    if (y > -x) { // Above y = -x
        if (y < x) { // Below y = x
            xx = x;
            if (y > 0) { // Above x-axis
                /*
                * Octant 1
                */
                offset = 0;
                yy = y;
            } 
            else { // Below and including x-axis
                /*
                * Octant 8
                */
                offset = (7*M_PI)/4;
                yy = x + y;
            }
        } 
        else { // Above and including y = x
            xx = y;
            if (x > 0) { // Right of y-axis
                /*
                * Octant 2
                */
                offset = M_PI/4;
                yy = (y - x);
            } 
            else { // Left of and including y-axis
                /*
                * Octant 3
                */
                offset = (2*M_PI)/4;
                yy = -x;
            }
        }
    } 
    else { // Below and including y = -x
        if (y > x) { // Above y = x
            xx = -x;
            if (y > 0) { // Above x-axis
                /*
                * Octant 4
                */
                offset = (3*M_PI)/4;
                yy = -x - y;
            } 
            else { // Below and including x-axis
                /*
                * Octant 5
                */
                offset = (4*M_PI)/4;
                yy = -y;
            }
        } 
        else { // Below and including y = x
            xx = -y;
            if (x > 0) { // Right of y-axis
                /*
                * Octant 7
                */
                offset = (6*M_PI)/4;
                yy = x;
            } 
            else { // Left of and including y-axis
                if (y != 0) {
                    /*
                    * Octant 6
                    */
                    offset = (5*M_PI)/4;
                    yy = x - y;
                } 
                else {
                    /*
                    * Origincreate_normal_coordinate_system
                    */
                    x_ret = 0.f;
                    y_ret = 1.f;
                    z_ret = 0.f;
                    return;
                }
            }
        }
    }
    theta = acos(1 - xx*xx);
    phi = offset + (M_PI/4)*(yy/xx);
    x_ret = sin(theta)*cos(phi);
    y_ret = cos(theta);
    z_ret = sin(theta)*sin(phi);
}

// Getters
vec4 RadianceVolume::get_position(){
    return this->position;
}

vec3 RadianceVolume::get_normal(){
    return this->normal;
}

// Setters
void RadianceVolume::set_position(vec4 position){
    this->position = position;
}

void RadianceVolume::set_normal(vec3 normal){
    this->normal = normal;
}