#include "radiance_volume.h"
#include "radiance_volumes_settings.h"
#include "hemisphere_helpers.h"
#include "path_tracing.h"
#include "printing.h"
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

// Gets the incoming radiance values from all grid samples and
// populates radiance_grid with the estimates
// NOTE: This should be called before any radiance_volumes are instantiated
// in the scene by surfaces or these surfaces will be taken into account
void RadianceVolume::get_radiance_estimate(vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes){
    // Path trace a ray in the direction from the centre of hemisphere towards
    // the centre of each sector to determine the radiance incoming from that direction
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            // Get the 3D coordinates of the centre of the sector
            float sector_x, sector_y, sector_z;
            map(
                (x + 0.5f)/(float)GRID_RESOLUTION,
                (y + 0.5f)/(float)GRID_RESOLUTION,
                sector_x, 
                sector_y, 
                sector_z
            );
            sector_x *= DIAMETER;
            sector_y *= DIAMETER;
            sector_z *= DIAMETER;
            // Convert the position to world space
            vec4 world_position = this->transformation_matrix * vec4(sector_x, sector_y, sector_z, 1.f);
            // Calculate the direction and starting position of the ray
            vec4 dir = world_position - vec4(this->position.x, this->position.y, this->position.z, 0.f);
            vec4 start = this->position + 0.00001f * dir;
            start.w = 1.f;
            // Create the ray and path trace to find the radiance in that direction
            Ray ray = Ray(start, dir);
            this->radiance_grid[x][y] = path_trace(true, ray, surfaces, light_planes, 0);
        }
    }
}

// Builds a radiance volume out of Surfaces, where each surfaces colour
// represents the incoming radiance at that position from that angle
void RadianceVolume::build_radiance_volume_shapes(vector<Surface>& surfaces){
    // Get its vertices
    vector<vector<vec4>> vertices;
    this->get_vertices(vertices);
    // Build Surfaces using the vertices
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            vec4 v1 = vertices[x][y];
            vec4 v2 = vertices[x+1][y];
            vec4 v3 = vertices[x][y+1];
            vec4 v4 = vertices[x+1][y+1];
            float r1 = ((float) rand() / (RAND_MAX));
            float r2 = ((float) rand() / (RAND_MAX));
            float r3 = ((float) rand() / (RAND_MAX));
            surfaces.push_back(Surface(v1, v3, v2, Material(this->radiance_grid[x][y])));
            surfaces.push_back(Surface(v2, v3, v4, Material(this->radiance_grid[x][y])));
        }
    }
}

// Builds a radiance volume out of Surfaces, where each surfaces colour
// represents the incoming radiance at that position from that angle
void RadianceVolume::build_radiance_magnitude_volume_shapes(vector<Surface>& surfaces){
    // Get its vertices
    vector<vector<vec4>> vertices;
    this->get_vertices(vertices);
    // Find the max radiance magnitude of the hemisphere
    float max_radiance = 0.0001f;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            if(length(this->radiance_grid[x][y]) > max_radiance){
                max_radiance = length(this->radiance_grid[x][y]);
            }
        }
    }
    // Build Surfaces using the vertices
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            vec4 v1 = vertices[x][y];
            vec4 v2 = vertices[x+1][y];
            vec4 v3 = vertices[x][y+1];
            vec4 v4 = vertices[x+1][y+1];
            float r1 = ((float) rand() / (RAND_MAX));
            float r2 = ((float) rand() / (RAND_MAX));
            float r3 = ((float) rand() / (RAND_MAX));
            vec3 colour = vec3(1.f) - length(this->radiance_grid[x][y])/max_radiance * vec3(1.f);
            surfaces.push_back(Surface(v1, v3, v2, Material(colour)));
            surfaces.push_back(Surface(v2, v3, v4, Material(colour)));
        }
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

// Gets the total radiance incident on the point from all incoming directions with
// current recorded estimates
vec3 RadianceVolume::get_total_radiance(const Intersection& intersection, vector<Surface *> surfaces){
    vec3 total_radiance = vec3(0);
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            // Get the coordinates on the unit hemisphere
            float x_h, y_h, z_h;
            map(x/(float)GRID_RESOLUTION, y/(float)GRID_RESOLUTION, x_h, y_h, z_h);
            // Scale to the correct diameter desired of the hemisphere
            x_h *= DIAMETER;
            y_h *= DIAMETER;
            z_h *= DIAMETER;
            // Convert to world space
            vec4 world_position = this->transformation_matrix * vec4(x_h, y_h, z_h, 1.f);
            vec3 world_position3 = vec3(world_position.x, world_position.y, world_position.z);
            // Get the direction
            vec3 dir = normalize(world_position3 - vec3(this->position));
            // Get the angle between the dir vector and the normal
            float cos_theta = dot(dir, this->normal); // No need to divide by lengths as they have been normalized
            total_radiance += cos_theta * this->radiance_grid[x][y];
        }
    }
    total_radiance /= (float)(GRID_RESOLUTION * GRID_RESOLUTION) * (1 / (2 * M_PI));
    total_radiance *= surfaces[intersection.index]->get_material().get_diffuse_c();
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