#include "radiance_volume.cuh"
#include <iostream>


__device__ __host__
RadianceVolume::RadianceVolume(){
    this->position = vec4(0.f);
    this->normal = vec3(0.f);
    this->transformation_matrix = mat4(0.f);
}

__host__
RadianceVolume::RadianceVolume(vec4 position, vec4 normal){
    initialise_radiance_grid();
    initialise_radiance_distribution();
    initialise_visits();

    this->position = position;
    this->normal = vec3(normal.x, normal.y, normal.z);

    // Create the transformation matrix for this hemisphere: local->world
    this->transformation_matrix = create_transformation_matrix(normal, position);
    this->initialised = true;
}

// Updates the transformation matrix with the current set values of the normal and position
__host__
void RadianceVolume::update_transformation_matrix(){
    this->transformation_matrix = create_transformation_matrix(normal, position);
} 

// Intialises a 2D grid to store radiance values at each grid point
__host__
void RadianceVolume::initialise_radiance_grid(){
    // this->radiance_grid = vec3[ GRID_RESOLUTION * GRID_RESOLUTION ];
    float initial = 1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION);
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            this->radiance_grid[ x*GRID_RESOLUTION + y ] = initial;
        }
    }
}

// Initialise radiance distribution to be equal in all angles initially
__host__
void RadianceVolume::initialise_radiance_distribution(){
    // this->radiance_distribution = new float[ GRID_RESOLUTION * GRID_RESOLUTION ];
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            this->radiance_distribution[ x*GRID_RESOLUTION + y ] = 1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION);        
        }
    }
}

// Initialises the alpha values (weighting of state-action pairs) to be 1
__host__
void RadianceVolume::initialise_visits(){
    // this->visits = new float[ GRID_RESOLUTION * GRID_RESOLUTION ];
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            this->visits[ x*GRID_RESOLUTION + y ] = 0;
        }
    }
}


// Returns a list of vertices for the generated radiance volume
__device__
vec4* RadianceVolume::get_vertices(){
    vec4* vertices = new vec4[ (GRID_RESOLUTION+1) * (GRID_RESOLUTION+1) ];
    // For every grid coordinate, add the corresponding 3D world coordinate
    for (int x = 0; x <= GRID_RESOLUTION; x++){
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
            vertices[ x*GRID_RESOLUTION + y ] = world_position;
        }
    }
    return vertices;
}

// Gets the irradiance for an intersection point by solving the rendering equations (summing up 
// radiance from all directions whilst multiplying by BRDF and cos(theta))
__device__
float RadianceVolume::get_irradiance(const Intersection& intersection, Surface* surfaces){
    float irradiance = 0.f;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            // Get the coordinates on the unit hemisphere
            float x_h, y_h, z_h;
            map(x/(float)GRID_RESOLUTION, y/(float)GRID_RESOLUTION, x_h, y_h, z_h);
            // Convert to world space
            vec4 world_position = this->transformation_matrix * vec4(x_h, y_h, z_h, 1.f);
            vec3 world_position3 = vec3(world_position.x, world_position.y, world_position.z);
            // Get the direction
            vec3 dir = normalize(world_position3 - vec3(this->position));
            // Get the angle between the dir std::vector and the normal
            float cos_theta = dot(dir, this->normal); // No need to divide by lengths as they have been normalized
            irradiance += cos_theta * this->radiance_grid[ x*GRID_RESOLUTION + y ];
        }
    }
    irradiance /= ((float)(GRID_RESOLUTION * GRID_RESOLUTION)) * (1.f / (2.f * (float)M_PI));
    irradiance *= length((surfaces[intersection.index].material.diffuse_c) / (float)M_PI);
    return irradiance;
}

// Normalizes this RadianceVolume so that all radiance values 
// i.e. their grid values all sum to 1 (taking the length of each vec3)
__device__
void RadianceVolume::update_radiance_distribution(){

    // Get the total radiance from all directions (as a float)
    float total = 0.00000001f;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            total += this->radiance_grid[ x*GRID_RESOLUTION + y ];
        }
    }
    // Use this total to convert all radiance_grid values into probabilities
    // and store in the radiance_distribution
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            float radiance = this->radiance_grid[ x*GRID_RESOLUTION + y ];
            this->radiance_distribution[ x*GRID_RESOLUTION + y ] = radiance/total;
            // float new_val = this->radiance_grid[ x*GRID_RESOLUTION + y ]/total;
            // atomicExch(&(this->radiance_distribution[ x*GRID_RESOLUTION + y ]), new_val);
        }
    }
}

// Samples a direction from the radiance volume
// volume
__device__
vec4 RadianceVolume::sample_direction_from_radiance_distribution(curandState* d_rand_state, int pixel_x, int pixel_y, int& sector_x, int& sector_y){
    
    // Generate a random float uniformly 
    float r = curand_uniform(&d_rand_state[pixel_x*SCREEN_HEIGHT + pixel_y]);

    // Sample from the inverse cumulative distribution
    float cumulative_sum = 0.f;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            cumulative_sum += this->radiance_distribution[ x*GRID_RESOLUTION + y ];
            // We have found where in the inverse cumulative distribution our
            // sample is. There for return an anlge at this location
            if ( r <= cumulative_sum ){
                sector_x = x;
                sector_y = y;
                // Get the coordinates on the unit hemisphere
                float x_h, y_h, z_h;
                map(x/(float)GRID_RESOLUTION, y/(float)GRID_RESOLUTION, x_h, y_h, z_h);
                // Convert to world space
                vec4 world_position = this->transformation_matrix * vec4(x_h, y_h, z_h, 1.f);
                vec3 world_position3 = vec3(world_position.x, world_position.y, world_position.z);
                // Get the direction
                return vec4(normalize(world_position3 - vec3(this->position)),1.f);
            }
        }
    }
    return vec4(0.f);
}

// Performs a temporal difference update for the current radiance volume for the incident
// radiance in the sector specified with the intersection surfaces irradiance value
__device__
void RadianceVolume::temporal_difference_update(float next_irradiance, int sector_x, int sector_y){

    int sector_location = sector_x*GRID_RESOLUTION + sector_y;

    // Calculate alpha and update the radiance grid values and increment the number of visits
    unsigned int vs = this->visits[ sector_location ];
    float alpha = 1.f / (1.f + (float)vs);

    // assert(alpha <= 1.00000f);

    // Calculate the new update value
    float radiance = this->radiance_grid[ sector_location ];
    float update = ((1.f - (alpha)) * radiance) + (alpha * next_irradiance);
    update = update > (float)RADIANCE_THRESHOLD ? update : (float)RADIANCE_THRESHOLD;

    // assert(update.x < 1.f);
    // assert(update.y < 1.f);
    // assert(update.z < 1.f);

    // Update the radiance grid and the alpha value
    atomicInc(&(this->visits[ sector_location ]), vs+1);
    atomicExch(&(this->radiance_grid[ sector_location     ]), update);
}

// Sets a voronoi colour for the radiance volume (random colour) in the first entry of its radiance grid
__host__
void RadianceVolume::set_voronoi_colour(){
    float r = ((float) rand() / (RAND_MAX));
    float g = ((float) rand() / (RAND_MAX));
    float b = ((float) rand() / (RAND_MAX));
    this->radiance_grid[0] = r;
    this->radiance_grid[1] = g;
    this->radiance_grid[2] = b;
}

// Gets the voronoi colour of the radiance volume
__device__
vec3 RadianceVolume::get_voronoi_colour(){
    vec3 colour(0.f);
    colour.x = this->radiance_grid[0];
    colour.y = this->radiance_grid[1];
    colour.z = this->radiance_grid[2];
    return colour;
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
__device__ 
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
