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
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            this->radiance_grid[ x*GRID_RESOLUTION + y ] = vec3((1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION))/3.f);
        }
    }
}

// Initialise radiance distribution to be equal in all angles initially
__host__
void RadianceVolume::initialise_radiance_distribution(){
    // this->radiance_distribution = new float[ GRID_RESOLUTION * GRID_RESOLUTION ];
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            this->radiance_distribution[ x*GRID_RESOLUTION + y] = 1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION);
        }
    }
}

// Initialises the alpha values (weighting of state-action pairs) to be 1
__host__
void RadianceVolume::initialise_visits(){
    // this->visits = new float[ GRID_RESOLUTION * GRID_RESOLUTION ];
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            this->visits[ x*GRID_RESOLUTION + y ] = 0.f;
        }
    }
}

// Gets the incoming radiance values from all grid samples and
// populates radiance_grid with the estimates
// NOTE: This should be called before any radiance_volumes are instantiated
// in the scene by surfaces or these surfaces will be taken into account
__device__
void RadianceVolume::get_radiance_estimate_per_sector(curandState* volume_rand_state, Surface* surfaces, AreaLight* light_planes){
    // Path trace a ray in the direction from the centre of hemisphere towards
    // the centre of each sector to determine the radiance incoming from that direction
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            // Path trace SAMPLES_PER_PIXEL rays and average their value
            vec3 radiance = vec3(0);
            for (int i = 0; i < SAMPLES_PER_PIXEL; i++){
                // Get the 3D coordinates of a random point in the sector
                float r1 = curand_uniform(&volume_rand_state[x*SCREEN_HEIGHT + y]);
                float r2 = curand_uniform(&volume_rand_state[x*SCREEN_HEIGHT + y]);
                float sector_x, sector_y, sector_z;
                map(
                    (x + r1)/(float)GRID_RESOLUTION,
                    (y + r2)/(float)GRID_RESOLUTION,
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
                // radiance += path_trace_iterative(ray, surfaces, light_planes, 0);
                radiance = vec3(0.5f); // TODO: Fix once this function is implmented on a GPU
            }
            this->radiance_grid[ x*GRID_RESOLUTION + y ] = radiance / (float)SAMPLES_PER_PIXEL;
        }
    }
}

// Builds a radiance volume out of Surfaces, where each surfaces colour
// represents the incoming radiance at that position from that angle
// __device__
// void RadianceVolume::build_radiance_volume_shapes(std::vector<Surface>& surfaces){
//     // Get its vertices
//     vec4* vertices = this->get_vertices();
//     // Build Surfaces using the vertices
//     for (int x = 0; x < GRID_RESOLUTION; x++){
//         for (int y = 0; y < GRID_RESOLUTION; y++){
//             vec4 v1 = vertices[ x*GRID_RESOLUTION + y ];
//             vec4 v2 = vertices[ (x+1)*GRID_RESOLUTION + y ];
//             vec4 v3 = vertices[ x*GRID_RESOLUTION + (y+1) ];
//             vec4 v4 = vertices[ (x+1)*GRID_RESOLUTION + (y+1) ];
//             float r1 = ((float) rand() / (RAND_MAX));
//             float r2 = ((float) rand() / (RAND_MAX));
//             float r3 = ((float) rand() / (RAND_MAX));
//             surfaces.push_back(Surface(v1, v3, v2, Material(this->radiance_grid[ x*GRID_RESOLUTION + y ])));
//             surfaces.push_back(Surface(v2, v3, v4, Material(this->radiance_grid[ x*GRID_RESOLUTION + y ])));
//         }
//     }
//     delete [] vertices;
// }

// Builds a radiance volume out of Surfaces, where each surfaces colour
// represents the incoming radiance at that position from that angle
// __device__
// void RadianceVolume::build_radiance_magnitude_volume_shapes(std::vector<Surface>& surfaces){
//     // Get its vertices
//     vec4* vertices = this->get_vertices();
//     // Find the max radiance magnitude of the hemisphere
//     float max_radiance = 0.0001f;
//     for (int x = 0; x < GRID_RESOLUTION; x++){
//         for (int y = 0; y < GRID_RESOLUTION; y++){
//             if(length(this->radiance_grid[ x*GRID_RESOLUTION + y ]) > max_radiance){
//                 max_radiance = length(this->radiance_grid[ x*GRID_RESOLUTION + y ]);
//             }
//         }
//     }
//     // Build Surfaces using the vertices
//     for (int x = 0; x < GRID_RESOLUTION; x++){
//         for (int y = 0; y < GRID_RESOLUTION; y++){
//             vec4 v1 = vertices[ x*GRID_RESOLUTION + y ];
//             vec4 v2 = vertices[ (x+1)*GRID_RESOLUTION + y ];
//             vec4 v3 = vertices[ x*GRID_RESOLUTION + (y+1) ];
//             vec4 v4 = vertices[ (x+1)*GRID_RESOLUTION + (y+1) ];
//             float r1 = ((float) rand() / (RAND_MAX));
//             float r2 = ((float) rand() / (RAND_MAX));
//             float r3 = ((float) rand() / (RAND_MAX));
//             vec3 colour = vec3(1.f) - length(this->radiance_grid[ x*GRID_RESOLUTION + y ])/max_radiance * vec3(1.f);
//             surfaces.push_back(Surface(v1, v3, v2, Material(colour)));
//             surfaces.push_back(Surface(v2, v3, v4, Material(colour)));
//         }
//     }
//     delete [] vertices;
// }

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
vec3 RadianceVolume::get_irradiance(const Intersection& intersection, Surface* surfaces){
    vec3 irradiance = vec3(0);
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
            // Get the angle between the dir std::vector and the normal
            float cos_theta = dot(dir, this->normal); // No need to divide by lengths as they have been normalized
            irradiance += cos_theta * this->radiance_grid[ x*GRID_RESOLUTION + y ];
        }
    }
    irradiance /= ((float)(GRID_RESOLUTION * GRID_RESOLUTION)) * (1.f / (2.f * M_PI));
    irradiance *= surfaces[intersection.index].material.diffuse_c;
    return irradiance;
}

// Normalizes this RadianceVolume so that all radiance values 
// i.e. their grid values all sum to 1 (taking the length of each vec3)
__device__
void RadianceVolume::update_radiance_distribution(){
    // Get the total radiance from all directions (as a float)
    vec3 total_rgb = vec3(0.f);
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            total_rgb += this->radiance_grid[ x*GRID_RESOLUTION + y ];
        }
    }
    // Use this total to convert all radiance_grid values into probabilities
    // and store in the radiance_distribution
    float total = length(total_rgb);
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            this->radiance_distribution[ x*GRID_RESOLUTION + y ] = length(this->radiance_grid[ x*GRID_RESOLUTION + y ])/total;
        }
    }
}

// Samples a direction from the radiance volume
// volume
__device__
vec4 RadianceVolume::sample_direction_from_radiance_distribution(curandState* volume_rand_state, int& sector_x, int& sector_y){
    
    // Generate a random float uniformly 
    float r = curand_uniform(&volume_rand_state[sector_x*SCREEN_HEIGHT + sector_y]);

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
}

// Performs a temporal difference update for the current radiance volume for the incident
// radiance in the sector specified with the intersection surfaces irradiance value
__device__
void RadianceVolume::temporal_difference_update(vec3 next_irradiance, int sector_x, int sector_y){
    // Calculate alpha and update the radiance grid values and increment the number of visits
    float alpha = 1.f / (1.f + this->visits[ sector_x*GRID_RESOLUTION + sector_y ]);
    vec3 update = ((1.f - (alpha)) * this->radiance_grid[ sector_x*GRID_RESOLUTION + sector_y ]) + (alpha * next_irradiance);
    this->radiance_grid[ sector_x*GRID_RESOLUTION + sector_y ] = (length(update) > length(vec3(RADIANCE_THRESHOLD)) ? update : vec3(RADIANCE_THRESHOLD));
    this->visits[ sector_x*GRID_RESOLUTION + sector_y ] += 1;
}

// Sets a voronoi colour for the radiance volume (random colour) in the first entry of its radiance grid
__host__
void RadianceVolume::set_voronoi_colour(){
    float r = ((float) rand() / (RAND_MAX));
    float g = ((float) rand() / (RAND_MAX));
    float b = ((float) rand() / (RAND_MAX));
    this->radiance_grid[0] = vec3(r,g,b);
}

// Gets the voronoi colour of the radiance volume
__device__
vec3 RadianceVolume::get_voronoi_colour(){
    return this->radiance_grid[0];
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
