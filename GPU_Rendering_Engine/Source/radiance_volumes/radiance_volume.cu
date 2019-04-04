#include "radiance_volume.cuh"
#include <iostream>

__device__ __host__
RadianceVolume::RadianceVolume(){
    this->position = vec4(0.f);
    this->normal = vec3(0.f);
    this->transformation_matrix = mat4(0.f);
}

__host__
RadianceVolume::RadianceVolume(Surface* surfaces, vec4 position, vec4 normal, unsigned int surface_index, int idx){
    this->surface_index = surface_index;
    initialise_radiance_grid(surfaces);
    initialise_radiance_distribution();
    initialise_visits();

    this->position = position;
    this->normal = vec3(normal.x, normal.y, normal.z);

    // Create the transformation matrix for this hemisphere: local->world
    this->transformation_matrix = create_transformation_matrix(normal, position);
    this->index = idx;
}

// Updates the transformation matrix with the current set values of the normal and position
__host__
void RadianceVolume::update_transformation_matrix(){
    this->transformation_matrix = create_transformation_matrix(normal, position);
} 

// Intialises a 2D grid to store radiance values at each grid point and
// sets the irradiance estimate
__host__
void RadianceVolume::initialise_radiance_grid(Surface* surfaces){
    // Set values in the radiance grid
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            this->radiance_grid[ x*GRID_RESOLUTION + y ] = INITIAL_RADIANCE;
        }
    }
    // Compute the current irradiance estimate
    float temp_irradiance = 0.f;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            vec3 dir = convert_grid_pos_to_direction((float)x,(float)y, vec3(this->position), this->transformation_matrix);
            // Get the angle between the dir std::vector and the normal
            float cos_theta = dot(dir, this->normal); // No need to divide by lengths as they have been normalized
            temp_irradiance += cos_theta * this->radiance_grid[ x*GRID_RESOLUTION + y ];
        }
    }
    float luminance = surfaces[this->surface_index].material.luminance;
    temp_irradiance *= luminance / (float)M_PI;
    this->irradiance_accum = temp_irradiance;
}

// Initialise radiance distribution to be equal in all angles initially
__host__
void RadianceVolume::initialise_radiance_distribution(){
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            this->radiance_distribution[ x*GRID_RESOLUTION + y ] = (x*GRID_RESOLUTION + y) * (1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION));        
        }
    }
}

// Initialises the alpha values (weighting of state-action pairs) to be 1
__host__
void RadianceVolume::initialise_visits(){
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
// radiance from all directions whilst multiplying by BRDF and cos(theta)) (Following expected SARSA)
__device__
void RadianceVolume::expected_sarsa_irradiance(Surface* surfaces, const float update, const int sector_x, const int sector_y){

    // Get the direction
    vec3 dir = convert_grid_pos_to_direction((float)sector_x, (float)sector_y, vec3(this->position), this->transformation_matrix);
    // Get the angle between the dir std::vector and the normal
    float cos_theta = dot(dir, this->normal); // No need to divide by lengths as they have been normalized

    // Get the BRDF
    float luminance = surfaces[this->surface_index].material.luminance;
    float BRDF = luminance / (float)M_PI;
    
    // Update the irradiance and the radiance_grid value
    int sector_location = sector_x*GRID_RESOLUTION + sector_y;
    float old_sector = this->radiance_grid[ sector_location ];
    float new_irradiance = (this->irradiance_accum - (old_sector*cos_theta*BRDF)) + (update*cos_theta*BRDF);

    atomicExch(&(this->radiance_grid[ sector_location ]), update);
    atomicExch(&(this->irradiance_accum), new_irradiance);
}


// // Gets the irradiance for an intersection point by getting the max directional sector values
// // multiplied by the BRDF and cos_theta
// __device__
// float RadianceVolume::q_learning_irradiance(Surface* surfaces){
//     float max_irradiance = 0.f;
//     for (int x = 0; x < GRID_RESOLUTION; x++){
//         for (int y = 0; y < GRID_RESOLUTION; y++){
//             // Get the coordinates on the unit hemisphere
//             float x_h, y_h, z_h;
//             map(x/(float)GRID_RESOLUTION, y/(float)GRID_RESOLUTION, x_h, y_h, z_h);
//             // Convert to world space
//             vec4 world_position = this->transformation_matrix * vec4(x_h, y_h, z_h, 1.f);
//             vec3 world_position3 = vec3(world_position.x, world_position.y, world_position.z);
//             // Get the direction
//             vec3 dir = normalize(world_position3 - vec3(this->position));
//             // Get the angle between the dir std::vector and the normal
//             float cos_theta = dot(dir, this->normal); // No need to divide by lengths as they have been normalized
//             max_irradiance = max(cos_theta * this->radiance_grid[ x*GRID_RESOLUTION + y ], max_irradiance);
//         }
//     }
//     vec3 BRDF_3 = surfaces[this->surface_index].material.diffuse_c;
//     max_irradiance *= ((BRDF_3.x + BRDF_3.y + BRDF_3.z)/3.f) / (float)M_PI;
//     return max_irradiance;
// }

// Updates the current irradiance estimate
__device__
void RadianceVolume::update_irradiance(Surface* surfaces, const float update, const int sector_x, const int sector_y){
    this->expected_sarsa_irradiance(surfaces, update, sector_x, sector_y);
}

// Normalizes this RadianceVolume so that all radiance values 
// i.e. their grid values all sum to 1 (taking the length of each vec3)
__device__
void RadianceVolume::update_radiance_distribution(){

    // Get the total radiance from all directions (as a float)
    float total = 0.0000000001f;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            total += this->radiance_grid[ x*GRID_RESOLUTION + y ]; 
        }
    }
    // Use this total to convert all radiance_grid values into probabilities
    // and store in the radiance_distribution
    float prev_radiance = 0.f;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            float radiance = this->radiance_grid[ x*GRID_RESOLUTION + y ]/total + prev_radiance;
            this->radiance_distribution[ x*GRID_RESOLUTION + y ] = radiance;
            prev_radiance = radiance;
        }
    }
}

// Samples a direction from the radiance volume using binary search for the sector
__device__
vec4 RadianceVolume::sample_direction_from_radiance_distribution(curandState* d_rand_state, int pixel_x, int pixel_y, int& sector_x, int& sector_y){
    
    // Generate a random float uniformly 
    float r = curand_uniform(&d_rand_state[pixel_x*SCREEN_HEIGHT + pixel_y]);

    // Check if in first sector location
    if (r <= this->radiance_distribution[ 0 ]){
        // Get sector 0
        sector_x = 0;
        sector_y = 0;
        // Randomly sample within the sector
        float rx = curand_uniform(&d_rand_state[pixel_x*SCREEN_HEIGHT + pixel_y]);
        float ry = curand_uniform(&d_rand_state[pixel_x*SCREEN_HEIGHT + pixel_y]);
        return vec4(convert_grid_pos_to_direction(sector_x+rx, sector_y+ry, vec3(this->position), this->transformation_matrix), 1.f);
    }

    // Binary Search for the sector to uniformly sample from
    int start = 0;
    int end = GRID_RESOLUTION*GRID_RESOLUTION - 1;

    while(start <= end){

        // Compute the mid index
        int mid = ((end + start)/2);

        // Check if found
        float mid_val = this->radiance_distribution[ mid ];
        float prev_mid_val = this->radiance_distribution[ mid - 1 ];
        if (r < mid_val && prev_mid_val <= r){
            // Found the sector at location mid
            sector_x = (int)mid/GRID_RESOLUTION;
            sector_y = mid - (sector_x*GRID_RESOLUTION);
            // Randomly sample within the sector
            float rx = curand_uniform(&d_rand_state[pixel_x*SCREEN_HEIGHT + pixel_y]);
            float ry = curand_uniform(&d_rand_state[pixel_x*SCREEN_HEIGHT + pixel_y]);
            return vec4(convert_grid_pos_to_direction(sector_x+rx, sector_y+ry, vec3(this->position), this->transformation_matrix), 1.f);
        }

        // Check if look right
        else if (mid_val < r){
            start = mid + 1;
        }

        // Look left
        else{
            end = mid - 1;
        }
    }
    return vec4(0.f,0.f,0.f,1.f);
}

// Performs a temporal difference update for the current radiance volume for the incident
// radiance in the sector specified with the intersection surfaces irradiance value
__device__
void RadianceVolume::temporal_difference_update(float sector_irradiance, int sector_x, int sector_y, Surface* surfaces){

    int sector_location = sector_x*GRID_RESOLUTION + sector_y;

    // Calculate alpha and update the radiance grid values and increment the number of visits
    unsigned int vs = this->visits[ sector_location ];
    float alpha = 1.f / (1.f + (float)vs);

    // Calculate the new update value
    float radiance = this->radiance_grid[ sector_location ];
    float update = ((1.f - (alpha)) * radiance) + (alpha * sector_irradiance);
    // update = update > (float)RADIANCE_THRESHOLD ? update : (float)RADIANCE_THRESHOLD;

    // Update the radiance grid value and the alpha value
    atomicInc(&(this->visits[ sector_location ]), vs+1);

    // Update the irradiance estimate
    update_irradiance(surfaces, update, sector_x, sector_y);
}

// Gets the current irradiance estimate for the radiance volume
__device__
float RadianceVolume::get_irradiance_estimate(){
    return this->irradiance_accum * (2.f * (float)M_PI) / ((float)(GRID_RESOLUTION * GRID_RESOLUTION));
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

// Conver the radiance volumes cumulative distribution to a distribution
__host__
void RadianceVolume::convert_radiance_distribution(){
    for (unsigned int i = 0; i < GRID_RESOLUTION*GRID_RESOLUTION; i++){
        this->radiance_distribution[ GRID_RESOLUTION*GRID_RESOLUTION - i] -= this->radiance_distribution[ GRID_RESOLUTION*GRID_RESOLUTION - (i + 1) ];
    }
}