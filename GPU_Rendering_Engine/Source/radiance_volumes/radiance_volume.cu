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

// Constructor for loading radiance volume to render
__host__
RadianceVolume::RadianceVolume(vec4 position, vec3 normal, std::vector<float>& q_vals){
    this->position = position;
    this->normal = normal;

    // Create the transformation matrix for this hemisphere: local->world
    this->transformation_matrix = create_transformation_matrix(normal, position);

    // Set the q_vals read in
    this->set_q_vals(q_vals);
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
    float luminance = surfaces[this->surface_index].material.luminance;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            vec3 dir = convert_grid_pos_to_direction((float)x+0.5f,(float)y+0.5f, vec3(this->position), this->transformation_matrix);
            // Get the angle between the dir std::vector and the normal
            float cos_theta = dot(dir, this->normal); // No need to divide by lengths as they have been normalized
            
            temp_irradiance += cos_theta * (luminance / M_PI) * this->radiance_grid[ x*GRID_RESOLUTION + y ];
        }
    }
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
            // Get the direction
            vec3 dir = convert_grid_pos_to_direction((float)x+0.5f, (float)y+0.5f, vec3(this->position), this->transformation_matrix);
            // Get the angle between the dir std::vector and the normal
            float cos_theta = dot(dir, this->normal);

            float temp = this->radiance_grid[ x*GRID_RESOLUTION + y ]*cos_theta;
            
            temp = temp > DISTRIBUTION_THRESHOLD ? temp : DISTRIBUTION_THRESHOLD;
            
            total += temp;
        }
    }
    // Use this total to convert all radiance_grid values into probabilities
    // and store in the radiance_distribution
    float prev_radiance = 0.f;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            // Get the direction
            vec3 dir = convert_grid_pos_to_direction((float)x+0.5f, (float)y+0.5f, vec3(this->position), this->transformation_matrix);
            // Get the angle between the dir std::vector and the normal
            float cos_theta = dot(dir, this->normal);

            float temp = this->radiance_grid[ x*GRID_RESOLUTION + y ]*cos_theta;
            
            temp = temp > DISTRIBUTION_THRESHOLD ? temp : DISTRIBUTION_THRESHOLD;

            float radiance = (temp)/total + prev_radiance;
            this->radiance_distribution[ x*GRID_RESOLUTION + y ] = radiance;
            prev_radiance = radiance;
        }
    }
    if (total < 0.f)
        printf("%.3f\n",total);
}

// Samples a direction from the radiance volume using binary search for the sector
__device__
vec4 RadianceVolume::sample_direction_from_radiance_distribution(curandState* d_rand_state, int pixel_x, int pixel_y, int& sector_x, int& sector_y, float& pdf){
    
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
        pdf = RHO * (this->radiance_distribution[ 0 ] / GRID_RHO);
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

            pdf = RHO * ((mid_val-prev_mid_val) / GRID_RHO);
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

// Samples a direction from the radiance volume using binary search for the sector
__device__
vec4 RadianceVolume::sample_max_direction_from_radiance_distribution(curandState* d_rand_state, int pixel_x, int pixel_y, int& sector_x, int& sector_y){
    
    // Find max val in radiance grid
    int max_idx = 0;
    float max_irradiance = this->radiance_grid[ 0 ];
    for (int i = 0; i < GRID_RESOLUTION*GRID_RESOLUTION; i++){
        if( max_irradiance < this->radiance_grid[i] ){
            max_irradiance = this->radiance_grid[i];
            max_idx = i;
        }
    }
    // Found the sector at location max
    sector_x = (int)max_idx/GRID_RESOLUTION;
    sector_y = max_idx - (sector_x*GRID_RESOLUTION);
    // Randomly sample within the sector
    float rx = curand_uniform(&d_rand_state[pixel_x*SCREEN_HEIGHT + pixel_y]);
    float ry = curand_uniform(&d_rand_state[pixel_x*SCREEN_HEIGHT + pixel_y]);
    return vec4(convert_grid_pos_to_direction(sector_x+rx, sector_y+ry, vec3(this->position), this->transformation_matrix), 1.f);
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
    update = update > (float)RADIANCE_THRESHOLD ? update : (float)RADIANCE_THRESHOLD;

    // Update the radiance grid value and the alpha value
    atomicInc(&(this->visits[ sector_location ]), vs+1);

    // Update the irradiance estimate
    update_irradiance(surfaces, update, sector_x, sector_y);
}

// Gets the current irradiance estimate for the radiance volume
__device__
float RadianceVolume::get_irradiance_estimate(){
    return this->irradiance_accum * ((2.f * (float)M_PI) / ((float)(GRID_RESOLUTION * GRID_RESOLUTION)));
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

// Write the radiance volumes Q-values out to a file
__host__
void RadianceVolume::write_volume_to_file(std::string filename){
    // Create the file 
    std::ofstream save_file (filename, std::ios::app);
    if (save_file.is_open()){
        // Write the position
        vec4 position = this->position;
        save_file << position.x << " " << position.y << " " << position.z;

        // Write the normal
        vec3 normal = this->normal;
        save_file << " " << normal.x << " " << normal.y << " " << normal.z;

        // Write each Q values
        for (int n = 0; n < GRID_RESOLUTION*GRID_RESOLUTION; n++){
            save_file << " " << this->radiance_distribution[n];
        }

        save_file << "\n";

        // Close the file
        save_file.close();
    }
    else{
        printf("Unable to save the Radiance Volume.\n");
    }
}

// Set the radiance distribution of the radiance volume to the
// supplied q_vals
void RadianceVolume::set_q_vals(std::vector<float>& q_vals){
    for (int n = 0; n < GRID_RESOLUTION*GRID_RESOLUTION; n++){
        this->radiance_distribution[n] = q_vals[n];
    }
}

// Read radiance Volumes from a file 
__host__
void RadianceVolume::read_radiance_volumes_from_file(
    std::string fname, 
    std::vector<RadianceVolume>& rvs
){
    
    // Read each line of the file individually and build a radiance
    // volume from the data
    std::string line;
    std::ifstream rvs_file(fname);
    
    if (rvs_file.is_open()){
        while ( std::getline (rvs_file, line)){

            // Data structures for radiance volume budiling
            vec4 position(1.f);
            vec3 normal(0.f);
            std::vector<float> q_vals(GRID_RESOLUTION*GRID_RESOLUTION);
            
            // For each space
            unsigned int idx = 0;
            size_t pos = 0;
            float data_elem;
            while ((pos = line.find(" ")) != std::string::npos){
                
                // Get the next string
                data_elem = std::stof(line.substr(0, pos));

                // 0 - 2: Position
                if (idx < 3){
                    position[idx] = data_elem;
                }

                // 3 - 5: Normal
                else if(idx < 6){
                    normal[idx%3] = data_elem;
                }

                // 6 - GRID_RES*GRID_RES+6: Radiance Distribution
                else{
                    q_vals[idx-6] = data_elem;
                }

                // Increment the index for the current data_elem
                idx++;

                // Delete the part that we have read
                line.erase(0, pos + 1);
            }

            // Add the final float in
            q_vals[idx-6] = std::stof(line);

            // Create the radiance volume and add it to the list
            RadianceVolume rv = RadianceVolume(position, normal, q_vals);
            rvs.push_back(rv);
        }
    }
    else{
        std::cout << "Could not read radiance volumes." << std::endl;
    }
}

// Returns a list of vertices for the generated radiance volume
__host__
std::vector<std::vector<vec4>> RadianceVolume::get_vertices(){
    std::vector<std::vector<vec4>> vertices;
    // For every grid coordinate, add the corresponding 3D world coordinate
    for (int x = 0; x <= GRID_RESOLUTION; x++){
        std::vector<vec4> vecs;
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
            vecs.push_back(world_position);
        }
        vertices.push_back(vecs);
    }
    return vertices;
}

// Build the radiance volumes surfaces and add it to the list based
// on the radiance distribution values
void RadianceVolume::build_surfaces(std::vector<Surface>& surfaces){
    // Find the max q_val to determine colour
    float max_q = 0.f;
    for (int n = 0; n < GRID_RESOLUTION*GRID_RESOLUTION; n++){
        if ( max_q < this->radiance_distribution[n] ) max_q = this->radiance_distribution[n];
    }

    // Get the vertices for the radiance volume
    std::vector<std::vector<vec4>> vertices = this->get_vertices();
    // Build the surfaces
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            // Get the square of vertices
            vec4 v0 = vertices[x][y];
            vec4 v1 = vertices[x+1][y];
            vec4 v2 = vertices[x][y+1];
            vec4 v3 = vertices[x+1][y+1];
            vec4 mid_p = (v0 + v1 + v2 + v3)/4.f;

            // Build two triangles using radiance_distribution for colour
            float ratio = this->radiance_distribution[x*GRID_RESOLUTION + y]/max_q;
            Surface s1 = Surface(v0, v2, v1, Material(vec3(ratio, 1.f-ratio, 0.f)));
            Surface s2 = Surface(v1, v2, v3, Material(vec3(ratio, 1.f-ratio, 0.f)));
            s1.normal = normalize(mid_p - this->position);
            s2.normal = s1.normal;
            // Add the surfaces to the list of surfaces
            surfaces.push_back(s1);
            surfaces.push_back(s2);
        }
    }
}

// Read the list of radiance volumes from a file and build surfaces of them
__host__
void RadianceVolume::read_radiance_volumes_to_surfaces(
    std::string fname,
    std::vector<Surface>& surfaces
){
    // Get the radiance volumes from the file
    std::vector<RadianceVolume> rvs;
    RadianceVolume::read_radiance_volumes_from_file(
        fname, 
        rvs
    );

    // For each RV, build its surfaces and add it to the list
    for (int i = 0; i < rvs.size(); i++){
        rvs[i].build_surfaces(surfaces);
    }
}