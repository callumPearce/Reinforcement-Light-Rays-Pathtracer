#include "nn_rendering_helpers.cuh"


// Randomly sample with the given grid index a 3D ray direction
__device__
void sample_ray_for_grid_index(
    curandState* d_rand_state,
    int grid_idx,
    float* ray_directions_device,
    float* ray_normals_device,
    float* ray_locations_device,
    float* ray_throughputs_device,
    int i
){
    // Convert the index to a grid position
    int dir_x = int(grid_idx/GRID_RESOLUTION);
    int dir_y = grid_idx - (dir_x*GRID_RESOLUTION);

    // Convert to 3D direction and update the direction
    vec3 position = vec3(ray_locations_device[(i*3)], ray_locations_device[(i*3) + 1], ray_locations_device[(i*3) + 2]);
    vec3 normal  = vec3(ray_normals_device[(i*3)], ray_normals_device[(i*3) + 1], ray_normals_device[(i*3) + 2]);
    mat4 transformation_matrix = create_transformation_matrix(normal, vec4(position, 1.f));
    vec3 dir = convert_grid_pos_to_direction_random(d_rand_state, (float) dir_x, (float) dir_y, i, position, transformation_matrix);
    ray_directions_device[(i*3)    ] = dir.x;
    ray_directions_device[(i*3) + 1] = dir.y;
    ray_directions_device[(i*3) + 2] = dir.z;

    // Update throughput with new sampled angle
    float cos_theta = dot(normal, dir);
    ray_throughputs_device[(i*3)    ] = (ray_throughputs_device[(i*3)    ] * cos_theta)/RHO;
    ray_throughputs_device[(i*3) + 1] = (ray_throughputs_device[(i*3) + 1] * cos_theta)/RHO;
    ray_throughputs_device[(i*3) + 2] = (ray_throughputs_device[(i*3) + 2] * cos_theta)/RHO;
}

// Randomly sample a ray within the given grid idx and return as vec3
__device__
vec3 sample_ray_for_grid_index(
    curandState* d_rand_state,
    int grid_idx,
    float* ray_normals_device,
    float* ray_locations_device,
    int i
){
    // Convert the index to a grid position
    int dir_x = int(grid_idx/GRID_RESOLUTION);
    int dir_y = grid_idx - (dir_x*GRID_RESOLUTION);

    // Convert to 3D direction and update the direction
    vec3 position = vec3(ray_locations_device[(i*3)], ray_locations_device[(i*3) + 1], ray_locations_device[(i*3) + 2]);
    vec3 normal  = vec3(ray_normals_device[(i*3)], ray_normals_device[(i*3) + 1], ray_normals_device[(i*3) + 2]);
    mat4 transformation_matrix = create_transformation_matrix(normal, vec4(position, 1.f));
    return convert_grid_pos_to_direction_random(d_rand_state, (float) dir_x, (float) dir_y, i, position, transformation_matrix);
}

// Sample random directions to further trace the rays in
__global__
void sample_next_ray_directions_randomly(
        curandState* d_rand_state,
        float* ray_normals, 
        float* ray_directions,
        float* ray_throughputs,
        bool* ray_terminated
){
    
    // Ray index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;

    // Do nothing if we have already intersected with the light
    if (ray_terminated[i] == true){
        return;
    }

    // Sample the new direction and record it along with cos_theta
    float cos_theta;
    vec3 dir = vec3(sample_random_direction_around_intersection(d_rand_state, vec3(ray_normals[(i*3)], ray_normals[(i*3)+1], ray_normals[(i*3)+2]), cos_theta));
    ray_directions[(i*3)    ] = dir.x;
    ray_directions[(i*3) + 1] = dir.y;
    ray_directions[(i*3) + 2] = dir.z;

    // Update throughput with new sampled angle
    ray_throughputs[(i*3)    ] = (ray_throughputs[(i*3)    ] * cos_theta)/RHO;
    ray_throughputs[(i*3) + 1] = (ray_throughputs[(i*3) + 1] * cos_theta)/RHO;
    ray_throughputs[(i*3) + 2] = (ray_throughputs[(i*3) + 2] * cos_theta)/RHO;
}


// Compute the TD targets for the current batch size
__global__
void compute_td_targets(
    curandState* d_rand_state,
    float* next_qs_device,
    float* td_targets_device,
    float* ray_locations,
    float* ray_normals,
    float* ray_rewards,
    float* ray_discounts,
    int batch_start_idx
){
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (batch_start_idx + batch_idx >= SCREEN_HEIGHT*SCREEN_WIDTH) return;

    // Get the max q_val
    unsigned int max_idx = 0;
    float max_q_val = next_qs_device[batch_idx];
    for (unsigned int i = 1; i < GRID_RESOLUTION*GRID_RESOLUTION; i++){
        float temp_q = next_qs_device[batch_idx + i];
        if (max_q_val < temp_q){
            max_q_val = temp_q;
            max_idx = i;
        }
    }

    // Calculate cos_theta
    vec3 dir = sample_ray_for_grid_index(
        d_rand_state,
        max_idx,
        ray_normals,
        ray_locations,
        (batch_idx+batch_start_idx)
    );
    vec3 normal(ray_normals[(batch_start_idx+batch_idx)*3], ray_normals[(batch_start_idx+batch_idx)*3 + 1], ray_normals[(batch_start_idx+batch_idx)*3 + 2]);
    float cos_theta = dot(normal, dir);

    //TODO: Fix this, cos_theta multiplication causing program crash
    td_targets_device[ batch_idx ] =  ray_rewards[ batch_idx + batch_start_idx ] + max_q_val*ray_discounts[ batch_idx + batch_start_idx ];//*cos_theta;
}   

// Update pixel values stored in the device_buffer
__global__
void update_total_throughput(
        float* ray_throughputs,
        vec3* total_throughputs
){

    // Ray index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;
    
    // Accumulate
    total_throughputs[i] += vec3(ray_throughputs[(i*3)], ray_throughputs[(i*3)+1], ray_throughputs[(i*3)+2]);
}

// Update the device_buffer with the throughput
__global__
void update_device_buffer(
    vec3* device_buffer,
    vec3* total_throughputs
){
    
    // Ray index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;

    // Update
    device_buffer[i] = total_throughputs[i]/(float)SAMPLES_PER_PIXEL;
}

// Sum up all path lengths
__global__
void sum_path_lengths(
    int* total_path_lengths_device,
    unsigned int* ray_bounces
){
    // Ray index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;

    atomicAdd(total_path_lengths_device, (int)ray_bounces[i]); 
}

inline bool file_exists (const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}


// Read the scene data file and populate the list of vertices
void load_scene_data(Scene& scene, std::vector<float>& scene_data){

    for (int i = 0; i < scene.surfaces_count; i++){
        
        Surface sf = scene.surfaces[i];

        // Normals
        scene_data.push_back(sf.normal.x);
        scene_data.push_back(sf.normal.y);
        scene_data.push_back(sf.normal.z);

        // Vertices
        scene_data.push_back(sf.v0.x);
        scene_data.push_back(sf.v0.y);
        scene_data.push_back(sf.v0.z);
        scene_data.push_back(sf.v1.x);
        scene_data.push_back(sf.v1.y);
        scene_data.push_back(sf.v1.z);
        scene_data.push_back(sf.v2.x);
        scene_data.push_back(sf.v2.y);
        scene_data.push_back(sf.v2.z);
    }

    for (int i = 0; i < scene.area_light_count; i++){

        AreaLight al = scene.area_lights[i];

        // Normals
        scene_data.push_back(al.normal.x);
        scene_data.push_back(al.normal.y);
        scene_data.push_back(al.normal.z);

        // Vertices
        scene_data.push_back(al.v0.x);
        scene_data.push_back(al.v0.y);
        scene_data.push_back(al.v0.z);
        scene_data.push_back(al.v1.x);
        scene_data.push_back(al.v1.y);
        scene_data.push_back(al.v1.z);
        scene_data.push_back(al.v2.x);
        scene_data.push_back(al.v2.y);
        scene_data.push_back(al.v2.z);
    }
}