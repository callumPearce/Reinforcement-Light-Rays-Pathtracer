#ifndef NN_RENDERING_HELPERS
#define NN_RENDERING_HELPERS

/* Dynet */
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

/* Project */
#include "image_settings.h"
#include "monte_carlo_settings.h"
#include "dq_network.cuh"
#include "sdl_screen.h"
#include "scene.cuh"
#include "hemisphere_helpers.cuh"
#include "radiance_volumes_settings.h"
#include "camera.cuh"
#include <sys/stat.h>
#include <unistd.h>
#include "deep_learning_settings.h"
#include <chrono>
#include <stdint.h>

/* Cuda */
#include "cuda_helpers.cuh"
#include <curand_kernel.h>
#include <curand.h>

/* GLM */
#include <glm/glm.hpp>
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

// Randomly sample with the given grid index a 3D ray direction
__device__
void sample_ray_for_grid_index(
    curandState* d_rand_state,
    int grid_idx,
    float* ray_directions_device,
    float* ray_normals_device,
    float* ray_locations_device,
    float* ray_throughputs_device,
    unsigned int* ray_states_device,
    int i
);

// Randomly sample a ray within the given grid idx and return as vec3
__device__
vec3 sample_ray_for_grid_index(
    curandState* d_rand_state,
    int grid_idx,
    float* ray_normals_device,
    float* ray_locations_device,
    int i
);

// Importance sample a ray direction over distribution proportional to cos_theta * Q_values
__global__
void sample_batch_ray_directions_importance_sample(
    curandState* d_rand_state,
    float* q_values_device,
    float* ray_directions_device,
    float* ray_locations_device,
    float* ray_normals_device,
    float* ray_throughputs_device,
    unsigned int* ray_states_device,
    unsigned int* ray_direction_indices,
    int batch_start_idx
);

// Sample random directions to further trace the rays in
__global__
void sample_next_ray_directions_randomly(
        curandState* d_rand_state,
        float* ray_normals, 
        float* ray_directions,
        float* ray_throughputs,
        unsigned int* ray_states
);

// Importance sample a ray direction for the current batch elem
__device__
void importance_sample_direction(
    curandState* d_rand_state,
    unsigned int* ray_direction_indices,
    float* current_qs_device,
    float* ray_directions,
    float* ray_locations,
    float* ray_normals,
    float* ray_throughputs,
    unsigned int* ray_states,
    int batch_start_idx,
    int batch_elem
);

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
    unsigned int* ray_states,
    int batch_start_idx
);


// Update pixel values stored in the device_buffer
__global__
void update_total_throughput(
        float* ray_throughputs,
        vec3* total_throughputs
);


// Update the device_buffer with the throughput
__global__
void update_device_buffer(
    vec3* device_buffer,
    vec3* total_throughputs
);


// Sum up all path lengths
__global__
void sum_path_lengths(
    int* total_path_lengths_device,
    unsigned int* ray_bounces
);

// Sample a random position on the scenes geometry and update the normal
__global__
void sample_random_scene_pos_for_terminated_rays(
    Scene* scene,
    curandState* d_rand_state,
    float* ray_normals,
    float* ray_locations,
    unsigned int* ray_states
);

// Get all vertices in corrdinate system for the current point
__global__
void convert_vertices_to_point_coord_system(
    float* ray_vertices_device, 
    float* ray_locations_device,
    float* scene_vertices,
    int vertices_count
);

// Check if a given file exists
inline bool file_exists (const std::string& name);

// Read the scene data file and populate the list of vertices
void load_scene_data(Scene& scene, std::vector<float>& scene_data);

// Sample index directions according the neural network q vals
__global__
void sample_batch_ray_directions_epsilon_greedy(
    float eta,
    curandState* d_rand_state,
    unsigned int* ray_direction_indices,
    float* current_qs_device,
    float* ray_directions,
    float* ray_locations,
    float* ray_normals,
    float* ray_throughputs,
    unsigned int* ray_states,
    int batch_start_idx
);

__global__
void sum_zero_contribution_light_paths(
    int* total_zero_contribution_light_paths,
    float* ray_throughputs
);


__device__
void sample_random_scene_pos(
    Scene* scene,
    curandState* d_rand_state,
    float* ray_normals,
    float* ray_locations,
    int i
);

#endif