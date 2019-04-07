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

/* Cuda */
#include <stdint.h>
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

// Sample random directions to further trace the rays in
__global__
void sample_next_ray_directions_randomly(
        curandState* d_rand_state,
        float* ray_normals, 
        float* ray_directions,
        float* ray_throughputs,
        bool* ray_terminated
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

inline bool file_exists (const std::string& name);

// Read the scene data file and populate the list of vertices
void load_scene_data(Scene& scene, std::vector<float>& scene_data);

#endif