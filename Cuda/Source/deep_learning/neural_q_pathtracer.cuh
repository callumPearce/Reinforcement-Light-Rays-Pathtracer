#ifndef NEURAL_Q_PATHTRACER
#define NEURAL_Q_PATHTRACER

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


// Trace a ray for all ray locations given in the angles specified within the scene
__global__
static void trace_ray(
        Scene* scene,
        int* rays_finished,
        float* ray_locations, 
        float* ray_normals, 
        float* ray_directions,
        bool* ray_terminated, 
        float* ray_rewards, 
        float* ray_discounts,
        float* ray_throughputs,
        unsigned int* ray_bounces,
        int bounces
    );

// Gets the initial direction to shoot a ray in
__global__
static void initialise_ray(
        curandState* d_rand_state,
        Camera* device_camera, 
        float* ray_locations, 
        float* ray_directions,
        bool* ray_terminated, 
        float* ray_rewards, 
        float* ray_discounts,
        float* ray_throughputs,
        unsigned int* ray_bounces
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

// Sample ray directions according the neural network q vals
__global__
void sample_batch_ray_indices_eta_greedy(
        float eta,
        curandState* d_rand_state,
        unsigned int* directions_device,
        float* current_qs_device,
        int batch_index,
        int batch_size
    );

// Randomly sample with the given grid index a 3D ray direction
__global__
void sample_ray_for_grid_index(
    curandState* d_rand_state,
    unsigned int* grid_indices,
    float* ray_directions,
    float* ray_locations,
    float* ray_normals,
    float* ray_throughputs,
    bool* ray_terminated
);

// Compute the TD target from the passed in data for the current batch
__global__
void compute_td_targets(
        float* td_targets_device,
        float* ray_rewards,
        float* ray_discounts
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

class NeuralQPathtracer{

    public:

        // Attributes
        unsigned int ray_batch_size;
        unsigned int num_batches;
        dim3 num_blocks;
        dim3 block_size;
        DQNetwork dqn;

        // Constructor
        __host__
        NeuralQPathtracer(
            unsigned int frames, 
            unsigned int batch_size, 
            SDLScreen& screen, 
            Scene& scene,
            Camera& camera,
            int argc,
            char** argv
        );
        
        // Render a frame to output
        __host__
        void render_frame(
            dynet::AdamTrainer trainer,
            curandState* d_rand_state,
            Camera* device_camera,
            Scene* device_scene,
            vec3* device_buffer,
            float* prev_location_host,   
            unsigned int* directions_host,
            float* ray_locations,   /* Ray intersection location (State)*/
            float* ray_normals,     /* Intersection normal */
            float* ray_directions,  /* Direction to next shoot the ray*/
            bool* ray_terminated,  /* Has the ray intersected with a light/nothing*/
            float* ray_rewards,    /* Reward recieved from Q(s,a) */
            float* ray_discounts,  /* Discount factor for current rays path */
            float* ray_throughputs,  /* Throughput for calc pixel value*/
            unsigned int* ray_bounces  /* Total number of bounces for each ray before intersection*/
        );
};

#endif