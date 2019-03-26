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
        vec3* ray_locations, 
        vec3* ray_normals, 
        vec3* ray_directions,
        bool* ray_terminated, 
        float* ray_rewards, 
        float* ray_discounts,
        vec3* ray_throughputs
    );

// Gets the initial direction to shoot a ray in
__global__
static void initialise_ray(
        curandState* d_rand_state,
        Camera* device_camera, 
        vec3* ray_locations, 
        vec3* ray_directions,
        bool* ray_terminated, 
        float* ray_rewards, 
        float* ray_discounts,
        vec3* ray_throughputs
    );

// Sample random directions to further trace the rays in
__global__
void sample_next_ray_directions_randomly(
        curandState* d_rand_state,
        vec3* ray_normals, 
        vec3* ray_directions,
        vec3* ray_throughputs,
        bool* ray_terminated
    );

// Update pixel values stored in the device_buffer
__global__
void update_total_throughput(
        vec3* ray_throughputs,
        vec3* total_throughputs
    );

// Update the device_buffer with the throughput
__global__
void update_device_buffer(
    vec3* device_buffer,
    vec3* total_throughputs
);

class NeuralQPathtracer{

    public:

        // Attributes
        unsigned int ray_batch_size;
        unsigned int num_batches;
        dim3 num_blocks;
        dim3 block_size;
        // DQNetwork dqn;

        // Constructor
        __host__
        NeuralQPathtracer(
            unsigned int frames, 
            unsigned int batch_size, 
            SDLScreen& screen, 
            Scene& scene,
            Camera& camera
        );
        
        // Render a frame to output
        __host__
        void render_frame(
            // dynet::ComputationGraph& graph,
            curandState* d_rand_state,
            Camera* device_camera,
            Scene* device_scene,
            vec3* device_buffer,
            vec3* ray_locations,   /* Ray intersection location (State)*/
            vec3* ray_normals,     /* Intersection normal */
            vec3* ray_directions,  /* Direction to next shoot the ray*/
            bool* ray_terminated,  /* Has the ray intersected with a light/nothing*/
            float* ray_rewards,    /* Reward recieved from Q(s,a) */
            float* ray_discounts,  /* Discount factor for current rays path */
            vec3* ray_throughputs  /* Throughput for calc pixel value*/
        );
};

#endif