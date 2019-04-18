#ifndef NEURAL_Q_PATHTRACER
#define NEURAL_Q_PATHTRACER

/* Dynet */
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

/* Project */
#include "nn_rendering_helpers.cuh"
#include "image_settings.h"
#include "monte_carlo_settings.h"
#include "dq_network.cuh"
#include "sdl_screen.h"
#include "scene.cuh"
#include "hemisphere_helpers.cuh"
#include "radiance_volumes_settings.h"
#include "camera.cuh"

/* File writing */
#include <iostream>
#include <fstream>

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
    curandState* d_rand_state,
    int* rays_finished,
    float* ray_locations, 
    float* prev_ray_locations,
    float* ray_normals, 
    float* ray_directions,
    unsigned int* ray_states, 
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
    float* prev_ray_locations,
    float* ray_directions,
    unsigned int* ray_states, 
    float* ray_rewards, 
    float* ray_discounts,
    float* ray_throughputs,
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
        int vertices_count;
        float epsilon;

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
            float* device_vertices,
            unsigned int* directions_host,
            float* ray_locations,   /* Ray intersection location (State) */
            float* prev_ray_locations,
            float* ray_normals,     /* Intersection normal */
            float* ray_directions,  /* Direction to next shoot the ray */
            unsigned int* ray_states,  /* Has the ray intersected with a light/nothing */
            float* ray_rewards,    /* Reward recieved from Q(s,a) */
            float* ray_discounts,  /* Discount factor for current rays path */
            float* ray_throughputs,  /* Throughput for calc pixel value */
            unsigned int* ray_bounces, /* Total number of bounces for each ray before intersection*/
            float* ray_vertices
        );
};

#endif