#ifndef PRE_TRAINED_PATHTRACER
#define PRE_TRAINED_PATHTRACER

/* std */
#include <string.h>
#include <sys/stat.h>

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

// Efficiently checks if a file exists
inline bool file_exists (const std::string& name);

// Gets the initial direction to shoot a ray in
__global__
void initialise_ray(
        curandState* d_rand_state,
        Camera* device_camera, 
        float* ray_locations, 
        float* ray_directions,
        bool* ray_terminated, 
        float* ray_throughputs,
        unsigned int* ray_bounces
    );

// Trace a ray for all ray locations given in the angles specified within the scene
__global__
void trace_ray(
    Scene* scene,
    int* rays_finished,
    float* ray_locations, 
    float* ray_normals, 
    float* ray_directions,
    bool* ray_terminated,  
    float* ray_throughputs,
    unsigned int* ray_bounces,
    int bounces
);

// Importance samples rays directions from Q-values
__global__
void importance_sample_ray_directions(
    curandState* d_rand_state,
    float* device_q_values,
    float* ray_normals_device,
    float* ray_directions_device,
    float* ray_locations_device,
    float* ray_throughputs_device,
    bool* ray_terminated_device,
    int batch_start_idx
);


class PretrainedPathtracer{

    public:

        // Attributes
        dim3 num_blocks;
        dim3 block_size;
        int batch_size;
        int num_batches;
        DQNetwork dqn;
    
        // Constructor
        __host__
        PretrainedPathtracer(
            unsigned int frames,
            int batch_size, 
            SDLScreen& screen, 
            Scene& scene,
            Camera& camera,
            int argc,
            char** argv
        );

        // Render a frame to output
        __host__
        void render_frame(
            curandState* d_rand_state,
            Camera* device_camera,
            Scene* device_scene,
            vec3* device_buffer, 
            float* ray_locations_device,
            float* ray_normals_device,   
            float* ray_directions_device,
            bool* ray_terminated_device,  
            float* ray_throughputs_device,
            unsigned int* ray_bounces_device
        );
};

#endif