#ifndef NEURAL_Q_PATHTRACER
#define NEURAL_Q_PATHTRACER

#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

#include "image_settings.h"
#include "monte_carlo_settings.h"
#include "dq_network.cuh"
#include "sdl_screen.h"
#include "scene.cuh"
#include "hemisphere_helpers.cuh"
#include "radiance_volumes_settings.h"

#include <glm/glm.hpp>

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

// Trace a ray for all ray locations given in the angles specified within the scene
__global__
void trace_ray(
        vec3* ray_locations, 
        vec3* ray_normals, 
        int* ray_directions,
        float* discount_factors, 
        bool* ray_terminated, 
        float* ray_rewards, 
        Scene* scene
    );

// Gets the initial direction to shoot a ray in
__global__
void get_initial_ray_dir(curandState* d_rand_state, Camera& camera, int x, int y, vec3* ray_directions, vec3* ray_locations);

class NeuralQPathtracer{

    public:

        // Attributes
        unsigned int ray_batch_size;
        unsigned int num_batches;
        DQNetwork dqn;

        // Constructor
        __host__
        NeuralQPathtracer(unsigned int frames, unsigned int batch_size, SDLScreen& screen, Scene* scene);
        
        // Render a frame to output
        __host__
        void render_frame(dynet::ComputationGraph& graph, vec3* host_buffer);
};

#endif