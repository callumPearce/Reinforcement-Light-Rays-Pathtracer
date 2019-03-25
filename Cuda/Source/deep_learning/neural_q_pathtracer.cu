#include "neural_q_pathtracer.cuh"

__host__
NeuralQPathtracer::NeuralQPathtracer(unsigned int frames, unsigned int batch_size, SDLScreen& screen, Scene* scene){

    // Assign attributes
    this->ray_batch_size = batch_size;
    this->num_batches = int((SCREEN_HEIGHT * SCREEN_WIDTH)/batch_size) - 1;
    this->dqn = DQNetwork();

    // Create the pixel buffer on the host device
    vec3* host_buffer = new vec3[ SCREEN_HEIGHT * SCREEN_WIDTH ];

    // Initialise the computation graph
    dynet::ComputationGraph graph;

    for (int i = 0; i < frames; i++){
        //Clear the pixel buffer
        memset(host_buffer, 0.f, sizeof(vec3)* SCREEN_HEIGHT * SCREEN_WIDTH);
        
        // Fill the pixel buffer each frame using Deep Q-Learning strategy
        this->render_frame(graph, host_buffer);

        // Display the rendered frame
        for (int x = 0; x < SCREEN_WIDTH; x++){
            for (int y = 0; y < SCREEN_HEIGHT; y++){
                screen.PutPixelSDL(x, y, host_buffer[x*(int)SCREEN_HEIGHT + y]);
            }
        }
        screen.SDL_Renderframe();
    }

    // Save the image and kill the screen
    screen.SDL_SaveImage("../../Images/render.bmp");
    screen.kill_screen();

    // Delete the pixel buffer
    delete [] host_buffer;
}

__host__
void NeuralQPathtracer::render_frame(dynet::ComputationGraph& graph, vec3* host_buffer){

    // Initialise: 
    vec3* ray_locations = new vec3[this->ray_batch_size]; /* Ray intersection location (State)*/
    vec3* ray_normals = new vec3[this->ray_batch_size]; /* Intersection normal */
    vec3* ray_directions = new vec3[this->ray_batch_size]; /* Direction to next shoot the ray*/
    bool* ray_terminated = new bool[this->ray_batch_size]; /* Has the ray intersected with a light/nothing*/
    float* ray_rewards = new float[this->ray_batch_size]; /* Reward recieved from Q(s,a) */

    // For every batch of rays (TODO: we will be missing some pixels in our last batch, need to account for this)
    for (int n = 0; n < this->num_batches; n++){

        // Initialise mini-batch ray variables TODO
        memset(ray_terminated, false, sizeof(bool) * this->ray_batch_size);

        // Trace batch rays path until all have intersected with a light
        bool paths_ended = false;
        while(!paths_ended){

            // 1) Get Q_values for current state for every ray in the batch
            dynet::Dim input_dim({3}, this->ray_batch_size);
            std::vector<dynet::real> input_states; 
            for (int i = 0; i < this->ray_batch_size; i++){
                vec3 ray_location = ray_locations[i];
                input_states.push_back(ray_location.x);
                input_states.push_back(ray_location.y);
                input_states.push_back(ray_location.z);
            }
            dynet::Expression input_expr = dynet::input(graph, input_dim, &input_states);
            
            // 2) Choose action via importance sampling over Q-Values and take the action(CUDA Kernel)
            //    return the reward, next state and discount factor

            // 3) Calculate Q values for next state (using const_parameter)

            // 4) Get max next state Q value and use it to calulate the loss 

            // 5) Forward pass, backwards pass the loss and update the network using the trainer
        }
    }

    // Delete ray locations array
    delete [] ray_locations;
    delete [] ray_normals;
    delete [] ray_directions;
    delete [] ray_terminated;
    delete [] ray_rewards;
}

// Gets the initial direction to shoot a ray in
__global__
void get_initial_ray_dir(
        curandState* d_rand_state,
        Camera& camera, 
        int x, 
        int y, 
        vec3* ray_directions,
        vec3* ray_locations
    ){
    // Ray index
    int i =  blockIdx.x * blockDim.x + threadIdx.x;

    // Randomly sample a ray within the pixel
    Ray r = Ray::sample_ray_through_pixel(d_rand_state, camera, x, y);
    ray_locations[i] = r.start;
    ray_directions[i] = r.direction;
}

// Trace a ray for all ray locations given in the angles specified within the scene
__global__
void trace_ray(
        vec3* ray_locations, 
        vec3* ray_normals, 
        vec3* ray_directions,
        float* discount_factors, 
        bool* ray_terminated, 
        float* ray_rewards, 
        Scene* scene
    ){
    // Ray index
    int i =  blockIdx.x * blockDim.x + threadIdx.x;

    // For the current ray, get its next state by shooting a ray in the direction stored in ray_directions
    vec3 position = ray_locations[i];
    vec3 normal = ray_normals[i];
    vec3 dir = ray_directions[i];

    // Get the ray direction by converting from grid coord to 4D direction
    // int x = dir_coord/GRID_RESOLUTION;
    // int y = dir_coord - x*GRID_RESOLUTION;
    // mat4 tranformation_mat = create_transformation_matrix(normal, vec4(position,1.f));
    // vec3 dir = convert_grid_pos_to_direction((float)x, (float)y, position, tranformation_mat);

    // Create the ray and trace it
    Ray ray(vec4(position, 1.f), vec4(dir, 1.f));
    ray.closest_intersection(scene);

    // Update position, normal, and discount factor based on intersection
    switch(ray.intersection.intersection_type){

        // TERMINAL STATE: R_(t+1) = Environment light power
        case NOTHING:
            ray_terminated[i] = true;
            ray_rewards[i] = ENVIRONMENT_LIGHT;
            break;
        
        // TERMINAL STATE: R_(t+1) = Area light power
        case AREA_LIGHT:
            ray_terminated[i] = true;
            float diffuse_light_power = scene->area_lights[ray.intersection.index].luminance; 
            ray_rewards[i] = diffuse_light_power;
            break;

        // NON-TERMINAL STATE: R_(t+1) + \gamma * max_a Q(S_t+1, a) 
        // where  R_(t+1) = 0 for diffuse surfaces
        case SURFACE:
            ray_locations[i] = vec3(ray.intersection.position);
            ray_normals[i] = ray.intersection.normal;
            vec3 BRDF = scene->surfaces[ray.intersection.index].material.diffuse_c;
            
            // Get luminance of material
            float max_rgb = max(BRDF.x, BRDF.y);
            max_rgb = max(BRDF.z, max_rgb);
            float min_rgb = min(BRDF.x, BRDF.y);
            min_rgb = min(BRDF.z, min_rgb);
            float luminance = 0.5f * (max_rgb + min_rgb);

            // We calculated cos_theta before, now multiply by BRDF
            discount_factors[i] *= luminance;
            ray_rewards[i] = 0.f; // Diffuse material, no immediate reward
            break;
    }
}