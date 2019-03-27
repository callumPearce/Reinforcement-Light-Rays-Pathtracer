#include "neural_q_pathtracer.cuh"

// 1) Get Q_values for current state for every ray in the batch
// dynet::Dim input_dim({3}, this->ray_batch_size);
// std::vector<dynet::real> input_states; 
// for (int i = 0; i < this->ray_batch_size; i++){
//     vec3 ray_location = ray_locations[i];
//     input_states.push_back(ray_location.x);
//     input_states.push_back(ray_location.y);
//     input_states.push_back(ray_location.z);
// }
// dynet::Expression input_expr = dynet::input(graph, input_dim, &input_states);

// 2) Choose action via importance sampling over Q-Values and take the action(CUDA Kernel)
//    return the reward, next state and discount factor

// 3) Calculate Q values for next state (using const_parameter)

// 4) Get max next state Q value and use it to calulate the loss 

// 5) Forward pass, backwards pass the loss and update the network using the trainer


__host__
NeuralQPathtracer::NeuralQPathtracer(
        unsigned int frames,
        unsigned int batch_size,
        SDLScreen& screen,
        Scene& scene,
        Camera& camera,
        int argc,
        char** argv
    ){

    //////////////////////////////////////////////////////////////
    /*                  Assign attributes                       */
    //////////////////////////////////////////////////////////////
    this->ray_batch_size = batch_size; /* How many rays to be processed at once */
    this->num_batches = int((SCREEN_HEIGHT * SCREEN_WIDTH)/batch_size) - 1; /* How many batches in total */

    dim3 b_size(8,8);
    this->block_size = b_size; /* How many threads in a single block to process the screen*/
    int blocks_x = (SCREEN_WIDTH + this->block_size.x - 1)/this->block_size.x;
    int blocks_y = (SCREEN_HEIGHT + this->block_size.y - 1)/this->block_size.y;
    dim3 n_bs(blocks_x, blocks_y);
    this->num_blocks = n_bs;/* How many blocks to process all pixels on the screen */

    //////////////////////////////////////////////////////////////
    /*                Initialise the DQN                        */
    //////////////////////////////////////////////////////////////
    //TODO: Might have to specify the amount of memory the GPU can use
    // beforehand, otherwise it seems to assign over memory allocated later
    // on. It may continue to do this when calculating back&forwad prop
    auto dyparams = dynet::extract_dynet_params(argc, argv);
    dynet::initialize(dyparams);
    dynet::ParameterCollection model;
    dynet::AdamTrainer trainer(model);
    this->dqn = DQNetwork();
    this->dqn.initialize(model);
    dynet::ComputationGraph graph;

    //////////////////////////////////////////////////////////////
    /*          Intialise Pixel value buffers                   */
    //////////////////////////////////////////////////////////////
    vec3* host_buffer = new vec3[ SCREEN_HEIGHT * SCREEN_WIDTH ];
    vec3* device_buffer;
    checkCudaErrors(cudaMalloc(&device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3)));
    
    //////////////////////////////////////////////////////////////
    /*          Initialise ray arrays on CUDA device            */
    //////////////////////////////////////////////////////////////
    vec3* ray_locations;   /* Ray intersection location (State) */
    vec3* ray_normals;     /* Intersection normal */
    vec3* ray_directions;  /* Direction to next shoot the ray */
    bool* ray_terminated;  /* Has the ray intersected with a light/nothing */
    float* ray_rewards;    /* Reward recieved from Q(s,a) */
    float* ray_discounts;  /* Discount factor for current rays path */
    vec3* ray_throughputs; /* Throughput for calc pixel value */

    checkCudaErrors(cudaMalloc(&ray_locations, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_normals, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_directions, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_terminated, sizeof(bool) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_rewards, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_discounts, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_throughputs, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH));
    
    Camera* device_camera; /* Camera on the CUDA device */
    Surface* device_surfaces;
    AreaLight* device_light_planes;
    Scene* device_scene;   /* Scenes to render */

    // Copy the camera
    checkCudaErrors(cudaMalloc(&device_camera, sizeof(Camera)));
    checkCudaErrors(cudaMemcpy(device_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
    
    // Copy surfaces into device memory space
    checkCudaErrors(cudaMalloc(&device_surfaces, scene.surfaces_count * sizeof(Surface)));
    checkCudaErrors(cudaMemcpy(device_surfaces, scene.surfaces, scene.surfaces_count * sizeof(Surface), cudaMemcpyHostToDevice));

    // Copy light planes into device memory space
    checkCudaErrors(cudaMalloc(&device_light_planes, scene.area_light_count * sizeof(AreaLight)));
    checkCudaErrors(cudaMemcpy(device_light_planes, scene.area_lights, scene.area_light_count * sizeof(AreaLight), cudaMemcpyHostToDevice));    

    // Copy the scene structure into the device and its corresponding pointers to Surfaces and Area Lights
    checkCudaErrors(cudaMalloc(&device_scene, sizeof(Scene)));
    checkCudaErrors(cudaMemcpy(device_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(device_scene->surfaces), &device_surfaces, sizeof(Surface*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(device_scene->area_lights), &device_light_planes, sizeof(AreaLight*), cudaMemcpyHostToDevice));    
    
    //////////////////////////////////////////////////////////////
    /*                  Intialise cuRand State                  */
    //////////////////////////////////////////////////////////////
    curandState * d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, (float)SCREEN_HEIGHT * (float)SCREEN_WIDTH * sizeof(curandState)));
    init_rand_state<<<this->num_blocks, this->block_size>>>(d_rand_state, SCREEN_WIDTH, SCREEN_HEIGHT);

    //////////////////////////////////////////////////////////////
    /*                  Render the frames                       */
    //////////////////////////////////////////////////////////////
    for (int i = 0; i < frames; i++){
        //Clear the pixel buffer
        memset(host_buffer, 0.f, sizeof(vec3)* SCREEN_HEIGHT * SCREEN_WIDTH);

        /* Compute frame time */
        static int t = SDL_GetTicks();
        int t2 = SDL_GetTicks();
        float dt = float(t2-t);
        t = t2;
        printf("Render Time: %.3f ms.\n", dt);
        
        // Fill the pixel buffer each frame using Deep Q-Learning strategy
        this->render_frame(
            // graph, 
            d_rand_state,
            device_camera,
            device_scene,
            device_buffer,
            ray_locations,
            ray_normals,   
            ray_directions,  
            ray_terminated,  
            ray_rewards,   
            ray_discounts, 
            ray_throughputs
        );

        // Copy the device buffer values to the host buffer
        checkCudaErrors(cudaMemcpy(host_buffer, device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3), cudaMemcpyDeviceToHost));

        // Display the rendered frame
        for (int x = 0; x < SCREEN_WIDTH; x++){
            for (int y = 0; y < SCREEN_HEIGHT; y++){
                screen.PutPixelSDL(x, y, host_buffer[x*(int)SCREEN_HEIGHT + y]);
            }
        }
        screen.SDL_Renderframe();
    }

    //////////////////////////////////////////////////////////////
    /*          Save the image and kill the screen              */
    //////////////////////////////////////////////////////////////
    screen.SDL_SaveImage("../Images/render.bmp");
    screen.kill_screen();

    //////////////////////////////////////////////////////////////
    /*                      Free memory used                    */
    //////////////////////////////////////////////////////////////
    delete [] host_buffer;
    cudaFree(device_buffer);
    cudaFree(d_rand_state);
    cudaFree(ray_locations);
    cudaFree(ray_normals);
    cudaFree(ray_directions);
    cudaFree(ray_terminated);
    cudaFree(ray_rewards);
    cudaFree(ray_throughputs);
    cudaFree(device_camera);
    cudaFree(device_surfaces);
    cudaFree(device_light_planes);
    cudaFree(device_scene);
}

__host__
void NeuralQPathtracer::render_frame(
        // dynet::ComputationGraph& graph,
        curandState* d_rand_state,
        Camera* device_camera,
        Scene* device_scene,
        vec3* device_buffer,
        vec3* ray_locations,   /* Ray intersection location (State) */
        vec3* ray_normals,     /* Intersection normal */
        vec3* ray_directions,  /* Direction to next shoot the ray */
        bool* ray_terminated,  /* Has the ray intersected with a light/nothing */
        float* ray_rewards,    /* Reward recieved from Q(s,a) */
        float* ray_discounts,  /* Discount factor for current rays path */
        vec3* ray_throughputs  /* Throughput for calc pixel value */
    ){
    
    // Initialise buffer to hold total throughput
    vec3* total_throughputs;
    checkCudaErrors(cudaMalloc(&total_throughputs, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMemset(total_throughputs, 0.f, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH));
    
    // Sample through each pixel SAMPLES_PER_PIXEL times
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++){
        // Initialise mini-batch ray variables
        initialise_ray<<<this->num_blocks, this->block_size>>>(
            d_rand_state,
            device_camera, 
            ray_locations, 
            ray_directions,
            ray_terminated, 
            ray_rewards, 
            ray_discounts,
            ray_throughputs
        );
        checkCudaErrors(cudaDeviceSynchronize());

        // Create bool to determine if all rays in the batch have collided with a light
        int rays_finished = 0; /* If not updated to false by trace_ray, end loop */
        int* device_rays_finished;
        checkCudaErrors(cudaMalloc(&device_rays_finished, sizeof(int)));
        checkCudaErrors(cudaMemset(device_rays_finished, 1, sizeof(int)));

        // Trace batch rays path until all have intersected with a light
        unsigned int bounces = 0;
        while(rays_finished == 0 && bounces < MAX_RAY_BOUNCES){

            // Does not apply to shooting from camera
            if (bounces > 0){
                // Get the direction to trace each ray in   
                // Sample random directions to further trace the rays in
                sample_next_ray_directions_randomly<<<this->num_blocks, this->block_size>>>(
                    d_rand_state,
                    ray_normals, 
                    ray_directions,
                    ray_throughputs,
                    ray_terminated
                );
                cudaDeviceSynchronize();
            }

            // Trace the rays in their set directions
            trace_ray<<<this->num_blocks, this->block_size>>>(
                device_scene,
                device_rays_finished,
                ray_locations, 
                ray_normals,
                ray_directions, 
                ray_terminated, 
                ray_rewards,
                ray_discounts,
                ray_throughputs
            );  
            cudaDeviceSynchronize();

            // Does not apply to shooting from camera
            if(bounces > 0){
                // Run learning rule on the network with the results received
            }

            // Copy over value to check if all rays have intersected with a light
            checkCudaErrors(cudaMemcpy(&rays_finished, device_rays_finished, sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemset(device_rays_finished, 1, sizeof(int)));

            // Increment the number of bounces
            bounces++;
        }

        // Add computed throughput values to the running total
        update_total_throughput<<<this->num_blocks, this->block_size>>>(
            ray_throughputs,
            total_throughputs
        );
        cudaDeviceSynchronize();
        cudaFree(device_rays_finished);
    }
    // Update the device_buffer with the throughput
    update_device_buffer<<<this->num_blocks, this->block_size>>>(
        device_buffer,
        total_throughputs
    );
    cudaDeviceSynchronize();
    cudaFree(total_throughputs);
}

// Gets the initial direction to shoot a ray in
__global__
void initialise_ray(
        curandState* d_rand_state,
        Camera* device_camera, 
        vec3* ray_locations, 
        vec3* ray_directions,
        bool* ray_terminated, 
        float* ray_rewards, 
        float* ray_discounts,
        vec3* ray_throughputs
    ){

    // Ray index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;

    // Randomly sample a ray within the pixel
    Ray r = Ray::sample_ray_through_pixel(d_rand_state, *device_camera, x, y);
    ray_locations[i] = r.start;
    ray_directions[i] = r.direction;

    // Initialise ray_variables
    ray_rewards[i] = 0.f;
    ray_terminated[i] = false;
    ray_throughputs[i] = vec3(1.f);
    ray_discounts[i] = 1.f;
}

// Trace a ray for all ray locations given in the angles specified within the scene
__global__
void trace_ray(
        Scene* scene,
        int* rays_finished,
        vec3* ray_locations, 
        vec3* ray_normals, 
        vec3* ray_directions,
        bool* ray_terminated, 
        float* ray_rewards,
        float* ray_discounts, 
        vec3* ray_throughputs
    ){
    
    // Ray index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;

    // Do nothing if we have already intersected with the light
    if (ray_terminated[i] == true){
        return;
    }

    // For the current ray, get its next state by shooting a ray in the direction stored in ray_directions
    vec3 position = ray_locations[i];
    vec3 dir = ray_directions[i];

    // Create the ray and trace it
    Ray ray(vec4(position + (dir * 0.00001f), 1.f), vec4(dir, 1.f));
    ray.closest_intersection(scene);

    // Update position, normal, and discount factor based on intersection
    switch(ray.intersection.intersection_type){

        // TERMINAL STATE: R_(t+1) = Environment light power
        case NOTHING:
            ray_terminated[i] = true;
            ray_rewards[i] = ENVIRONMENT_LIGHT;
            ray_throughputs[i] = ray_throughputs[i] * vec3(ENVIRONMENT_LIGHT);
            break;
        
        // TERMINAL STATE: R_(t+1) = Area light power
        case AREA_LIGHT:
            ray_terminated[i] = true;
            float diffuse_light_power = scene->area_lights[ray.intersection.index].luminance; 
            ray_rewards[i] = diffuse_light_power;
            ray_throughputs[i] = ray_throughputs[i] * scene->area_lights[ray.intersection.index].diffuse_p;
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

            // discount_factors holds cos_theta currently, update rgb throughput first
            ray_throughputs[i] = ray_throughputs[i] * (BRDF / (float)M_PI); 
            // Now update discount_factors with luminance
            ray_discounts[i] *= luminance;
            // Still a ray being to bounce, so not finished
            atomicExch(rays_finished, 0);
            break;
    }
}

// Sample random directions to further trace the rays in
__global__
void sample_next_ray_directions_randomly(
        curandState* d_rand_state,
        vec3* ray_normals, 
        vec3* ray_directions,
        vec3* ray_throughputs,
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
    ray_directions[i] = vec3(sample_random_direction_around_intersection(d_rand_state, ray_normals[i], cos_theta));
    
    // Update throughput with new sampled angle
    ray_throughputs[i] = (ray_throughputs[i] * cos_theta)/RHO;
}

// Update pixel values stored in the device_buffer
__global__
void update_total_throughput(
        vec3* ray_throughputs,
        vec3* total_throughputs
    ){

    // Ray index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;
    
    // Accumulate
    total_throughputs[i] += ray_throughputs[i];
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