#include "neural_q_pathtracer.cuh"

// 1) Create the input expression to the neural network for S_t+1
// std::vector<dynet::real> input_states; 
// int j = this->ray_batch_size * b;
// while (j < this->ray_batch_size * (b+1) && j < SCREEN_HEIGHT * SCREEN_WIDTH){
//     vec3 ray_location = ray_locations_host[ b*this->ray_batch_size + j ];
//     input_states.push_back(ray_location.x);
//     input_states.push_back(ray_location.y);
//     input_states.push_back(ray_location.z);
//     j++;
// }
// dynet::Dim input_dim({3}, int(input_states.size()/3));
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
    this->num_batches = int((SCREEN_HEIGHT * SCREEN_WIDTH)/batch_size) + 1; /* How many batches in total */
    printf("Batch Size: %d\n", batch_size);
    printf("Number of Batches: %d\n", num_batches);

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

    //////////////////////////////////////////////////////////////
    /*          Intialise Pixel value buffers                   */
    //////////////////////////////////////////////////////////////
    vec3* host_buffer = new vec3[ SCREEN_HEIGHT * SCREEN_WIDTH ];
    vec3* device_buffer;
    checkCudaErrors(cudaMalloc(&device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3)));

    //////////////////////////////////////////////////////////////
    /*          Initialise Prev Host buffers                    */
    //////////////////////////////////////////////////////////////
    float* prev_location_host = new float[ SCREEN_HEIGHT * SCREEN_WIDTH * 3 ];
    int* directions_host = new int[ SCREEN_HEIGHT * SCREEN_WIDTH ];
    
    //////////////////////////////////////////////////////////////
    /*          Initialise ray arrays on CUDA device            */
    //////////////////////////////////////////////////////////////
    float* ray_locations;   /* Ray intersection location (State) */
    float* ray_normals;     /* Intersection normal */
    float* ray_directions;  /* Direction to next shoot the ray */
    bool* ray_terminated;  /* Has the ray intersected with a light/nothing */
    float* ray_rewards;    /* Reward recieved from Q(s,a) */
    float* ray_discounts;  /* Discount factor for current rays path */
    float* ray_throughputs; /* Throughput for calc pixel value */

    checkCudaErrors(cudaMalloc(&ray_locations, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_normals, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_directions, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_terminated, sizeof(bool) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_rewards, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_discounts, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_throughputs, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    
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
            trainer, 
            d_rand_state,
            device_camera,
            device_scene,
            device_buffer,
            prev_location_host,
            directions_host,
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
    delete [] prev_location_host;
    delete [] directions_host;
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
        dynet::AdamTrainer trainer,
        curandState* d_rand_state,
        Camera* device_camera,
        Scene* device_scene,
        vec3* device_buffer,
        float* prev_location_host,
        int* directions_host,
        float* ray_locations,   /* Ray intersection location (State) */
        float* ray_normals,     /* Intersection normal */
        float* ray_directions,  /* Direction to next shoot the ray */
        bool* ray_terminated,  /* Has the ray intersected with a light/nothing */
        float* ray_rewards,    /* Reward recieved from Q(s,a) */
        float* ray_discounts,  /* Discount factor for current rays path */
        float* ray_throughputs  /* Throughput for calc pixel value */
    ){

    // Initialise the computation graph
    dynet::ComputationGraph graph;

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
        float loss = 0.f;
        while(rays_finished == 0 && bounces < MAX_RAY_BOUNCES){

            // Maintain previous locations for reinforcment Q(s,a) update
            checkCudaErrors(cudaMemcpy(prev_location_host, ray_locations, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyDeviceToHost));

            // Does not apply to shooting from camera
            if (bounces > 0){
                // No Q values sampled yet
                if( bounces == 1){
                    // Sample random directions to further trace the rays in
                    sample_next_ray_directions_randomly<<<this->num_blocks, this->block_size>>>(
                        d_rand_state,
                        ray_normals, 
                        ray_directions,
                        ray_throughputs,
                        ray_terminated
                    );
                }
                // Use Q value to sample new direction
                else{
                    // Copy over the direction sampled from the network
                    int* ray_direction_indices;
                    checkCudaErrors(cudaMalloc(&ray_direction_indices, sizeof(int) * SCREEN_HEIGHT * SCREEN_WIDTH));
                    checkCudaErrors(cudaMemcpy(ray_direction_indices, directions_host, sizeof(int) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyHostToDevice));

                    // Sample the next directions
                    sample_next_ray_directions_q_val<<<this->num_blocks, this->block_size>>>(
                        d_rand_state,
                        ray_normals,
                        ray_locations,
                        ray_directions,
                        ray_direction_indices,
                        ray_throughputs,
                        ray_terminated
                    );

                    cudaFree(ray_direction_indices);
                }
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

                // Copy data from Cuda device to host for usage
                vec3* ray_locations_host = new vec3[ SCREEN_HEIGHT * SCREEN_WIDTH ];
                float* ray_discounts_host = new float[ SCREEN_HEIGHT * SCREEN_WIDTH ];
                float* ray_rewards_host = new float[ SCREEN_HEIGHT * SCREEN_WIDTH ];
                checkCudaErrors(cudaMemcpy(ray_locations_host, ray_locations, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(ray_discounts_host, ray_discounts, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(ray_rewards_host, ray_rewards, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyDeviceToHost));

                // Run learning rule on the network with the results received and sample new direction for each ray
                for(int n = 0; n < SCREEN_HEIGHT * SCREEN_WIDTH; n++){

                    graph.clear();
                    // 1) Create the input expression to the neural network for S_t+1
                    vec3 location = ray_locations_host[n];
                    std::vector<dynet::real> input_states = {location.x, location.y, location.z}; 
                    dynet::Dim input_dim({3});
                    dynet::Expression input_expr = dynet::input(graph, input_dim, &input_states);
                    
                    // 2) Sample new direction based on: max_a Q(S_{t+1}, a)
                    dynet::Expression next_qs = this->dqn.network_inference(graph, input_expr, false);
                    std::vector<dynet::real> next_qs_vals = dynet::as_vector(graph.forward(next_qs));

                    // Get the max q val for the next state, this is the action we will take
                    int max_qs_index = 0;
                    int max_qs_value = next_qs_vals[0];
                    for (int k = 1; k < GRID_RESOLUTION*GRID_RESOLUTION; k++){
                        if (next_qs_vals[k] > max_qs_value){
                            max_qs_index = k;
                            max_qs_value = next_qs_vals[k];
                        }
                    }
                    directions_host[n] = max_qs_index;

                    // 3) Compute TD-Target
                    max_qs_value = ray_rewards_host[n] + max_qs_value*ray_discounts_host[n];

                    // 4) Reset computational graph and use target_value as a constant
                    graph.clear();
                    dynet::Expression td_target = dynet::constant(graph, {1}, max_qs_value);

                    // 5) Get current Q(s,a) value
                    location = prev_location_host[n];
                    input_states = {location.x, location.y, location.z}; ;
                    dynet::Expression input = dynet::input(graph, input_dim, &input_states);
                    dynet::Expression current_qs = dynet::max_dim(this->dqn.network_inference(graph, input, true)); // May need to softmax both
                    
                    // 6) Calculate the loss
                    dynet::Expression loss_expr = dynet::squared_distance(td_target, current_qs);
                    loss += dynet::as_scalar(graph.forward(loss_expr));

                    // 7) Train the network
                    graph.backward(loss_expr);
                    trainer.update();

                }

                // Dete the host arrays
                delete [] ray_locations_host;
                delete [] ray_discounts_host;
                delete [] ray_rewards_host;
            }

            // Copy over value to check if all rays have intersected with a light
            checkCudaErrors(cudaMemcpy(&rays_finished, device_rays_finished, sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemset(device_rays_finished, 1, sizeof(int)));

            // Increment the number of bounces
            bounces++;
        }
        printf("loss: %.3f\n",loss);

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
        float* ray_locations, 
        float* ray_directions,
        bool* ray_terminated, 
        float* ray_rewards, 
        float* ray_discounts,
        float* ray_throughputs
    ){

    // Ray index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;

    // Randomly sample a ray within the pixel
    Ray r = Ray::sample_ray_through_pixel(d_rand_state, *device_camera, x, y);
    ray_locations[(i*3)    ] = r.start.x;
    ray_locations[(i*3) + 1] = r.start.y;
    ray_locations[(i*3) + 2] = r.start.z;
    ray_directions[(i*3)    ] = r.direction.x;
    ray_directions[(i*3) + 1] = r.direction.y;
    ray_directions[(i*3) + 2] = r.direction.z;

    // Initialise ray_variables
    ray_rewards[i] = 0.f;
    ray_terminated[i] = false;
    ray_throughputs[(i*3)    ] = 1.f;
    ray_throughputs[(i*3) + 1] = 1.f;
    ray_throughputs[(i*3) + 2] = 1.f;
    ray_discounts[i] = 1.f;
}

// Trace a ray for all ray locations given in the angles specified within the scene
__global__
void trace_ray(
        Scene* scene,
        int* rays_finished,
        float* ray_locations, 
        float* ray_normals, 
        float* ray_directions,
        bool* ray_terminated, 
        float* ray_rewards,
        float* ray_discounts, 
        float* ray_throughputs
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
    vec3 position = vec3(ray_locations[(i*3)], ray_locations[(i*3)+1], ray_locations[(i*3)]+2);
    vec3 dir = vec3(ray_directions[(i*3)], ray_directions[(i*3)+1], ray_directions[(i*3)+2]);

    // Create the ray and trace it
    Ray ray(vec4(position + (dir * 0.00001f), 1.f), vec4(dir, 1.f));
    ray.closest_intersection(scene);

    // Update position, normal, and discount factor based on intersection
    switch(ray.intersection.intersection_type){

        // TERMINAL STATE: R_(t+1) = Environment light power
        case NOTHING:
            ray_terminated[i] = true;
            ray_rewards[i] = ENVIRONMENT_LIGHT;
            ray_throughputs[(i*3)] = ray_throughputs[(i*3)] * ENVIRONMENT_LIGHT;
            ray_throughputs[(i*3)+1] = ray_throughputs[(i*3)+1] * ENVIRONMENT_LIGHT;
            ray_throughputs[(i*3)+2] = ray_throughputs[(i*3)+2] * ENVIRONMENT_LIGHT;
            break;
        
        // TERMINAL STATE: R_(t+1) = Area light power
        case AREA_LIGHT:
            ray_terminated[i] = true;
            float diffuse_light_power = scene->area_lights[ray.intersection.index].luminance; 
            ray_rewards[i] = diffuse_light_power;
            
            vec3 diffuse_p = * scene->area_lights[ray.intersection.index].diffuse_p;
            ray_throughputs[(i*3)] = ray_throughputs[(i*3)] * diffuse_p.x;
            ray_throughputs[(i*3)+1] = ray_throughputs[(i*3)+1] * diffuse_p.y;
            ray_throughputs[(i*3)+2] = ray_throughputs[(i*3)+2] * diffuse_p.x.z;

            break;

        // NON-TERMINAL STATE: R_(t+1) + \gamma * max_a Q(S_t+1, a) 
        // where  R_(t+1) = 0 for diffuse surfaces
        case SURFACE:
            vec3 new_loc = vec3(ray.intersection.position);
            ray_locations[(i*3)  ] = new_loc.x;
            ray_locations[(i*3)+1] = new_loc.y;
            ray_locations[(i*3)+2] = new_loc.z;

            vec3 new_norm = ray.intersection.normal;
            ray_normals[(i*3)  ] = new_norm.x; 
            ray_normals[(i*3)+1] = new_norm.y;
            ray_normals[(i*3)+2] = new_norm.z;

            vec3 BRDF = scene->surfaces[ray.intersection.index].material.diffuse_c;
            
            // Get luminance of material
            float max_rgb = max(BRDF.x, BRDF.y);
            max_rgb = max(BRDF.z, max_rgb);
            float min_rgb = min(BRDF.x, BRDF.y);
            min_rgb = min(BRDF.z, min_rgb);
            float luminance = 0.5f * (max_rgb + min_rgb);

            // discount_factors holds cos_theta currently, update rgb throughput first
            ray_throughputs[(i*3)] = ray_throughputs[(i*3)] * (BRDF.x / (float)M_PI);
            ray_throughputs[(i*3)+1] = ray_throughputs[(i*3)+1] * (BRDF.y / (float)M_PI);
            ray_throughputs[(i*3)+2] = ray_throughputs[(i*3)+2] * (BRDF.z / (float)M_PI);

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
    vec3 dir = vec3(sample_random_direction_around_intersection(d_rand_state, ray_normals[i], cos_theta));
    ray_directions[(i*3)    ] = dir.x;
    ray_directions[(i*3) + 1] = dir.y;
    ray_directions[(i*3) + 2] = dir.z;

    // Update throughput with new sampled angle
    ray_throughputs[(i*3)    ] = (ray_throughputs[(i*3)    ] * cos_theta)/RHO;
    ray_throughputs[(i*3) + 1] = (ray_throughputs[(i*3) + 1] * cos_theta)/RHO;
    ray_throughputs[(i*3) + 2] = (ray_throughputs[(i*3) + 2] * cos_theta)/RHO;
}

// Sample ray directions according the neural network q vals
__global__
void sample_next_ray_directions_q_val(
        curandState* d_rand_state,
        float* ray_normals,
        float* ray_locations,
        float* ray_directions,
        int* ray_direction_indices,
        float* ray_throughputs,
        bool* ray_terminated
    ){
    
    // Ray Index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;

    // Do nothing if we have already intersected with the light
    if (ray_terminated[i] == true){
        return;
    }

    // Convert the index to a grid position
    int dir_grid = ray_direction_indices[i];
    int dir_x = int(dir_grid/GRID_RESOLUTION);
    int dir_y = dir_grid - (dir_x*GRID_RESOLUTION);

    // Convert to 3D direction and update the direction
    vec3 position = vec3(ray_locations[(i*3)], ray_locations[(i*3) + 1], ray_locations[(i*3) + 2]);
    vec3 normal  = vec3(ray_normals[(i*3)], ray_normals[(i*3) + 1], ray_normals[(i*3) + 2]);
    mat4 transformation_matrix = create_transformation_matrix(normal, vec4(position, 1.f));
    vec3 dir = convert_grid_pos_to_direction_random(d_rand_state, (float) dir_x, (float) dir_y, i, position, transformation_matrix);
    ray_directions[(i*3)    ] = dir.x;
    ray_directions[(i*3) + 1] = dir.y;
    ray_directions[(i*3) + 2] = dir.z;

    // Update throughput with new sampled angle
    float cos_theta = dot(normal, dir);
    ray_throughputs[(i*3)    ] = (ray_throughputs[(i*3)    ] * cos_theta)/RHO;
    ray_throughputs[(i*3) + 1] = (ray_throughputs[(i*3) + 1] * cos_theta)/RHO;
    ray_throughputs[(i*3) + 2] = (ray_throughputs[(i*3) + 2] * cos_theta)/RHO;
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