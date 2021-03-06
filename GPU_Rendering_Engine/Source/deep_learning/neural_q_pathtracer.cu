#include "neural_q_pathtracer.cuh"


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
    /*                  Estimate Render Time                    */
    //////////////////////////////////////////////////////////////
    const float time_per_bounce = 2.f;
    std::cout << "Weakly Estimated Frame Render Time: " << time_per_bounce * (float)MAX_RAY_BOUNCES * (float)SAMPLES_PER_PIXEL << "s" << std::endl;

    //////////////////////////////////////////////////////////////
    /*                  Assign attributes                       */
    //////////////////////////////////////////////////////////////
    this->epsilon = EPSILON_START;
    this->vertices_count = scene.vertices_count;
    this->ray_batch_size = batch_size; /* How many rays to be processed at once */
    this->num_batches = (SCREEN_HEIGHT*SCREEN_WIDTH + (batch_size -1))/batch_size; /* How many batches in total */
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
    unsigned int input_dim = TRAIN_ON_POSITION ? 3 : this->vertices_count;
    this->dqn.initialize(model, input_dim /* Input dims */, GRID_RESOLUTION*GRID_RESOLUTION /* Output dims */);

    //////////////////////////////////////////////////////////////
    /*                Load the previous DQN Model               */
    //////////////////////////////////////////////////////////////
    std::string fname = "../Radiance_Map_Data/deep_q_learning_12_12.model";
    if (LOAD_MODEL && file_exists(fname)){
        dynet::TextFileLoader loader(fname);
        loader.populate(model);
    }

    //////////////////////////////////////////////////////////////
    /*          Intialise Pixel value buffers                   */
    //////////////////////////////////////////////////////////////
    vec3* host_buffer = new vec3[ SCREEN_HEIGHT * SCREEN_WIDTH ];
    vec3* device_buffer;
    checkCudaErrors(cudaMalloc(&device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3)));
    
    //////////////////////////////////////////////////////////////
    /*          Initialise Prev Host buffers                    */
    //////////////////////////////////////////////////////////////
    unsigned int* directions_host = new unsigned int[ SCREEN_HEIGHT * SCREEN_WIDTH ];

    //////////////////////////////////////////////////////////////
    /*          Initialise ray arrays on CUDA device            */
    //////////////////////////////////////////////////////////////
    float* ray_locations;   /* Ray intersection location (State) */
    float* prev_ray_locations; /* Location the ray was previously sampled from */
    float* ray_normals;     /* Intersection normal */
    float* ray_directions;  /* Direction to next shoot the ray */
    unsigned int* ray_states;  /* Ray can either be in the process of being traced, terminated, or post tracing for learning */
    float* ray_rewards;    /* Reward recieved from Q(s,a) */
    float* ray_discounts;  /* Discount factor for current rays path */
    float* ray_throughputs; /* Throughput for calc pixel value */
    unsigned int* ray_bounces; /* Total number of bounces for each ray before intersection*/
    float* ray_vertices;    /* Stores vertices in coordinate system centered at the current ray position */

    checkCudaErrors(cudaMalloc(&ray_locations, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&prev_ray_locations, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_normals, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_directions, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_states, sizeof(unsigned int) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_rewards, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_discounts, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_throughputs, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_bounces, sizeof(unsigned int) *SCREEN_HEIGHT *SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_vertices, sizeof(float) * this->vertices_count * SCREEN_HEIGHT * SCREEN_WIDTH));
    
    Camera* device_camera; /* Camera on the CUDA device */
    Surface* device_surfaces;
    AreaLight* device_light_planes;
    float* device_vertices;
    Scene* device_scene;   /* Scene to render */

    // Copy the camera
    checkCudaErrors(cudaMalloc(&device_camera, sizeof(Camera)));
    checkCudaErrors(cudaMemcpy(device_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
    
    // Copy surfaces into device memory space
    checkCudaErrors(cudaMalloc(&device_surfaces, scene.surfaces_count * sizeof(Surface)));
    checkCudaErrors(cudaMemcpy(device_surfaces, scene.surfaces, scene.surfaces_count * sizeof(Surface), cudaMemcpyHostToDevice));

    // Copy light planes into device memory space
    checkCudaErrors(cudaMalloc(&device_light_planes, scene.area_light_count * sizeof(AreaLight)));
    checkCudaErrors(cudaMemcpy(device_light_planes, scene.area_lights, scene.area_light_count * sizeof(AreaLight), cudaMemcpyHostToDevice));    

    // Copy vertices into device memory space
    checkCudaErrors(cudaMalloc(&device_vertices, scene.vertices_count * sizeof(float)));
    checkCudaErrors(cudaMemcpy(device_vertices, scene.vertices, scene.vertices_count * sizeof(float), cudaMemcpyHostToDevice));  

    // Copy the scene structure into the device and its corresponding pointers to Surfaces, Area Lights and Vertices
    checkCudaErrors(cudaMalloc(&device_scene, sizeof(Scene)));
    checkCudaErrors(cudaMemcpy(device_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(device_scene->surfaces), &device_surfaces, sizeof(Surface*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(device_scene->area_lights), &device_light_planes, sizeof(AreaLight*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(device_scene->vertices), &device_vertices, sizeof(float*), cudaMemcpyHostToDevice));

    //////////////////////////////////////////////////////////////
    /*                  Intialise cuRand State                  */
    //////////////////////////////////////////////////////////////
    curandState * d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, (float)SCREEN_HEIGHT * (float)SCREEN_WIDTH * sizeof(curandState)));
    init_rand_state<<<this->num_blocks, this->block_size>>>(d_rand_state, SCREEN_WIDTH, SCREEN_HEIGHT);

    //////////////////////////////////////////////////////////////
    /*               Reset the file to write to                 */
    //////////////////////////////////////////////////////////////
    if (SAVE_TRAINING_STATS){
        if (file_exists("../Radiance_Map_Data/nn_training_stats.txt"))
            std::remove("../Radiance_Map_Data/nn_training_stats.txt");
    }

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
            device_vertices,
            directions_host,
            ray_locations,
            prev_ray_locations,
            ray_normals,   
            ray_directions,
            ray_states,  
            ray_rewards,   
            ray_discounts, 
            ray_throughputs,
            ray_bounces,
            ray_vertices
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

        //////////////////////////////////////////////////////////////
        /*                  Save the DQN Model                      */
        //////////////////////////////////////////////////////////////
        std::cout << "Saving model..." << std::endl;
        if (SAVE_MODEL){
            dynet::TextFileSaver saver(fname);
            saver.save(model);
        }
        std::cout << "Model saved." << std::endl;
    }

    //////////////////////////////////////////////////////////////
    /*                        Save the image                    */
    //////////////////////////////////////////////////////////////
    screen.SDL_SaveImage("../Images/render.bmp");

    //////////////////////////////////////////////////////////////
    /*                      Free memory used                    */
    //////////////////////////////////////////////////////////////
    delete [] host_buffer;
    delete [] directions_host;
    cudaFree(device_buffer);
    cudaFree(d_rand_state);
    cudaFree(ray_locations);
    cudaFree(ray_normals);
    cudaFree(ray_directions);
    cudaFree(ray_states);
    cudaFree(ray_rewards);
    cudaFree(ray_throughputs);
    cudaFree(ray_bounces);
    cudaFree(device_camera);
    cudaFree(device_surfaces);
    cudaFree(device_light_planes);
    cudaFree(device_vertices);
    cudaFree(device_scene);
}

__host__
void NeuralQPathtracer::render_frame(
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
    ){

    // Initialise buffer to hold total throughput
    vec3* total_throughputs;
    checkCudaErrors(cudaMalloc(&total_throughputs, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMemset(total_throughputs, 0.f, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH));
    
    // Sample through each pixel SAMPLES_PER_PIXEL times
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++){
        // Initialise ray variables
        initialise_ray<<<this->num_blocks, this->block_size>>>(
            d_rand_state,
            device_camera, 
            ray_locations, 
            prev_ray_locations,
            ray_directions,
            ray_states, 
            ray_rewards, 
            ray_discounts,
            ray_throughputs,
            ray_bounces
        );
        checkCudaErrors(cudaDeviceSynchronize());

        // Create bool to determine if all rays in the batch have collided with a light
        int rays_finished = 0; /* If not updated to false by trace_ray, end loop */
        int* device_rays_finished;
        checkCudaErrors(cudaMalloc(&device_rays_finished, sizeof(int)));
        checkCudaErrors(cudaMemset(device_rays_finished, 1, sizeof(int)));

        // Trace rays path until all have intersected with a light
        unsigned int bounces = 0;
        float loss = 0.f;
        while(rays_finished == 0 && bounces < MAX_RAY_BOUNCES){

            printf("Bounce: %d/%d\n", bounces, MAX_RAY_BOUNCES);

            std::chrono::time_point<std::chrono::high_resolution_clock> start;
            if (TIMING){
                // TIMER START: Sampling ray directions
                start = std::chrono::high_resolution_clock::now();
            }

            // Does not apply to shooting from camera
            if (bounces > 0){

                // Convert vertices into coordinate system centred around each rays current location
                // Get vertices in coordinate system surrounding current location
                convert_vertices_to_point_coord_system<<<this->num_blocks, this->block_size>>>(
                    ray_vertices, 
                    ray_locations,
                    device_vertices,
                    this->vertices_count
                );
                cudaDeviceSynchronize();

                // For each batch sample Q-values and apply eta-greedy policy
                for(int n = 0; n < this->num_batches; n++){
                    
                    // Compute Batch-Size
                    unsigned int current_batch_size = std::min(SCREEN_HEIGHT*SCREEN_WIDTH - (n*this->ray_batch_size), this->ray_batch_size);

                    // Initialise the graph
                    dynet::ComputationGraph graph;
                    
                    // Formulate the expression with the state and the scenes vertices
                    dynet::Dim input_dim;
                    std::vector<float> input_vals;
                    if (TRAIN_ON_POSITION){
                        input_dim = dynet::Dim({(unsigned int)3},current_batch_size);
                        input_vals = std::vector<float>(3*current_batch_size);
                        checkCudaErrors(cudaMemcpy(&(input_vals[0]), &(ray_locations[n*this->ray_batch_size*3]), sizeof(float) * 3 * current_batch_size, cudaMemcpyDeviceToHost));
                    } else{
                        input_dim = dynet::Dim({(unsigned int)this->vertices_count},current_batch_size);
                        input_vals = std::vector<float>(this->vertices_count*current_batch_size);
                        checkCudaErrors(cudaMemcpy(&(input_vals[0]), &(ray_vertices[n*this->ray_batch_size*this->vertices_count]), sizeof(float) * this->vertices_count * current_batch_size, cudaMemcpyDeviceToHost));
                    }
                    dynet::Expression states_batch = dynet::input(graph, input_dim, input_vals); 
                    
                    // Get the Q-values
                    dynet::Expression prediction = this->dqn.network_inference(graph, states_batch, false);
                    std::vector<float> current_qs = dynet::as_vector(graph.forward(prediction));

                    // Copy Q-vals to GPU for find the argmax
                    float* current_qs_device;
                    checkCudaErrors(cudaMalloc(&current_qs_device, sizeof(float) * current_qs.size()));
                    checkCudaErrors(cudaMemcpy(current_qs_device, &(current_qs[0]), sizeof(float) * current_qs.size() , cudaMemcpyHostToDevice));

                    // Setup the deivce storage for the ray direction indices
                    unsigned int* ray_direction_indices;
                    checkCudaErrors(cudaMalloc(&ray_direction_indices, sizeof(unsigned int) * current_batch_size));

                    // Get direction indices (Call once for every element in the batch)
                    int threads = 32;
                    int blocks = (current_batch_size + (threads-1))/threads;
                    sample_batch_ray_directions_epsilon_greedy<<<blocks, threads>>>(
                        this->epsilon,
                        d_rand_state,
                        ray_direction_indices,
                        current_qs_device,
                        ray_directions,
                        ray_locations,
                        ray_normals,
                        ray_throughputs,
                        ray_states,
                        (n*this->ray_batch_size)
                    );
                    cudaDeviceSynchronize();

                    // Copy over ray_direction_indices to the host
                    checkCudaErrors(cudaMemcpy(&(directions_host[ this->ray_batch_size * n ]), ray_direction_indices, sizeof(unsigned int) * current_batch_size, cudaMemcpyDeviceToHost));

                    // Free memory
                    cudaFree(ray_direction_indices);
                    cudaFree(current_qs_device);
                }
            }

            if (TIMING){
                // TIMER END: Sampling ray directions
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "Sampling Ray Dir Time: " << elapsed.count() << "s" << std::endl;
            }

            if (TIMING){
                // TIMER START: Tracing rays
                start = std::chrono::high_resolution_clock::now();
            }
            // Trace the rays in their set directions
            trace_ray<<<this->num_blocks, this->block_size>>>(
                device_scene,
                d_rand_state,
                device_rays_finished,
                ray_locations, 
                prev_ray_locations,
                ray_normals,
                ray_directions, 
                ray_states, 
                ray_rewards,
                ray_discounts,
                ray_throughputs,
                ray_bounces,
                bounces
            );  
            cudaDeviceSynchronize();

            if (TIMING){
                // TIMER END: Tracing rays
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "Tracing Rays Time: " << elapsed.count() << "s" << std::endl;
            }

            if (TIMING){
                // TIMER START: Train network
                start = std::chrono::high_resolution_clock::now();
            }
            // Does not apply to shooting from camera //TODO: We are current backpropogating rays that have terminated continually, this is bad
            if(bounces > 0){

                // Make a copy of the old converted vertices
                float* prev_ray_vertices;
                checkCudaErrors(cudaMalloc(&prev_ray_vertices, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH * this->vertices_count));
                checkCudaErrors(cudaMemcpy(prev_ray_vertices, ray_vertices, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH * this->vertices_count, cudaMemcpyDeviceToDevice));

                // Convert vertices into coordinate system centred around each rays current location
                // Get vertices in coordinate system surrounding current location
                convert_vertices_to_point_coord_system<<<this->num_blocks, this->block_size>>>(
                    ray_vertices, 
                    ray_locations,
                    device_vertices,
                    this->vertices_count
                );
                cudaDeviceSynchronize();

                // Run learning rule on the network with the results received and sample new direction for each ray in batches
                for(int n = 0; n < this->num_batches; n++){
                    
                    dynet::ComputationGraph graph;

                    // 1) Create the input expression to the neural network for S_t+1
                    unsigned int current_batch_size = std::min(SCREEN_HEIGHT*SCREEN_WIDTH - (n*this->ray_batch_size), this->ray_batch_size);

                    dynet::Dim input_dim;
                    std::vector<float> input_vals;
                    if (TRAIN_ON_POSITION){
                        input_dim = dynet::Dim({(unsigned int)3},current_batch_size);
                        input_vals = std::vector<float>(3*current_batch_size);
                        checkCudaErrors(cudaMemcpy(&(input_vals[0]), &(ray_locations[n*this->ray_batch_size*3]), sizeof(float) * 3 * current_batch_size, cudaMemcpyDeviceToHost));
                    } else{
                        input_dim = dynet::Dim({(unsigned int)this->vertices_count},current_batch_size);
                        input_vals = std::vector<float>(this->vertices_count*current_batch_size);
                        checkCudaErrors(cudaMemcpy(&(input_vals[0]), &(ray_vertices[n*this->ray_batch_size*this->vertices_count]), sizeof(float) * this->vertices_count * current_batch_size, cudaMemcpyDeviceToHost));
                    }

                    dynet::Expression input_batch = dynet::input(graph, input_dim, input_vals); 

                    // 2) Get max_a Q(S_{t+1}, a)
                    dynet::Expression next_qs_expr = this->dqn.network_inference(graph, input_batch, false);
                    std::vector<float> next_qs = dynet::as_vector(graph.forward(next_qs_expr));

                    // 3) Compute TD-Targets
                    float* next_qs_device;
                    checkCudaErrors(cudaMalloc(&next_qs_device, sizeof(float) * current_batch_size * GRID_RESOLUTION * GRID_RESOLUTION));
                    checkCudaErrors(cudaMemcpy(next_qs_device, &(next_qs[0]), sizeof(float) * current_batch_size * GRID_RESOLUTION * GRID_RESOLUTION, cudaMemcpyHostToDevice));

                    float* td_targets_device;
                    checkCudaErrors(cudaMalloc(&td_targets_device, sizeof(float) * current_batch_size));

                    int threads = 32;
                    int blocks = int((current_batch_size + (threads-1))/threads);
                    compute_td_targets<<<blocks, threads>>>(
                        d_rand_state,
                        next_qs_device,
                        td_targets_device,
                        ray_locations,
                        ray_normals,
                        ray_rewards,
                        ray_discounts,
                        ray_states,
                        (n*this->ray_batch_size)
                    );
                    cudaDeviceSynchronize();

                    cudaFree(next_qs_device);

                    std::vector<float> td_targets(current_batch_size);
                    checkCudaErrors(cudaMemcpy(&(td_targets[0]), td_targets_device, sizeof(float) * current_batch_size, cudaMemcpyDeviceToHost));
                    cudaFree(td_targets_device);

                    // 4) Reset computational graph and use target_value as a constant
                    graph.clear();
                    dynet::Expression td_target = dynet::input(graph, dynet::Dim({1}, current_batch_size), td_targets);

                    // // 5) Get current Q(s,a) value
                    // Formulate the expression with the state and the scenes vertices
                    std::vector<float> input_curr_state;
                    if (TRAIN_ON_POSITION){
                        input_dim = dynet::Dim({(unsigned int)3},current_batch_size);
                        input_curr_state = std::vector<float>(3*current_batch_size);
                        checkCudaErrors(cudaMemcpy(&(input_curr_state[0]), &(prev_ray_locations[n*this->ray_batch_size*3]), sizeof(float) * 3 * current_batch_size, cudaMemcpyDeviceToHost));
                    } else{
                        input_dim = dynet::Dim({(unsigned int)this->vertices_count},current_batch_size);
                        input_curr_state = std::vector<float>(this->vertices_count*current_batch_size);
                        checkCudaErrors(cudaMemcpy(&(input_curr_state[0]), &(prev_ray_vertices[n*this->ray_batch_size*this->vertices_count]), sizeof(float) * this->vertices_count * current_batch_size, cudaMemcpyDeviceToHost));
                    }

                    // std::vector<float> input_curr_state(this->vertices_count*current_batch_size);
                    // checkCudaErrors(cudaMemcpy(&(input_curr_state[0]), &(prev_ray_vertices[n*this->ray_batch_size*this->vertices_count]), sizeof(float) * this->vertices_count * current_batch_size, cudaMemcpyDeviceToHost));
                    dynet::Expression states_batch = dynet::input(graph, input_dim, input_curr_state); 

                    dynet::Expression prediction_qs = this->dqn.network_inference(graph, states_batch, true);
                    
                    // Get the vector of action value indices we took 
                    std::vector<unsigned int> action_value_indices(current_batch_size);
                    memcpy(&(action_value_indices[0]), &directions_host[this->ray_batch_size*n], sizeof(unsigned int) * current_batch_size);

                    // Get the current Q values for the actions taken
                    dynet::Expression current_qs = dynet::pick(prediction_qs, action_value_indices, (unsigned int) 0);
                    
                    // 6) Calculate the loss
                    dynet::Expression loss_expr = dynet::pow((td_target - current_qs), dynet::input(graph, 2.f));  
                    loss_expr = dynet::sum_batches(loss_expr); 
                    loss += dynet::as_scalar(graph.forward(loss_expr));

                    // 7) Train the network
                    graph.backward(loss_expr);
                    trainer.update();
                }

                cudaFree(prev_ray_vertices);
            }

            // Sample a new direction for rays which have reached the terminal state
            sample_random_scene_pos_for_terminated_rays<<<this->num_blocks, this->block_size>>>(
                device_scene,
                d_rand_state,
                ray_normals,
                ray_locations,
                ray_states
            );
            cudaDeviceSynchronize();

            if (TIMING){
                // TIMER END: Train network
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "Train Network Time: " << elapsed.count() << "s" << std::endl;
            }

            // Copy over value to check if all rays have intersected with a light
            checkCudaErrors(cudaMemcpy(&rays_finished, device_rays_finished, sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemset(device_rays_finished, 1, sizeof(int)));

            // Increment the number of bounces
            bounces++;
        }

        // Update epsilon
        this->epsilon = std::max( this->epsilon - EPSILON_DECAY, EPSILON_MIN);

        printf("epsilon: %.3f\n", this->epsilon);

        printf("loss: %.3f\n",loss);

        // Calculate the average path length
        int* total_path_lengths_device;
        checkCudaErrors(cudaMalloc(&total_path_lengths_device, sizeof(int)));
        checkCudaErrors(cudaMemset(total_path_lengths_device, 0, sizeof(int)));
        sum_path_lengths<<<this->num_blocks, this->block_size>>>(
            total_path_lengths_device,
            ray_bounces
        );
        int total_path_lengths = 0;
        checkCudaErrors(cudaMemcpy(&total_path_lengths, total_path_lengths_device, sizeof(int), cudaMemcpyDeviceToHost));
        float avg_path_length = total_path_lengths/((float)SCREEN_HEIGHT*(float)SCREEN_WIDTH);
        std::cout << "Avg Path Length: " << avg_path_length << std::endl;
        cudaFree(total_path_lengths_device);

        // Calculate the number of zero-contribution light paths by throughputs below THROUGHPUT_THRESHOLD
        int* total_zero_contribution_light_paths;
        checkCudaErrors(cudaMalloc(&total_zero_contribution_light_paths, sizeof(int)));
        checkCudaErrors(cudaMemset(total_zero_contribution_light_paths, 0, sizeof(int)));
        sum_zero_contribution_light_paths<<<this->num_blocks, this->block_size>>>(
            total_zero_contribution_light_paths,
            ray_throughputs
        );
        int total_zclp = 0;
        checkCudaErrors(cudaMemcpy(&total_zclp, total_zero_contribution_light_paths, sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << "Total zero contribution light paths: " << total_zclp << std::endl;
        cudaFree(total_zero_contribution_light_paths);

        // Save collected stats to the file
        if (SAVE_TRAINING_STATS){
            std::ofstream stats_file;   
            stats_file.open("../Radiance_Map_Data/nn_training_stats.txt", std::ios::app);
            stats_file << avg_path_length << " " << loss << " " << total_zclp << "\n";
            stats_file.close();
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
        float* ray_locations, 
        float* prev_ray_locations,
        float* ray_directions,
        unsigned int* ray_states, 
        float* ray_rewards, 
        float* ray_discounts,
        float* ray_throughputs,
        unsigned int* ray_bounces
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
    prev_ray_locations[(i*3)    ] = r.start.x;
    prev_ray_locations[(i*3) + 1] = r.start.y;
    prev_ray_locations[(i*3) + 2] = r.start.z;

    ray_directions[(i*3)    ] = r.direction.x;
    ray_directions[(i*3) + 1] = r.direction.y;
    ray_directions[(i*3) + 2] = r.direction.z;

    // Initialise ray_variables
    ray_rewards[i] = 0.f;
    ray_states[i] = 0;
    ray_throughputs[(i*3)    ] = 1.f;
    ray_throughputs[(i*3) + 1] = 1.f;
    ray_throughputs[(i*3) + 2] = 1.f;
    ray_discounts[i] = 1.f;
    ray_bounces[i] = MAX_RAY_BOUNCES;
}

// Trace a ray for all ray locations given in the angles specified within the scene
__global__
void trace_ray(
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
    ){
    
    // Ray index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;


    // For the current ray, get its next state by shooting a ray in the direction stored in ray_directions
    vec3 position = vec3(ray_locations[(i*3)], ray_locations[(i*3)+1], ray_locations[(i*3)+2]);
    vec3 dir = vec3(ray_directions[(i*3)], ray_directions[(i*3)+1], ray_directions[(i*3)+2]);

    // Create the ray and trace it
    Ray ray(vec4(position + (dir * 0.00001f), 1.f), vec4(dir, 1.f));
    ray.closest_intersection(scene);

    // Update position, normal, and discount factor based on intersection
    switch(ray.intersection.intersection_type){

        // TERMINAL STATE: R_(t+1) = Environment light power
        case NOTHING:
            ray_rewards[i] = 0.f;
            ray_discounts[i] = 0.f;

            if ( ray_states[i] == 0 ){
                ray_throughputs[(i*3)] = ray_throughputs[(i*3)] * ENVIRONMENT_LIGHT;
                ray_throughputs[(i*3)+1] = ray_throughputs[(i*3)+1] * ENVIRONMENT_LIGHT;
                ray_throughputs[(i*3)+2] = ray_throughputs[(i*3)+2] * ENVIRONMENT_LIGHT;
                ray_bounces[i] = (unsigned int)bounces;
            }
            ray_states[i] = 1;
            break;
        
        // TERMINAL STATE: R_(t+1) = Area light power
        case AREA_LIGHT:
            float diffuse_light_power = scene->area_lights[ray.intersection.index].luminance; 
            ray_rewards[i] = diffuse_light_power*200.f;
            ray_discounts[i] = 0.f;

            if ( ray_states[i] == 0 ){
                vec3 diffuse_p = scene->area_lights[ray.intersection.index].diffuse_p;
                ray_throughputs[(i*3)] = ray_throughputs[(i*3)] * diffuse_p.x;
                ray_throughputs[(i*3)+1] = ray_throughputs[(i*3)+1] * diffuse_p.y;
                ray_throughputs[(i*3)+2] = ray_throughputs[(i*3)+2] * diffuse_p.z;
                ray_bounces[i] = (unsigned int)bounces;
            }
            ray_states[i] = 1;
            break;

        // NON-TERMINAL STATE: R_(t+1) + \gamma * max_a Q(S_t+1, a) 
        // where  R_(t+1) = 0 for diffuse surfaces
        case SURFACE:
            vec3 new_loc = vec3(ray.intersection.position);
            prev_ray_locations[(i*3)  ] = ray_locations[(i*3)  ];
            prev_ray_locations[(i*3)+1] = ray_locations[(i*3)+1];
            prev_ray_locations[(i*3)+2] = ray_locations[(i*3)+2]; 
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
            if ( ray_states[i] == 0 ){
                ray_throughputs[(i*3)] = ray_throughputs[(i*3)] * (BRDF.x / (float)M_PI);
                ray_throughputs[(i*3)+1] = ray_throughputs[(i*3)+1] * (BRDF.y / (float)M_PI);
                ray_throughputs[(i*3)+2] = ray_throughputs[(i*3)+2] * (BRDF.z / (float)M_PI);
            }

            // Now update discount_factors with luminance
            ray_rewards[i] = 0.f;
            ray_discounts[i] = luminance;

            // Still a ray being to bounce, so not finished
            if ( ray_states[i] == 0 ){
                atomicExch(rays_finished, 0);
            }
            break;
    }
}
