#include "pre_trained_pathtracer.cuh"

inline bool file_exists (const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

// Constructor
__host__
PretrainedPathtracer::PretrainedPathtracer(
    unsigned int frames,
    int batch_size, 
    SDLScreen& screen, 
    Scene& scene,
    Camera& camera,
    int argc,
    char** argv
){

    //////////////////////////////////////////////////////////////
    /*                  Assign attributes                       */
    //////////////////////////////////////////////////////////////
    this->vertices_count = scene.vertices_count;
    this->batch_size = batch_size;
    this->num_batches = (SCREEN_HEIGHT*SCREEN_WIDTH + (batch_size -1))/batch_size;
    dim3 b_size(8,8);
    this->block_size = b_size;
    int blocks_x = (SCREEN_WIDTH + this->block_size.x - 1)/this->block_size.x;
    int blocks_y = (SCREEN_HEIGHT + this->block_size.y - 1)/this->block_size.y;
    dim3 n_bs(blocks_x, blocks_y);
    this->num_blocks = n_bs;

    //////////////////////////////////////////////////////////////
    /*                Initialise the DQN                        */
    //////////////////////////////////////////////////////////////
    auto dyparams = dynet::extract_dynet_params(argc, argv);
    dynet::initialize(dyparams);
    dynet::ParameterCollection model;
    this->dqn = DQNetwork();
    this->dqn.initialize(model, this->vertices_count, GRID_RESOLUTION*GRID_RESOLUTION);

    //////////////////////////////////////////////////////////////
    /*             Load in the Parameter Values                 */
    //////////////////////////////////////////////////////////////
    std::string fname = "/home/calst/Documents/year4/thesis/monte_carlo_raytracer/Radiance_Map_Data/archway_12_12.model";
    if (file_exists(fname)){
        dynet::TextFileLoader loader(fname);
        loader.populate(model);
    }
    else{
        printf("Failed to load model, terminating program.\n");
        return;
    }

    //////////////////////////////////////////////////////////////
    /*          Intialise Pixel value buffers                   */
    //////////////////////////////////////////////////////////////
    vec3* host_buffer = new vec3[ SCREEN_HEIGHT * SCREEN_WIDTH ];
    vec3* device_buffer;
    checkCudaErrors(cudaMalloc(&device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3)));

    //////////////////////////////////////////////////////////////
    /*               Initialise device buffers                  */
    //////////////////////////////////////////////////////////////
    float* ray_directions_device;   /* Direction to next shoot the ray (3D) */
    float* ray_locations_device;    /* Current intersection location of the ray (3D) */
    float* ray_normals_device;      /* Current intersected surfaces normal for the ray (3D) */
    unsigned int* ray_states_device;    /* Has the ray intersected with a light/nothing? */
    float* ray_throughputs_device;  /* RGB scalars representing current colour throughput of the ray (3D) */
    unsigned int* ray_bounces_device;      /* Number of time the ray has bounced */
    float* ray_vertices_device;     /* All vertices of the scene in a coordinate system relative to current ray position*/

    checkCudaErrors(cudaMalloc(&ray_directions_device, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_locations_device, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_normals_device, sizeof(float) * 3 *SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_states_device, sizeof(unsigned int) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_throughputs_device, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_bounces_device, sizeof(unsigned int) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_vertices_device, sizeof(float) * this->vertices_count * SCREEN_HEIGHT * SCREEN_WIDTH));

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
            d_rand_state,
            device_camera,
            device_scene,
            device_buffer,
            device_vertices,
            ray_locations_device,
            ray_normals_device,   
            ray_directions_device,
            ray_states_device,  
            ray_throughputs_device,
            ray_bounces_device,
            ray_vertices_device
        );

        std::cout << "Rendered " << i+1 << " frames." << std::endl;

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
    /*                      Save the image                      */
    //////////////////////////////////////////////////////////////
    screen.SDL_SaveImage("/home/calst/Documents/year4/thesis/monte_carlo_raytracer/Images/render.bmp");

    //////////////////////////////////////////////////////////////
    /*                      Free memory used                    */
    //////////////////////////////////////////////////////////////
    delete [] host_buffer;
    cudaFree(device_buffer);
    cudaFree(d_rand_state);
    cudaFree(ray_locations_device);
    cudaFree(ray_normals_device);
    cudaFree(ray_directions_device);
    cudaFree(ray_states_device);
    cudaFree(ray_throughputs_device);
    cudaFree(ray_bounces_device);
    cudaFree(ray_vertices_device);
    cudaFree(device_camera);
    cudaFree(device_surfaces);
    cudaFree(device_light_planes);
    cudaFree(device_vertices);
    cudaFree(device_scene);
    cudaFree(device_vertices);
}

// Render a frame to output
__host__
void PretrainedPathtracer::render_frame(
    curandState* d_rand_state,
    Camera* device_camera,
    Scene* device_scene,
    vec3* device_buffer,
    float* device_vertices,
    float* ray_locations_device,
    float* ray_normals_device,   
    float* ray_directions_device,
    unsigned int* ray_states_device,  
    float* ray_throughputs_device,
    unsigned int* ray_bounces_device,
    float* ray_vertices_device
){
    // Initialise the buffer to hold the total throughput
    vec3* total_throughputs;
    checkCudaErrors(cudaMalloc(&total_throughputs, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMemset(total_throughputs, 0.f, sizeof(vec3) * SCREEN_HEIGHT * SCREEN_WIDTH));

    // Sample through each pixel SAMPLES_PER_PIXEL times
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++){

        // Initialise the rays
        initialise_ray<<<this->num_blocks, this->block_size>>>(
            d_rand_state,
            device_camera,
            ray_locations_device,
            ray_directions_device,
            ray_states_device,
            ray_throughputs_device,
            ray_bounces_device
        );
        checkCudaErrors(cudaDeviceSynchronize());

        // Create bool to determine if all rays in the batch have collided with a light
        int rays_finished = 0; /* If not updated to false by trace_ray, end loop */
        int* device_rays_finished;
        checkCudaErrors(cudaMalloc(&device_rays_finished, sizeof(int)));
        checkCudaErrors(cudaMemset(device_rays_finished, 1, sizeof(int)));

        // Trace ray paths until all have intersected with a light/nothing
        unsigned int bounces = 0;
        while(rays_finished == 0 && bounces < MAX_RAY_BOUNCES){

            printf("Bounces: %d/%d\n",bounces+1, MAX_RAY_BOUNCES);

            std::chrono::time_point<std::chrono::high_resolution_clock> start;
            if (TIMING){
                // TIMER START: Sampling ray directions
                start = std::chrono::high_resolution_clock::now();
            }

            // Get vertices in coordinate system surrounding current location
            convert_vertices_to_point_coord_system<<<this->num_blocks, this->block_size>>>(
                ray_vertices_device, 
                ray_locations_device,
                device_vertices,
                this->vertices_count
            );
            cudaDeviceSynchronize();

            // DIRECTION UPDATE
            // Don't modify the direction of the initial ray from the camera
            if (bounces > 0){
                // For each ray, compute the Q-values and importance sample a direction over them
                for (int b = 0; b < this->num_batches; b++){
                    
                    // Compute the current batch size
                    int batch_start_idx = b*this->batch_size;
                    int current_batch_size = std::min(SCREEN_HEIGHT*SCREEN_WIDTH - batch_start_idx, this->batch_size);

                    // Initialise the computational graph
                    dynet::ComputationGraph graph;

                    // Formulate the expression with the state and the scenes vertices
                    dynet::Dim input_dim({(unsigned int)this->vertices_count},current_batch_size);
                    std::vector<float> input_vals(this->vertices_count*current_batch_size);
                    checkCudaErrors(cudaMemcpy(&(input_vals[0]), &(ray_vertices_device[b*this->batch_size*this->vertices_count]), sizeof(float) * this->vertices_count * current_batch_size, cudaMemcpyDeviceToHost));
                    dynet::Expression input = dynet::input(graph, input_dim, input_vals);                  

                    // Get the q-vals
                    dynet::Expression prediction = this->dqn.network_inference(graph, input, false);
                    std::vector<float> q_vals = dynet::as_vector( graph.forward(prediction));                 // Some q_vals are all zero
                    
                    // Copy q-values to device
                    // Initialise the Q-value storage on device
                    float* device_q_values;
                    checkCudaErrors(cudaMalloc(&device_q_values, sizeof(float) * q_vals.size()));
                    checkCudaErrors(cudaMemcpy(device_q_values, &(q_vals[0]), sizeof(float) * q_vals.size(), cudaMemcpyHostToDevice));

                    // Setup the deivce storage for the ray direction indices
                    unsigned int* ray_direction_indices;
                    checkCudaErrors(cudaMalloc(&ray_direction_indices, sizeof(unsigned int) * current_batch_size));

                    // Run cuda kernel to compute new ray directions
                    int threads = 32;
                    int blocks = (current_batch_size + (threads-1))/threads;
                    sample_batch_ray_directions_importance_sample<<<blocks, threads>>>(
                        d_rand_state,
                        device_q_values,
                        ray_directions_device,
                        ray_locations_device,
                        ray_normals_device,
                        ray_throughputs_device,
                        ray_states_device,
                        ray_direction_indices,
                        (b*this->batch_size)
                    );
                    cudaDeviceSynchronize();

                    // Free memory
                    cudaFree(ray_direction_indices);
                    cudaFree(device_q_values);
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
            // TRACE RAYS
            trace_ray<<<this->num_blocks, this->block_size>>>(
                device_scene,
                device_rays_finished,
                ray_locations_device, 
                ray_normals_device, 
                ray_directions_device,
                ray_states_device,  
                ray_throughputs_device,
                ray_bounces_device,
                bounces
            );
            cudaDeviceSynchronize();

            // Copy over value to check if all rays have intersected with a light
            checkCudaErrors(cudaMemcpy(&rays_finished, device_rays_finished, sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemset(device_rays_finished, 1, sizeof(int)));

            if (TIMING){
                // TIMER END: Tracing rays
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "Tracing Ray Time: " << elapsed.count() << "s" << std::endl;
            }

            // Increment the number of bounces
            bounces++;            
        }

        // Add computed throughput values to the running total
        update_total_throughput<<<this->num_blocks, this->block_size>>>(
            ray_throughputs_device,
            total_throughputs
        );
        cudaDeviceSynchronize();
        cudaFree(device_rays_finished);

        std::cout << "SPP: " << i+1 << std::endl;
    }
    // Update the device_buffer with the throughput
    update_device_buffer<<<this->num_blocks, this->block_size>>>(
        device_buffer,
        total_throughputs
    );
    cudaDeviceSynchronize();
    cudaFree(total_throughputs);

    // Calculate the average path length
    int* total_path_lengths_device;
    checkCudaErrors(cudaMalloc(&total_path_lengths_device, sizeof(int)));
    checkCudaErrors(cudaMemset(total_path_lengths_device, 0, sizeof(int)));
    sum_path_lengths<<<this->num_blocks, this->block_size>>>(
        total_path_lengths_device,
        ray_bounces_device
    );
    int total_path_lengths = 0;
    checkCudaErrors(cudaMemcpy(&total_path_lengths, total_path_lengths_device, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Avg Path Length: " << total_path_lengths/(SCREEN_HEIGHT*SCREEN_WIDTH) << std::endl;    
}

// Gets the initial direction to shoot a ray in
__global__
void initialise_ray(
    curandState* d_rand_state,
    Camera* device_camera, 
    float* ray_locations, 
    float* ray_directions,
    unsigned int* ray_states, 
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
    ray_directions[(i*3)    ] = r.direction.x;
    ray_directions[(i*3) + 1] = r.direction.y;
    ray_directions[(i*3) + 2] = r.direction.z;

    // Initialise ray_variables
    ray_states[i] = 0;
    ray_throughputs[(i*3)    ] = 1.f;
    ray_throughputs[(i*3) + 1] = 1.f;
    ray_throughputs[(i*3) + 2] = 1.f;
    ray_bounces[i] = MAX_RAY_BOUNCES;
}

// Trace a ray for all ray locations given in the angles specified within the scene
__global__
void trace_ray(
    Scene* scene,
    int* rays_finished,
    float* ray_locations, 
    float* ray_normals, 
    float* ray_directions,
    unsigned int* ray_states,  
    float* ray_throughputs,
    unsigned int* ray_bounces,
    int bounces
){
    
    // Ray index
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    int i = SCREEN_HEIGHT*x + y;

    // Do nothing if we have already intersected with the light
    if (ray_states[i] != 0){
        return;
    }

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
            ray_states[i] = 1;
            ray_throughputs[(i*3)] = ray_throughputs[(i*3)] * ENVIRONMENT_LIGHT;
            ray_throughputs[(i*3)+1] = ray_throughputs[(i*3)+1] * ENVIRONMENT_LIGHT;
            ray_throughputs[(i*3)+2] = ray_throughputs[(i*3)+2] * ENVIRONMENT_LIGHT;
            ray_bounces[i] = (unsigned int)bounces;
            break;
        
        // TERMINAL STATE: R_(t+1) = Area light power
        case AREA_LIGHT:
            ray_states[i] = 1;
            
            vec3 diffuse_p = scene->area_lights[ray.intersection.index].diffuse_p;
            ray_throughputs[(i*3)] = ray_throughputs[(i*3)] * diffuse_p.x;
            ray_throughputs[(i*3)+1] = ray_throughputs[(i*3)+1] * diffuse_p.y;
            ray_throughputs[(i*3)+2] = ray_throughputs[(i*3)+2] * diffuse_p.z;
            ray_bounces[i] = (unsigned int)bounces;
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

            vec3 BRDF = scene->surfaces[ray.intersection.index].material.diffuse_c / (float)M_PI;

            // discount_factors holds cos_theta currently, update rgb throughput first
            ray_throughputs[(i*3)] *= BRDF.x;
            ray_throughputs[(i*3)+1] *= BRDF.y;
            ray_throughputs[(i*3)+2] *= BRDF.z;

            // Still a ray being to bounce, so not finished
            atomicExch(rays_finished, 0);
            break;
    }
}