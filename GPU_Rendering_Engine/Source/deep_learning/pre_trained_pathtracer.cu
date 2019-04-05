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
    this->dqn.initialize(model, GRID_RESOLUTION*GRID_RESOLUTION);

    //////////////////////////////////////////////////////////////
    /*             Load in the Parameter Values                 */
    //////////////////////////////////////////////////////////////
    std::string fname = "/home/calst/Documents/year4/thesis/monte_carlo_raytracer/Radiance_Map_Data/radiance_map_model.model";
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
    bool* ray_terminated_device;    /* Has the ray intersected with a light/nothing? */
    float* ray_throughputs_device;  /* RGB scalars representing current colour throughput of the ray (3D) */
    unsigned int* ray_bounces_device;      /* Number of time the ray has bounced */

    checkCudaErrors(cudaMalloc(&ray_directions_device, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_locations_device, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_normals_device, sizeof(float) * 3 *SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_terminated_device, sizeof(bool) * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_throughputs_device, sizeof(float) * 3 * SCREEN_HEIGHT * SCREEN_WIDTH));
    checkCudaErrors(cudaMalloc(&ray_bounces_device, sizeof(unsigned int) * SCREEN_HEIGHT * SCREEN_WIDTH));

    Camera* device_camera; /* Camera on the CUDA device */
    Surface* device_surfaces;
    AreaLight* device_light_planes;
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
            d_rand_state,
            device_camera,
            device_scene,
            device_buffer,
            ray_locations_device,
            ray_normals_device,   
            ray_directions_device,
            ray_terminated_device,  
            ray_throughputs_device,
            ray_bounces_device
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
    screen.SDL_SaveImage("/home/calst/Documents/year4/thesis/monte_carlo_raytracer/Images/render.bmp");
    screen.kill_screen();

    //////////////////////////////////////////////////////////////
    /*                      Free memory used                    */
    //////////////////////////////////////////////////////////////
    delete [] host_buffer;
    cudaFree(device_buffer);
    cudaFree(d_rand_state);
    cudaFree(ray_locations_device);
    cudaFree(ray_normals_device);
    cudaFree(ray_directions_device);
    cudaFree(ray_terminated_device);
    cudaFree(ray_throughputs_device);
    cudaFree(ray_bounces_device);
}

// Render a frame to output
__host__
void PretrainedPathtracer::render_frame(
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
            ray_terminated_device,
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

            printf("Bounces: %d\n",bounces);

            // DIRECTION UPDATE
            // Don't modify the direction of the initial ray from the camera
            if (bounces > 0){

                // sample_next_ray_directions_randomly<<<this->num_blocks, this->block_size>>>(
                //     d_rand_state,
                //     ray_normals_device, 
                //     ray_directions_device,
                //     ray_throughputs_device,
                //     ray_terminated_device
                // );

                // Copy over the ray locations from the device to the host for inference
                float* ray_locations_host = new float[ SCREEN_HEIGHT * SCREEN_WIDTH * 3 ];
                checkCudaErrors(cudaMemcpy(ray_locations_host, ray_locations_device, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH * 3, cudaMemcpyDeviceToHost));

                // For each ray, compute the Q-values and importance sample a direction over them
                for (int b = 0; b < this->num_batches; b++){

                    // Compute the current batch size
                    int batch_start_idx = b*this->batch_size;
                    int current_batch_size = std::min(SCREEN_HEIGHT*SCREEN_WIDTH - batch_start_idx, this->batch_size);

                    // Initialise the Q-value storage on device
                    float* device_q_values;
                    checkCudaErrors(cudaMalloc(&device_q_values, sizeof(float) * current_batch_size * GRID_RESOLUTION * GRID_RESOLUTION));

                    // Initialise the computational graph
                    dynet::ComputationGraph graph;

                    // Get the input expression 
                    std::vector<float> positions(current_batch_size*3);
                    memcpy(&(positions[0]), &(ray_locations_host[batch_start_idx*3]), sizeof(float)*current_batch_size*3);
                    dynet::Expression input = dynet::input(graph, dynet::Dim({3}, current_batch_size), positions);

                    // Get the q-vals
                    dynet::Expression prediction = this->dqn.network_inference(graph, input, false);
                    std::vector<float> q_vals = dynet::as_vector( graph.forward(prediction));                 // Some q_vals are all zero
                    
                    // Copy q-values to device
                    checkCudaErrors(cudaMemcpy(device_q_values, &(q_vals[0]), sizeof(float) * q_vals.size(), cudaMemcpyHostToDevice));
                
                    // Run cuda kernel to compute new ray directions
                    int threads = 16;
                    int blocks = int(current_batch_size/threads);
                    importance_sample_ray_directions<<<blocks, threads>>>(
                        d_rand_state,
                        device_q_values,
                        ray_normals_device,
                        ray_directions_device,
                        ray_locations_device,
                        ray_throughputs_device,
                        ray_terminated_device,
                        (b*this->batch_size)
                    );
                    cudaDeviceSynchronize();

                    // Free memory
                    cudaFree(device_q_values);
                }

                delete [] ray_locations_host;
            }

            // TRACE RAYS
            trace_ray<<<this->num_blocks, this->block_size>>>(
                device_scene,
                device_rays_finished,
                ray_locations_device, 
                ray_normals_device, 
                ray_directions_device,
                ray_terminated_device,  
                ray_throughputs_device,
                ray_bounces_device,
                bounces
            );
            cudaDeviceSynchronize();

            // Copy over value to check if all rays have intersected with a light
            checkCudaErrors(cudaMemcpy(&rays_finished, device_rays_finished, sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemset(device_rays_finished, 1, sizeof(int)));

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
    bool* ray_terminated, 
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
    ray_terminated[i] = false;
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
    bool* ray_terminated,  
    float* ray_throughputs,
    unsigned int* ray_bounces,
    int bounces
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
    vec3 position = vec3(ray_locations[(i*3)], ray_locations[(i*3)+1], ray_locations[(i*3)+2]);
    vec3 dir = vec3(ray_directions[(i*3)], ray_directions[(i*3)+1], ray_directions[(i*3)+2]);

    // Create the ray and trace it
    Ray ray(vec4(position + (dir * 0.00001f), 1.f), vec4(dir, 1.f));
    ray.closest_intersection(scene);

    // Update position, normal, and discount factor based on intersection
    switch(ray.intersection.intersection_type){

        // TERMINAL STATE: R_(t+1) = Environment light power
        case NOTHING:
            ray_terminated[i] = true;
            ray_throughputs[(i*3)] = ray_throughputs[(i*3)] * ENVIRONMENT_LIGHT*1;
            ray_throughputs[(i*3)+1] = ray_throughputs[(i*3)+1] * ENVIRONMENT_LIGHT*1;
            ray_throughputs[(i*3)+2] = ray_throughputs[(i*3)+2] * ENVIRONMENT_LIGHT*1;
            ray_bounces[i] = (unsigned int)bounces;
            break;
        
        // TERMINAL STATE: R_(t+1) = Area light power
        case AREA_LIGHT:
            ray_terminated[i] = true;
            
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

            vec3 BRDF = scene->surfaces[ray.intersection.index].material.diffuse_c;

            // discount_factors holds cos_theta currently, update rgb throughput first
            ray_throughputs[(i*3)] = ray_throughputs[(i*3)] * (BRDF.x / (float)M_PI);
            ray_throughputs[(i*3)+1] = ray_throughputs[(i*3)+1] * (BRDF.y / (float)M_PI);
            ray_throughputs[(i*3)+2] = ray_throughputs[(i*3)+2] * (BRDF.z / (float)M_PI);

            // Still a ray being to bounce, so not finished
            atomicExch(rays_finished, 0);
            break;
    }
}

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
){

    // Batch index
    int i =  blockIdx.x * blockDim.x + threadIdx.x;

    if ( batch_start_idx+i >= SCREEN_HEIGHT*SCREEN_WIDTH ) return;

    int q_start_idx = i * GRID_RESOLUTION * GRID_RESOLUTION;

     // Do nothing if we have already intersected with the light
     if (ray_terminated_device[batch_start_idx + i] == true){
        return;
    }

    // // Copy array onto local memory to speed-up processing
    // // Importance sample over Q_values
    float rv = curand_uniform(&d_rand_state[batch_start_idx + i]);
    int direction_idx = 0;
    float q_sum = 0.f;
    for (int n = 0; n < GRID_RESOLUTION*GRID_RESOLUTION; n++){ 

        q_sum += device_q_values[q_start_idx + n];
        // printf("%.5f\n",q_sum);
        if ( q_sum > rv ){
            direction_idx = n;
            break;
        }
    }

    // // Get max q-val
    // int max_q_index = 0;
    // int max_q = device_q_values[0];
    // // float summation = 0.f;
    // for (int n = 1; n < GRID_RESOLUTION*GRID_RESOLUTION; n++){
    //     float temp_q = device_q_values[q_start_idx + n];
    //     // summation += temp_q;
    //     if (temp_q > max_q){
    //         max_q_index = n;
    //         max_q = temp_q;
    //     }
    // }
    // int direction_idx = max_q_index;

    // if (max_q_index == GRID_RESOLUTION*GRID_RESOLUTION-1){
    //     printf("%.3f\n",device_q_values[q_start_idx + max_q_index]);
    // }

    // Convert the direction index sampled into an actual 3D direction
    sample_ray_for_grid_index(
        d_rand_state,
        direction_idx,
        ray_directions_device,
        ray_normals_device,
        ray_locations_device,
        ray_throughputs_device,
        (batch_start_idx + i)
    );
}
