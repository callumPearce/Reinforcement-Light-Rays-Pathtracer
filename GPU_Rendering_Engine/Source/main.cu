/*
    Entry point for the Raytracer
*/
#include <assert.h>
#include <glm/glm.hpp>
#include <SDL.h>
#include "sdl_screen.h"
#include <stdint.h>
#include <memory>
#include <vector>

#include "image_settings.h"
#include "radiance_volumes_settings.h"
#include "cornell_box_scene.cuh"
#include "ray.cuh"
#include "printing.h"
#include "camera.cuh"
#include "area_light.cuh"
#include "radiance_map.cuh"
#include "radiance_volume.cuh"
#include "scene.cuh"
#include "radiance_tree.cuh"

// Path Tracing Types
#include "default_path_tracing.cuh"
#include "reinforcement_path_tracing.cuh"
#include "voronoi_trace.cuh"
#include "neural_q_pathtracer.cuh"
#include "pre_trained_pathtracer.cuh"
#include "q_value_extractor.cuh"

// Cuda
#include "cuda_helpers.cuh"
#include <curand_kernel.h>
#include <curand.h>

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

bool Update(Camera& camera){
    static int t = SDL_GetTicks();
    /* Compute frame time */
    int t2 = SDL_GetTicks();
    float dt = float(t2-t);
    t = t2;

    printf("Render Time: %.3f ms.\n", dt);
  
    SDL_Event e;
    while(SDL_PollEvent(&e))
    {
        if (e.type == SDL_QUIT)
        {
            return false;
        }
        else
            if (e.type == SDL_KEYDOWN)
            {
                int key_code = e.key.keysym.sym;
                switch(key_code)
                {
                    case SDLK_UP:
                        /* Move camera forwards*/
                        // camera.move_forwards(0.01f);
                        camera.rotate_up(0.008f);
                    break;
                    case SDLK_DOWN:
                        /* Move camera backwards */
                        // camera.move_backwards(0.01f);
                        camera.rotate_down(0.008f);
                    break;
                    case SDLK_LEFT:
                        /* Move camera left */
                        camera.rotate_left(0.008f);
                    break;
                    case SDLK_RIGHT:
                        /* Move camera right */
                        camera.rotate_right(0.008f);
                    break;
                    case SDLK_ESCAPE:
                        /* Move camera quit */
                        return false;
                }
            }  
    }
    return true;
}

int main (int argc, char** argv) {

    // Initialise SDL screen
    SDLScreen screen = SDLScreen(SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE);
    
    // Reset the SDL screen to black
    memset(screen.buffer, 0, screen.height*screen.width*sizeof(uint32_t));

    // Create the camera
    // Door scene: vec4(0.f, 0.5f, -0.9f, 1.f)
    // Cornell Box: vec4(0.f,0.f, -3.f, 1.f);
    Camera camera = Camera(vec4(-1.f, -1.f, -0.4f, 1.f));
    // camera.rotate_right(3.14f);
    // camera.rotate_down(0.0f);

    // Initialise the scene
    Scene scene = Scene();
    // scene.load_cornell_box_scene();
    scene.load_custom_scene("../Models/complex_light_room.obj", true);
    scene.save_vertices_to_file();

    // CASE: Deep Reinforcement Learning
    if ( PATH_TRACING_METHOD == 3 ){
        NeuralQPathtracer(
            100, 
            4096,
            screen, 
            scene,
            camera,
            argc,
            argv
        );
    }
    // CASE: Trained network inferece
    else if( PATH_TRACING_METHOD == 4 ){
        PretrainedPathtracer(
            1, 
            4096,
            screen, 
            scene,
            camera,
            argc,
            argv
        );
    }
    // CASE: Save specified Q-values for pretrained network
    else if( PATH_TRACING_METHOD == 5){
        save_selected_radiance_volumes_vals_nn(
            "../Radiance_Map_Data/selected_radiance_volumes/",
            "../Radiance_Map_Data/deep_q_learning_12_12.model",
            scene,
            argc,
            argv
        );
    }
    else{

        /* Setup defautl CUDA memory */
        vec3 * device_buffer;
        Scene* device_scene;
        Surface* device_surfaces;
        AreaLight* device_light_planes;
        float* device_vertices;
        curandState * d_rand_state;
        vec3* host_buffer = new vec3[ SCREEN_HEIGHT * SCREEN_WIDTH ];
        Camera* device_camera;

        // Create the shared RGB screen buffer
        checkCudaErrors(cudaMalloc(&device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3)));
        
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

        // Create the random state array for random number generation
        checkCudaErrors(cudaMalloc(&d_rand_state, (float)SCREEN_HEIGHT * (float)SCREEN_WIDTH * sizeof(curandState)));

        // Initialise the space for the camera on the device
        checkCudaErrors(cudaMalloc(&device_camera, sizeof(Camera)));

        /* Render with specified rendering approach */
        //DEFAULT
        if(PATH_TRACING_METHOD == 0){

            // Get the block size and block count to compute over all pixels
            dim3 block_size(8, 8);
            int blocks_x = (SCREEN_WIDTH + block_size.x - 1)/block_size.x;
            int blocks_y = (SCREEN_HEIGHT + block_size.y - 1)/block_size.y;
            dim3 num_blocks(blocks_x, blocks_y);

            init_rand_state<<<num_blocks, block_size>>>(d_rand_state, SCREEN_WIDTH, SCREEN_HEIGHT);

            // Create the buffer to store the path lengths for each pixel
            int* host_path_lengths = new int[ SCREEN_HEIGHT*SCREEN_WIDTH ];
            int* device_path_lengths;
            checkCudaErrors(cudaMalloc(&device_path_lengths, sizeof(int)*SCREEN_HEIGHT*SCREEN_WIDTH ));
            checkCudaErrors(cudaMemset(device_path_lengths, 0, sizeof(int)*SCREEN_HEIGHT*SCREEN_WIDTH ));

            // RENDER LOOP
            while (Update(camera)){

                // Copy the camera to the device
                checkCudaErrors(cudaMemcpy(device_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

                draw_default_path_tracing<<<num_blocks, block_size>>>(
                    device_buffer, 
                    d_rand_state, 
                    device_camera, 
                    device_scene,
                    device_path_lengths
                );

                cudaDeviceSynchronize();

                // Copy the path length values and calculate the average path length
                checkCudaErrors(cudaMemcpy(host_path_lengths, device_path_lengths, sizeof(int) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyDeviceToHost));
                int total = 0;
                for (int i = 0; i < (SCREEN_HEIGHT * SCREEN_WIDTH); i++){
                    total += host_path_lengths[i];
                }
                float avg = total / (SCREEN_HEIGHT * SCREEN_WIDTH);
                printf("Average Path Length: %.3f\n", avg);

                // Copy the render back to the host
                checkCudaErrors(cudaMemcpy(host_buffer, device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3), cudaMemcpyDeviceToHost));

                // Put pixels in the SDL buffer, ready for rendering
                for (int x = 0; x < SCREEN_WIDTH; x++){
                    for (int y = 0; y < SCREEN_HEIGHT; y++){
                        screen.PutPixelSDL(x, y, host_buffer[x*(int)SCREEN_HEIGHT + y]);
                    }
                }

                cudaMemset(device_buffer, 0.f, sizeof(vec3)* SCREEN_HEIGHT * SCREEN_WIDTH);

                screen.SDL_Renderframe();
            }
        }
        // REINFORCEMENT
        else if(PATH_TRACING_METHOD == 1){

            // Get the block size and block count to compute over all pixels
            dim3 render_block_size(8, 8);
            int blocks_x = (SCREEN_WIDTH + render_block_size.x - 1)/render_block_size.x;
            int blocks_y = (SCREEN_HEIGHT + render_block_size.y - 1)/render_block_size.y;
            dim3 render_num_blocks(blocks_x, blocks_y);

            init_rand_state<<<render_num_blocks, render_block_size>>>(d_rand_state, SCREEN_WIDTH, SCREEN_HEIGHT);

            // Create the buffer to store the path lengths for each pixel
            int* host_path_lengths = new int[ SCREEN_HEIGHT*SCREEN_WIDTH ];
            int* device_path_lengths;
            checkCudaErrors(cudaMalloc(&device_path_lengths, sizeof(int)*SCREEN_HEIGHT*SCREEN_WIDTH ));
            checkCudaErrors(cudaMemset(device_path_lengths, 0, sizeof(int)*SCREEN_HEIGHT*SCREEN_WIDTH ));

            // Setup the radiance map
            std::vector<RadianceVolume> host_rvs;
            std::vector<RadianceTreeElement> radiance_array_v;
            RadianceMap* radiance_map = new RadianceMap(
                scene.surfaces,
                scene.surfaces_count,
                host_rvs,
                radiance_array_v
            );

            // Copy the radiance map onto the device
            RadianceMap* device_radiance_map;
            checkCudaErrors(cudaMalloc(&device_radiance_map, sizeof(RadianceMap)));
            checkCudaErrors(cudaMemcpy(device_radiance_map, radiance_map, sizeof(RadianceMap), cudaMemcpyHostToDevice));

            // Copy the list of radiance volumes
            int volumes = radiance_map->radiance_volumes_count;
            RadianceVolume* device_radiance_volumes;
            checkCudaErrors(cudaMalloc(&device_radiance_volumes, sizeof(RadianceVolume) * volumes));
            checkCudaErrors(cudaMemcpy(device_radiance_volumes, &host_rvs[0], host_rvs.size() * sizeof(RadianceVolume), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(&(device_radiance_map->radiance_volumes), &device_radiance_volumes, sizeof(RadianceMap*), cudaMemcpyHostToDevice));

            // Copy the radiance_array onto the device
            RadianceTreeElement* device_radiance_array;
            checkCudaErrors(cudaMalloc(&device_radiance_array, sizeof(RadianceTreeElement) * radiance_map->radiance_array_size));
            checkCudaErrors(cudaMemcpy(device_radiance_array, &radiance_array_v[0], sizeof(RadianceTreeElement) * radiance_map->radiance_array_size, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(&(device_radiance_map->radiance_array), &device_radiance_array, sizeof(RadianceTreeElement*), cudaMemcpyHostToDevice));
            
            // Get the number of blocks for updating the radiance volumes list
            int radiance_volume_block_size = 32;
            int radaince_volume_num_blocks = (volumes + radiance_volume_block_size - 1) / radiance_volume_block_size;
            
            // RENDER LOOP
            int frames = 0;
            while (Update(camera)){

                // Copy the camera to the device
                checkCudaErrors(cudaMemcpy(device_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

                draw_reinforcement_path_tracing<<<render_num_blocks, render_block_size>>>(
                    device_buffer,
                    d_rand_state,
                    device_radiance_map,
                    device_camera,
                    device_scene,
                    device_path_lengths
                );

                cudaDeviceSynchronize();

                // Copy the path length values and calculate the average path length
                checkCudaErrors(cudaMemcpy(host_path_lengths, device_path_lengths, sizeof(int) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyDeviceToHost));
                int total = 0;
                for (int i = 0; i < (SCREEN_HEIGHT * SCREEN_WIDTH); i++){
                    total += host_path_lengths[i];
                }
                float avg = total / (SCREEN_HEIGHT * SCREEN_WIDTH);
                printf("Average Path Length: %.3f\n", avg);

                update_radiance_volume_distributions<<<radaince_volume_num_blocks, radiance_volume_block_size>>>(
                    device_radiance_map
                );

                cudaDeviceSynchronize();

                // Copy the render back to the host
                checkCudaErrors(cudaMemcpy(host_buffer, device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3), cudaMemcpyDeviceToHost));
                

                // Put pixels in the SDL buffer, ready for rendering
                for (int x = 0; x < SCREEN_WIDTH; x++){
                    for (int y = 0; y < SCREEN_HEIGHT; y++){
                        screen.PutPixelSDL(x, y, host_buffer[x*(int)SCREEN_HEIGHT + y]);
                    }
                }

                cudaMemset(device_buffer, 0.f, sizeof(vec3)* SCREEN_HEIGHT * SCREEN_WIDTH);

                screen.SDL_Renderframe();
                screen.SDL_SaveImage("../Images/reinforcement_render/render.bmp");
                frames++;
            }

            // Save the radiance_map out to a file if chosen in options
            if (SAVE_RADIANCE_MAP){
                // Copy radiance map back to host
                checkCudaErrors(cudaMemcpy(radiance_map, device_radiance_map, sizeof(RadianceMap), cudaMemcpyDeviceToHost));

                // Copy radiance volumes back to host
                checkCudaErrors(cudaMemcpy(&host_rvs[0], device_radiance_volumes, host_rvs.size() * sizeof(RadianceVolume), cudaMemcpyDeviceToHost));

                // Set the radiance volumes pointer
                radiance_map->radiance_volumes = &host_rvs[0];

                // Conver the radiance cumulative distributions to regular distributions
                radiance_map->convert_radiance_volumes_distributions();
                
                // Save the radiance_maps q-values
                radiance_map->save_q_vals_to_file();
            }

            // Save the selected radiance volumes q-values to a file
            if (SAVE_REQUESTED_VOLUMES){
                // Copy radiance map back to host
                checkCudaErrors(cudaMemcpy(radiance_map, device_radiance_map, sizeof(RadianceMap), cudaMemcpyDeviceToHost));

                // Copy radiance volumes back to host
                checkCudaErrors(cudaMemcpy(&host_rvs[0], device_radiance_volumes, host_rvs.size() * sizeof(RadianceVolume), cudaMemcpyDeviceToHost));

                // Copy the radiance array back to the host
                checkCudaErrors(cudaMemcpy(&radiance_array_v[0], device_radiance_array, radiance_array_v.size() * sizeof(RadianceTreeElement), cudaMemcpyDeviceToHost));

                // Set the radiance volumes pointer
                radiance_map->radiance_volumes = &host_rvs[0];

                // Set the radiance array pointer
                radiance_map->radiance_array = &radiance_array_v[0];

                // Conver the radiance cumulative distributions to regular distributions
                radiance_map->convert_radiance_volumes_distributions();

                // Save the selected radiance volumes q-values to a file
                radiance_map->save_selected_radiance_volumes_vals("../Radiance_Map_Data/selected_radiance_volumes/");
            }
            
            // Delete radiance map variables
            cudaFree(device_radiance_volumes);
            cudaFree(device_radiance_map);
        }
        // VORONOI
        else if(PATH_TRACING_METHOD == 2){
            
            Update(camera);

            // Copy the camera to the device
            checkCudaErrors(cudaMemcpy(device_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

            // Get the block size and block count to compute over all pixels
            dim3 block_size(8, 8);
            int blocks_x = (SCREEN_WIDTH + block_size.x - 1)/block_size.x;
            int blocks_y = (SCREEN_HEIGHT + block_size.y - 1)/block_size.y;
            dim3 num_blocks(blocks_x, blocks_y);

            init_rand_state<<<num_blocks, block_size>>>(d_rand_state, SCREEN_WIDTH, SCREEN_HEIGHT);

            // Setup the radiance map
            std::vector<RadianceVolume> temp_rvs;
            std::vector<RadianceTreeElement> radiance_array_v;
            RadianceMap* radiance_map = new RadianceMap(
                scene.surfaces,
                scene.surfaces_count,
                temp_rvs,
                radiance_array_v
            );

            
            // Setup the colours of the voronoi plot
            radiance_map->set_voronoi_colours(temp_rvs);

            // Copy the radiance map onto the device
            RadianceMap* device_radiance_map;
            checkCudaErrors(cudaMalloc(&device_radiance_map, sizeof(RadianceMap)));
            checkCudaErrors(cudaMemcpy(device_radiance_map, radiance_map, sizeof(RadianceMap), cudaMemcpyHostToDevice));

            // Copy the list of radiance volumes
            int volumes = radiance_map->radiance_volumes_count;
            RadianceVolume* device_radiance_volumes;
            checkCudaErrors(cudaMalloc(&device_radiance_volumes, sizeof(RadianceVolume) * volumes));
            checkCudaErrors(cudaMemcpy(device_radiance_volumes, &temp_rvs[0], temp_rvs.size() * sizeof(RadianceVolume), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(&(device_radiance_map->radiance_volumes), &device_radiance_volumes, sizeof(RadianceMap*), cudaMemcpyHostToDevice));

            // Copy the radiance_array onto the device
            RadianceTreeElement* device_radiance_array;
            checkCudaErrors(cudaMalloc(&device_radiance_array, sizeof(RadianceTreeElement) * radiance_array_v.size()));
            checkCudaErrors(cudaMemcpy(device_radiance_array, &radiance_array_v[0], sizeof(RadianceTreeElement) * radiance_array_v.size(), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(&(device_radiance_map->radiance_array), &device_radiance_array, sizeof(RadianceTreeElement*), cudaMemcpyHostToDevice));

            // RENDER LOOP
            while (Update(camera)){

                // Copy the camera to the device
                checkCudaErrors(cudaMemcpy(device_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

                draw_voronoi_trace<<<num_blocks, block_size>>>(
                    device_buffer,
                    d_rand_state,
                    device_radiance_map,
                    device_camera,
                    device_scene
                );

                cudaDeviceSynchronize();

                // Copy the render back to the host
                checkCudaErrors(cudaMemcpy(host_buffer, device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3), cudaMemcpyDeviceToHost));

                // Put pixels in the SDL buffer, ready for rendering
                for (int x = 0; x < SCREEN_WIDTH; x++){
                    for (int y = 0; y < SCREEN_HEIGHT; y++){
                        screen.PutPixelSDL(x, y, host_buffer[x*(int)SCREEN_HEIGHT + y]);
                    }
                }

                cudaMemset(device_buffer, 0.f, sizeof(vec3)* SCREEN_HEIGHT * SCREEN_WIDTH);

                screen.SDL_Renderframe();
            }
            
            // Delete radiance map variables
            cudaFree(device_radiance_volumes);
            cudaFree(device_radiance_map);
        }

        /* Free memeory within CPU/GPU */
        delete [] host_buffer;
        delete [] scene.surfaces;
        delete [] scene.area_lights;
        cudaFree(device_buffer);
        cudaFree(device_surfaces);
        cudaFree(device_light_planes);
        cudaFree(device_vertices);
        cudaFree(d_rand_state);
        cudaFree(device_scene);
        cudaFree(device_camera);

        /*                  Rendering                       */
        screen.SDL_Renderframe();
        screen.SDL_SaveImage("../Images/render.bmp");
        screen.kill_screen();
    }
    
    return 0;
}
