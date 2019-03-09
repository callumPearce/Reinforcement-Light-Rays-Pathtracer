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
#include "scene.cuh"
#include "radiance_tree.cuh"

// Path Tracing Types
#include "default_path_tracing.cuh"
#include "reinforcement_path_tracing.cuh"
#include "voronoi_trace.cuh"

//cuRand
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
                        camera.move_forwards(0.01f);
                    break;
                    case SDLK_DOWN:
                        /* Move camera backwards */
                        camera.move_backwards(0.01f);
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

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void init_rand_state(curandState* d_rand_state, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if((x >= width) || (y >= height)) return;
    int pixel_index = x*height + y;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &d_rand_state[pixel_index]);
 }

int main (int argc, char* argv[]) {

    // Initialise SDL screen
    SDLScreen screen = SDLScreen(SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE);
    
    // Reset the SDL screen to black
    memset(screen.buffer, 0, screen.height*screen.width*sizeof(uint32_t));

    // Create the camera
    Camera camera = Camera(vec4(0, 0, -3, 1));

    // Initialise the scene
    Scene scene = Scene();
    scene.load_cornell_box_scene();

    /* Setup defautl CUDA memory */
    vec3 * device_buffer;
    Scene* device_scene;
    Surface* device_surfaces;
    AreaLight* device_light_planes;
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

    // Copy the scene structure into the device and its corresponding pointers to Surfaces and Area Lights
    checkCudaErrors(cudaMalloc(&device_scene, sizeof(Scene)));
    checkCudaErrors(cudaMemcpy(device_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(device_scene->surfaces), &device_surfaces, sizeof(Surface*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(device_scene->area_lights), &device_light_planes, sizeof(AreaLight*), cudaMemcpyHostToDevice));

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

        // RENDER LOOP
        while (Update(camera)){

            // Copy the camera to the device
            checkCudaErrors(cudaMemcpy(device_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

            draw_default_path_tracing<<<num_blocks, block_size>>>(
                device_buffer, 
                d_rand_state, 
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
    }
    // REINFORCEMENT
    else if(PATH_TRACING_METHOD == 1){

        // Get the block size and block count to compute over all pixels
        dim3 render_block_size(8, 8);
        int blocks_x = (SCREEN_WIDTH + render_block_size.x - 1)/render_block_size.x;
        int blocks_y = (SCREEN_HEIGHT + render_block_size.y - 1)/render_block_size.y;
        dim3 render_num_blocks(blocks_x, blocks_y);

        init_rand_state<<<render_num_blocks, render_block_size>>>(d_rand_state, SCREEN_WIDTH, SCREEN_HEIGHT);

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
        while (Update(camera)){

            // Copy the camera to the device
            checkCudaErrors(cudaMemcpy(device_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

            draw_reinforcement_path_tracing<<<render_num_blocks, render_block_size>>>(
                device_buffer,
                d_rand_state,
                device_radiance_map,
                device_camera,
                device_scene
            );

            cudaDeviceSynchronize();

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

        // printf("%d, %d\n", radiance_array_v[20].left_idx, radiance_array_v[20].right_idx);
        
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
    cudaFree(d_rand_state);
    cudaFree(device_scene);
    cudaFree(device_camera);

    /*                  Rendering                       */
    screen.SDL_Renderframe();
    screen.SDL_SaveImage("../Images/render.bmp");
    screen.kill_screen();
    
    return 0;
}