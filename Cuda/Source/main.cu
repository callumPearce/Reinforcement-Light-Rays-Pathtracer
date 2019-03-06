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

void Update(Camera& camera){
    static int t = SDL_GetTicks();
    /* Compute frame time */
    int t2 = SDL_GetTicks();
    float dt = float(t2-t);
    t = t2;

    printf("Render Time: %.3f ms.\n", dt);

    /* Update variables*/
    const Uint8* keystate = SDL_GetKeyboardState(NULL);

    if (keystate[SDL_SCANCODE_UP]) {
        camera.move_forwards(0.1);
    }
    if (keystate[SDL_SCANCODE_DOWN]) {
        camera.move_backwards(0.1);
    }
    if (keystate[SDL_SCANCODE_LEFT]) {
        camera.rotate_left(0.1);
    }
    if (keystate[SDL_SCANCODE_RIGHT]) {
        camera.rotate_right(0.1);
    }
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

    // Load the shapes within the scene
    std::vector<Surface> surfaces_load;
    std::vector<AreaLight> area_lights_load;
    get_cornell_shapes(surfaces_load, area_lights_load);
    // load_scene("Models/simple_room.obj", surfaces_load);

    // Create the camera
    Camera camera = Camera(vec4(0, 0, -3, 1));

    // Convert the vector of surfaces into a fixed size array
    int surfaces_count = surfaces_load.size();
    Surface* surfaces = new Surface[ surfaces_load.size() ];
    for (int i = 0 ; i < surfaces_count; i++) {
        surfaces[i] = surfaces_load[i];
    }

    // Convert the vector of light planes into a fixed size array
    int light_plane_count = area_lights_load.size();
    AreaLight* light_planes = new AreaLight[ area_lights_load.size() ];
    for (int i = 0 ; i < light_plane_count; i++) {
        light_planes[i] = area_lights_load[i];
    }

    /* Setup defautl CUDA memory */
    vec3 * device_buffer;
    Surface* device_surfaces;
    AreaLight* device_light_planes;
    curandState * d_rand_state;
    vec3* host_buffer = new vec3[ SCREEN_HEIGHT * SCREEN_WIDTH ];

    // Create the shared RGB screen buffer
    checkCudaErrors(cudaMalloc(&device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3)));
    
    // Copy surfaces into device memory space
    checkCudaErrors(cudaMalloc(&device_surfaces, surfaces_count * sizeof(Surface)));
    checkCudaErrors(cudaMemcpy(device_surfaces, surfaces, surfaces_count * sizeof(Surface), cudaMemcpyHostToDevice));

    // Copy light planes into device memory space
    checkCudaErrors(cudaMalloc(&device_light_planes, light_plane_count * sizeof(AreaLight)));
    checkCudaErrors(cudaMemcpy(device_light_planes, light_planes, light_plane_count * sizeof(AreaLight), cudaMemcpyHostToDevice));

    // Create the random state array for random number generation
    checkCudaErrors(cudaMalloc(&d_rand_state, (float)SCREEN_HEIGHT * (float)SCREEN_WIDTH * sizeof(curandState)));

    /* Render with specified rendering approach */
    //DEFAULT
    if(PATH_TRACING_METHOD == 0){

        Update(camera);

        // Get the block size and block count to compute over all pixels
        dim3 block_size(8, 8);
        int blocks_x = (SCREEN_WIDTH + block_size.x - 1)/block_size.x;
        int blocks_y = (SCREEN_HEIGHT + block_size.y - 1)/block_size.y;
        dim3 num_blocks(blocks_x, blocks_y);

        init_rand_state<<<num_blocks, block_size>>>(d_rand_state, SCREEN_WIDTH, SCREEN_HEIGHT);

        draw_default_path_tracing<<<num_blocks, block_size>>>(
            device_buffer, 
            d_rand_state, 
            camera, 
            device_light_planes, 
            device_surfaces, 
            light_plane_count, 
            surfaces_count
        );

        // Copy the render back to the host
        checkCudaErrors(cudaMemcpy(host_buffer, device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3), cudaMemcpyDeviceToHost));

        // Put pixels in the SDL buffer, ready for rendering
        for (int x = 0; x < SCREEN_WIDTH; x++){
            for (int y = 0; y < SCREEN_HEIGHT; y++){
                screen.PutPixelSDL(x, y, host_buffer[x*(int)SCREEN_HEIGHT + y]);
            }
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
        RadianceMap* radiance_map = new RadianceMap(
            surfaces,
            surfaces_count,
            host_rvs
        );

        // Copy the radiance map onto the device
        RadianceMap* device_radiance_map;
        checkCudaErrors(cudaMalloc(&device_radiance_map, sizeof(RadianceMap)));
        checkCudaErrors(cudaMemcpy(device_radiance_map, radiance_map, sizeof(RadianceMap), cudaMemcpyHostToDevice));

        // Copy the list of radiance volumes
        int volumes = radiance_map->radiance_volumes_count;
        RadianceVolume* device_radiance_volumes;
        checkCudaErrors(cudaMalloc(&device_radiance_volumes, sizeof(RadianceVolume) * volumes));
        cudaMemcpy(device_radiance_volumes, &host_rvs[0], host_rvs.size() * sizeof(RadianceVolume), cudaMemcpyHostToDevice);

        // Copy the top level pointer value of RadianceMap.radiance_volumes 
        checkCudaErrors(cudaMemcpy(&(device_radiance_map->radiance_volumes), &device_radiance_volumes, sizeof(RadianceMap*), cudaMemcpyHostToDevice));

        // Get the number of blocks for updating the radiance volumes list
        int radiance_volume_block_size = 32;
        int radaince_volume_num_blocks = (volumes + radiance_volume_block_size - 1) / radiance_volume_block_size;
        
        // RENDER LOOP
        while (1){
            Update(camera);

            draw_reinforcement_path_tracing<<<render_num_blocks, render_block_size>>>(
                device_buffer,
                d_rand_state,
                device_radiance_map,
                camera,
                device_light_planes, 
                device_surfaces, 
                light_plane_count, 
                surfaces_count
            );

            cudaDeviceSynchronize();

            update_radiance_volume_distributions<<<radaince_volume_num_blocks, radiance_volume_block_size>>>(
                device_radiance_map
            );

            cudaDeviceSynchronize();

            // Copy the render back to the host
            checkCudaErrors(cudaMemcpy(host_buffer, device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3), cudaMemcpyDeviceToHost));
            
            cudaDeviceSynchronize();

            // Put pixels in the SDL buffer, ready for rendering
            for (int x = 0; x < SCREEN_WIDTH; x++){
                for (int y = 0; y < SCREEN_HEIGHT; y++){
                    screen.PutPixelSDL(x, y, host_buffer[x*(int)SCREEN_HEIGHT + y]);
                }
            }

            cudaMemset(device_buffer, 0.f, sizeof(vec3)* SCREEN_HEIGHT * SCREEN_WIDTH);

            screen.SDL_Renderframe();
            // screen.SDL_SaveImage("../Images/render.bmp");
        }
        
        // Delete radiance map variables
        cudaFree(device_radiance_volumes);
        cudaFree(device_radiance_map);
    }
    // VORONOI
    else if(PATH_TRACING_METHOD == 2){
        
        Update(camera);

        // Get the block size and block count to compute over all pixels
        dim3 block_size(8, 8);
        int blocks_x = (SCREEN_WIDTH + block_size.x - 1)/block_size.x;
        int blocks_y = (SCREEN_HEIGHT + block_size.y - 1)/block_size.y;
        dim3 num_blocks(blocks_x, blocks_y);

        init_rand_state<<<num_blocks, block_size>>>(d_rand_state, SCREEN_WIDTH, SCREEN_HEIGHT);

        // Setup the radiance map
        std::vector<RadianceVolume> temp_rvs;
        RadianceMap* radiance_map = new RadianceMap(
            surfaces,
            surfaces_count,
            temp_rvs
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
        cudaMemcpy(device_radiance_volumes, &temp_rvs[0], temp_rvs.size() * sizeof(RadianceVolume), cudaMemcpyHostToDevice);

        // Copy the top level pointer value of RadianceMap.radiance_volumes 
        checkCudaErrors(cudaMemcpy(&(device_radiance_map->radiance_volumes), &device_radiance_volumes, sizeof(RadianceMap*), cudaMemcpyHostToDevice));

        draw_voronoi_trace<<<num_blocks, block_size>>>(
            device_buffer,
            d_rand_state,
            device_radiance_map,
            camera,
            device_light_planes, 
            device_surfaces, 
            light_plane_count, 
            surfaces_count
        );

        // Copy the render back to the host
        checkCudaErrors(cudaMemcpy(host_buffer, device_buffer, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(vec3), cudaMemcpyDeviceToHost));

        // Put pixels in the SDL buffer, ready for rendering
        for (int x = 0; x < SCREEN_WIDTH; x++){
            for (int y = 0; y < SCREEN_HEIGHT; y++){
                screen.PutPixelSDL(x, y, host_buffer[x*(int)SCREEN_HEIGHT + y]);
            }
        }
        
        // Delete radiance map variables
        cudaFree(device_radiance_volumes);
        cudaFree(device_radiance_map);
    }

    /* Free memeory within CPU/GPU */
    delete [] host_buffer;
    delete [] surfaces;
    delete [] light_planes;
    cudaFree(device_buffer);
    cudaFree(device_surfaces);
    cudaFree(device_light_planes);
    cudaFree(d_rand_state);

    /*                  Rendering                       */
    screen.SDL_Renderframe();
    screen.SDL_SaveImage("../Images/render.bmp");
    screen.kill_screen();
    
    return 0;
}