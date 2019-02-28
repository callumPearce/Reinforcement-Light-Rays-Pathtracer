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
#include "cornell_box_scene.h"
#include "ray.h"
#include "printing.h"
#include "camera.h"
#include "area_light.h"

// Path Tracing Types
#include "default_path_tracing.cuh"

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

int main (int argc, char* argv[]) {
    
    // Initialise SDL screen
    SDLScreen screen = SDLScreen(SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE);

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

    // Default Path tracing algorithm
    if(PATH_TRACING_METHOD == 0){

        // Create the shared RGB screen buffer
        vec3 * shared_buffer;
        cudaMallocManaged(&shared_buffer, (float)SCREEN_HEIGHT*(float)SCREEN_WIDTH*sizeof(vec3));

        Update(camera);
        dim3 block_size(8, 8);
        int blocks_x = (SCREEN_WIDTH + block_size.x - 1)/block_size.x;
        int blocks_y = (SCREEN_HEIGHT + block_size.y - 1)/block_size.y;
        dim3 num_blocks(blocks_x, blocks_y);
        draw_default_path_tracing<<<num_blocks, block_size>>>(shared_buffer, camera, light_planes, surfaces, light_plane_count, surfaces_count);

        // Put pixels in the SDL buffer, ready for rendering
        for (int x = 0; x < SCREEN_WIDTH; x++){
            for (int y = 0; y < SCREEN_HEIGHT; y++){
                screen.PutPixelSDL(x, y, shared_buffer[x*(int)SCREEN_HEIGHT + y]);
            }
        }
    }

    /*                  Rendering                       */
    screen.SDL_Renderframe();
    screen.SDL_SaveImage("../Images/render.bmp");
    screen.kill_screen();
    
    return 0;
}