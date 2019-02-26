/*
    Entry point for the Raytracer
*/
#include <omp.h>
#include <assert.h>
#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "sdl_screen.h"
#include <stdint.h>
#include <memory>
#include <vector>

#include "image_settings.h"
#include "cornell_box_scene.h"
#include "monte_carlo_test_scene.h"
#include "object_importer.h"
#include "ray.h"
#include "printing.h"
#include "camera.h"
#include "area_light_plane.h"
#include "radiance_map.h"
#include "radiance_tree.h"

// Path Tracing Types
#include "default_path_tracing.h"
#include "importance_sampling_path_tracing.h"
#include "precompute_irradiance_path_tracing.h"
#include "reinforcement_path_tracing.h"
#include "voronoi_trace.h"

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

    std::cout << "Render time: " << dt << "ms." << std::endl;

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

    omp_set_num_threads(6);
    
    // Initialise SDL screen
    SDLScreen screen = SDLScreen(SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE);

    // Load the shapes within the scene
    std::vector<Surface> surfaces_load;
    std::vector<AreaLightPlane> light_planes_load;
    get_cornell_shapes(surfaces_load, light_planes_load);
    // load_scene("Models/simple_room.obj", surfaces_load);

    // Create the camera
    Camera camera = Camera(vec4(0, 0, -3, 1));

    // Convert all surfaces into a unified list of pointers to them
    std::vector<Surface *> surfaces;
    for (int i = 0 ; i < surfaces_load.size(); i++) {
        Surface * sptr (&surfaces_load[i]);
        surfaces.push_back(sptr);
    }

    /*                  Filling buffer                   */
    // Convert all area lights into a unified list of pointers to them
    std::vector<AreaLightPlane* > light_planes;
    for (int i = 0 ; i < light_planes_load.size(); i++) {
        AreaLightPlane * sptr (&light_planes_load[i]);
        light_planes.push_back(sptr);
    }

    // Reinforcment Learning path tracer
    if (PATH_TRACING_METHOD == 0){
        // Initialise the radiance map without pre-compute
        RadianceMap radiance_map = RadianceMap(false, surfaces, light_planes, surfaces_load);
        Update(camera);
        draw_reinforcement_path_tracing(screen, camera, radiance_map, light_planes, surfaces);
    }
    
    // Importance sampling with pre-computed radiance map values
    else if(PATH_TRACING_METHOD == 1){
        // Initialise the radiance map with pre-compute
        RadianceMap radiance_map = RadianceMap(true, surfaces, light_planes, surfaces_load);
        Update(camera);
        draw_importance_sampling_path_tracing(screen, camera, radiance_map, light_planes, surfaces);
    }

    // Irradiance estimation via average closest precomputed Radiance Volumes
    else if(PATH_TRACING_METHOD == 2){
        // Initialise the radiance map with pre-compute
        RadianceMap radiance_map = RadianceMap(true, surfaces, light_planes, surfaces_load);
        Update(camera);
        draw_radiance_map_path_tracing(screen, camera, radiance_map, light_planes, surfaces);
    }

    // Default Path tracing algorithm
    else if(PATH_TRACING_METHOD == 3){
        Update(camera);
        draw_default_path_tracing(screen, camera, light_planes, surfaces);
    }

    // Voronoi Ploat
    else if(PATH_TRACING_METHOD == 4){
        RadianceMap radiance_map = RadianceMap(false, surfaces, light_planes, surfaces_load);
        Update(camera);
        draw_voronoi_trace(screen, radiance_map, camera, light_planes, surfaces);
    }


    /*                  Rendering                       */
    screen.SDL_Renderframe();
    screen.SDL_SaveImage("Images/render.bmp");
    screen.kill_screen();

    // Clear the list of surfaces and add the surfaces for the radiance spheres to be rendered
    // radiance_map.build_radiance_map_shapes(surfaces_load);
    // surfaces.clear();
    // for (int i = 0 ; i < surfaces_load.size(); i++) {
    //     Surface * sptr (&surfaces_load[i]);
    //     surfaces.push_back(sptr);
    // }

    // Render
    // while (NoQuitMessageSDL()){
    // Draw(screen, camera, light_planes, surfaces, radiance_map);
    // screen.SDL_Renderframe();
    // screen.SDL_SaveImage("Images/render.bmp");
    // }
    
    return 0;
}