/*
    Entry point for the Raytracer
*/
#include <omp.h>
#include <assert.h>
#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "sdl_auxiliary.h"
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
#include "path_tracing.h"
#include "radiance_map.h"
#include "radiance_tree.h"

using namespace std;
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

    cout << "Render time: " << dt << "ms." << endl;

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

void Draw(screen* screen, Camera& camera, vector<AreaLightPlane *> light_planes, vector<Surface *> surfaces, RadianceMap& radiance_map){

    // Reset the SDL screen to black
    memset(screen->buffer, 0, screen->height*screen->width*sizeof(uint32_t));

    #pragma omp parallel for
    for (int x = 0; x < SCREEN_WIDTH; x++){
        for (int y = 0; y < SCREEN_HEIGHT; y++){

            // Change the ray's direction to work for the current pixel (pixel space -> Camera space)
            vec4 dir((x - SCREEN_WIDTH / 2) , (y - SCREEN_HEIGHT / 2) , FOCAL_LENGTH , 1);

            // Create a ray that we will change the direction for below
            Ray ray(camera.get_position(), dir);
            ray.rotate_ray(camera.get_yaw());

            // Initialise the closest intersection
            Intersection closest_intersection;

            // Path trace the ray to find the colour to paint the pixel
            // vec3 radiance = path_trace(false, ray, surfaces, light_planes, 0);
            vec3 radiance = path_trace_radiance_map(radiance_map, ray, surfaces, light_planes);
            PutPixelSDL(screen, x, y, radiance);
        }
    }
}

int main (int argc, char* argv[]) {

    omp_set_num_threads(6);
    
    // Initialise SDL screen
    screen *screen = InitializeSDL(SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE);

    // Load the shapes within the scene
    vector<Surface> surfaces_load;
    vector<AreaLightPlane> light_planes_load;
    get_monte_carlo_shapes(surfaces_load, light_planes_load);
    // load_scene("Models/simple_room.obj", surfaces_load);

    // Create the camera
    Camera camera = Camera(vec4(0, 0, -3, 1));

    // Convert all surfaces into a unified list of pointers to them
    vector<Surface *> surfaces;
    for (int i = 0 ; i < surfaces_load.size(); i++) {
        Surface * sptr (&surfaces_load[i]);
        surfaces.push_back(sptr);
    }

    // Convert all area lights into a unified list of pointers to them
    vector<AreaLightPlane* > light_planes;
    for (int i = 0 ; i < light_planes_load.size(); i++) {
        AreaLightPlane * sptr (&light_planes_load[i]);
        light_planes.push_back(sptr);
    }

    // Initialise the radiance map
    RadianceMap radiance_map = RadianceMap(surfaces);
    RadianceTree* radiance_tree_pointer = radiance_map.get_global_radiance_tree_pointer();

    // Clear the list of surfaces and add the surfaces for the radiance spheres to be rendered
    // radiance_map.build_radiance_map_shapes(surfaces_load);
    // surfaces.clear();
    // for (int i = 0 ; i < surfaces_load.size(); i++) {
    //     Surface * sptr (&surfaces_load[i]);
    //     surfaces.push_back(sptr);
    // }

    // Render
    while (NoQuitMessageSDL()){
        Update(camera);
        Draw(screen, camera, light_planes, surfaces, radiance_map);
        SDL_Renderframe(screen);
    }

    delete radiance_tree_pointer;

    KillSDL(screen);
    
    return 0;
}