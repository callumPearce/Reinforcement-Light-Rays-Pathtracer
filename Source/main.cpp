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
#include "ray.h"
#include "printing.h"
#include "camera.h"
#include "light_sphere.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

void Update(Camera& camera, LightSphere& light_sphere){
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
    if (keystate[SDL_SCANCODE_A]) {
        light_sphere.translate_left(0.1);
    }
    if (keystate[SDL_SCANCODE_D]) {
        light_sphere.translate_right(0.1);
    }
    if (keystate[SDL_SCANCODE_Q]) {
        light_sphere.translate_up(0.1);
    }
    if (keystate[SDL_SCANCODE_E]) {
        light_sphere.translate_down(0.1);
    }
    if (keystate[SDL_SCANCODE_W]) {
        light_sphere.translate_forwards(0.1);
    }
    if (keystate[SDL_SCANCODE_S]) {
        light_sphere.translate_backwards(0.1);
    }
}

void Draw(screen* screen, Camera& camera, LightSphere& light_sphere, vector<Shape *> shapes){

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

            // Find the closest intersection and plot the colour of the shape
            if (ray.closest_intersection(shapes, closest_intersection)) {
                vec3 light_sphere_light = light_sphere.get_intersection_radiance(closest_intersection, shapes); 
                vec3 final_colour = light_sphere_light;
                PutPixelSDL(screen, x, y, final_colour);
            }
            else {
                PutPixelSDL(screen, x, y, vec3(0,0,0));
            }
        }
    }
}

int main (int argc, char* argv[]) {

    omp_set_num_threads(6);
    
    // Initialise SDL screen
    screen *screen = InitializeSDL(SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE);

    // Load the shapes within the scene
    vector<Triangle> triangles;
    get_cornell_shapes(triangles);

    // Convert all shapes into a unified list of pointers to them
    vector<Shape *> shapes;
    for (int i = 0 ; i < triangles.size(); i++) {
        Shape * sptr (&triangles[i]);
        shapes.push_back(sptr);
    }

    // Create the camera
    Camera camera = Camera(vec4(0, 0, -3, 1));

    // Create the light-sphere
    vec3 diffuse_p = 5.0f * vec3(1, 1, 0.9);
    vec3 ambient_p = 0.0f * vec3(1,1,1);
    float r = 0.05f;
    LightSphere light_sphere(vec4(0, -0.4, -0.9, 1.0), r, 1, diffuse_p, ambient_p);

    // Render
    while (NoQuitMessageSDL()){
        Update(camera, light_sphere);
        Draw(screen, camera, light_sphere, shapes);
        SDL_Renderframe(screen);
    }

    KillSDL(screen);
    
    return 0;
}