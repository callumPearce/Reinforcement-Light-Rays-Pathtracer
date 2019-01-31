/*
    Entry point for the Raytracer
*/
#include <assert.h>
#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "sdl_auxiliary.h"
#include <stdint.h>
#include <memory>
#include <omp.h>
#include <vector>

#include "image_settings.h"
#include "cornell_box_scene.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

int main (int argc, char* argv[]) {

    omp_set_num_threads(6);
    
    screen *screen = InitializeSDL(SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE);

    vector<Triangle> triangles;
    get_cornell_shapes(triangles);

    vector<Shape *> shapes;
    for (int i = 0 ; i < triangles.size(); i++) {
        Shape * sptr (&triangles[i]);
        shapes.push_back(sptr);
    }

    vec4 camera_pos = vec4(0, 0, -3, 1);

    while (NoQuitMessageSDL()){
        for (int x = 0; x < SCREEN_WIDTH; x++){
            for (int y = 0; y < SCREEN_HEIGHT; y++){

                // Change the ray's direction to work for the current pixel (pixel space -> Camera space)
                vec4 dir((x - SCREEN_WIDTH / 2) , (y - SCREEN_HEIGHT / 2) , FOCAL_LENGTH , 1);

                // Create a ray that we will change the direction for below
                Ray ray(camera_pos, dir);
                // ray.rotateRay(camera.getYaw());

                // Initialise the closest intersection
                Intersection closest_intersection;

                // Find the closest intersection and plot the colour of the shape
                if (ray.closest_intersection(shapes, closest_intersection)) {
                    vec3 colour = shapes[closest_intersection.index]->get_material().get_diffuse_c();

                    PutPixelSDL(screen, x, y, colour);
                }
                else {
                    PutPixelSDL(screen, x, y, vec3(0,0,0));
                }
            }
        }
        SDL_Renderframe(screen);
    }

    KillSDL(screen);
    
    return 0;
}