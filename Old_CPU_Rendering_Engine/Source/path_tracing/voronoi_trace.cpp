#include "voronoi_trace.h"
#include <iostream>
#include <omp.h>

void draw_voronoi_trace(SDLScreen screen, RadianceMap& radiance_map, Camera& camera, std::vector<AreaLightPlane *> light_planes, std::vector<Surface *> surfaces){
    // Reset the SDL screen to black
    memset(screen.buffer, 0, screen.height*screen.width*sizeof(uint32_t));

    // Updates the radiance map to contain voronoi colours within the scene
    radiance_map.set_voronoi_colours();

    #pragma omp parallel for
    for (int x = 0; x < SCREEN_WIDTH; x++){
        for (int y = 0; y < SCREEN_HEIGHT; y++){
            // Path trace the ray to find the colour to paint the pixel
            vec3 irradiance = voronoi_trace(camera, radiance_map, x, y, surfaces, light_planes);
            screen.PutPixelSDL(x, y, irradiance);
        }
    }
}

vec3 voronoi_trace(Camera& camera, RadianceMap& radiance_map, int pixel_x, int pixel_y, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes){
        // Generate the random point within a pixel for the ray to pass through
        float x = (float)pixel_x + ((float) rand() / (RAND_MAX));
        float y = (float)pixel_y + ((float) rand() / (RAND_MAX));

        // Set direction to pass through pixel (pixel space -> Camera space)
        vec4 dir((x - (float)SCREEN_WIDTH / 2.f) , (y - (float)SCREEN_HEIGHT / 2.f) , (float)FOCAL_LENGTH , 1);
        
        // Create a ray that we will change the direction for below
        Ray ray(camera.get_position(), dir);
        ray.rotate_ray(camera.get_yaw());

        // Trace the path of the ray to find the closest intersection
        Intersection closest_intersection;
        ray.closest_intersection(surfaces, light_planes, closest_intersection);

        // Get the voronoi colour of the intersection point
        return radiance_map.get_voronoi_colour(closest_intersection);
}