#include "voronoi_trace.cuh"
#include <iostream>

void draw_voronoi_trace(vec3* host_buffer, RadianceMap* radiance_map, Camera& camera, AreaLight* light_planes, Surface* surfaces, int light_plane_count, int surfaces_count){

    // Updates the radiance map to contain voronoi colours within the scene
    radiance_map->set_voronoi_colours();

    for (int x = 0; x < SCREEN_WIDTH; x++){
        for (int y = 0; y < SCREEN_HEIGHT; y++){
            // Path trace the ray to find the colour to paint the pixel
            host_buffer[ x*SCREEN_HEIGHT + y ] = voronoi_trace(camera, radiance_map, x, y, surfaces, light_planes, light_plane_count, surfaces_count);
        }
    }
}

vec3 voronoi_trace(Camera& camera, RadianceMap* radiance_map, int pixel_x, int pixel_y, Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count){
        // Generate the random point within a pixel for the ray to pass through
        float x = (float)pixel_x + ((float) rand() / (RAND_MAX));
        float y = (float)pixel_y + ((float) rand() / (RAND_MAX));

        // Set direction to pass through pixel (pixel space -> Camera space)
        vec4 dir((x - (float)SCREEN_WIDTH / 2.f) , (y - (float)SCREEN_HEIGHT / 2.f) , (float)FOCAL_LENGTH , 1);
        
        // Create a ray that we will change the direction for below
        Ray ray(camera.position, dir, true);
        ray.rotate_ray_host(camera.yaw);

        // Trace the path of the ray to find the closest intersection
        ray.closest_intersection_host(surfaces, light_planes, light_plane_count, surfaces_count);

        if (ray.intersection.intersection_type == SURFACE){
            // Get the voronoi colour of the intersection point
            return radiance_map->get_voronoi_colour(ray.intersection);
        }
        else{
            return vec3(1.f);
        }
}