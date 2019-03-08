#include "voronoi_trace.cuh"
#include <iostream>

__global__
void draw_voronoi_trace(vec3* device_buffer, curandState* d_rand_state, RadianceMap* radiance_map, Camera* camera, Scene* scene){

    // Populate the shared GPU/CPU screen buffer
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    device_buffer[ x*SCREEN_HEIGHT + y ] = voronoi_trace(d_rand_state, camera, radiance_map, x, y, scene);
}

__device__
vec3 voronoi_trace(curandState* d_rand_state, Camera* camera, RadianceMap* radiance_map, int pixel_x, int pixel_y, Scene* scene){
        // Generate the random point within a pixel for the ray to pass through
        float x = (float)pixel_x + curand_uniform(&d_rand_state[pixel_x*(int)SCREEN_HEIGHT + pixel_y]);
        float y = (float)pixel_y + curand_uniform(&d_rand_state[pixel_x*(int)SCREEN_HEIGHT + pixel_y]);
        // Set direction to pass through pixel (pixel space -> Camera space)
        vec4 dir((x - (float)SCREEN_WIDTH / 2.f) , (y - (float)SCREEN_HEIGHT / 2.f) , (float)FOCAL_LENGTH , 1);

        // Create a ray that we will change the direction for below
        Ray ray(camera->position, dir);
        ray.rotate_ray(camera->yaw);

        // Trace the path of the ray to find the closest intersection
        ray.closest_intersection(scene);

        if (ray.intersection.intersection_type == SURFACE){
            // Get the voronoi colour of the intersection point
            return radiance_map->get_voronoi_colour(ray.intersection);
        }
        else{
            return vec3(1.f);
        }
}