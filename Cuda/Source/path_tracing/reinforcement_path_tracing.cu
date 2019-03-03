#include "reinforcement_path_tracing.cuh"
//cuRand
#include <curand.h>
#include <curand_kernel.h>

__global__
void draw_reinforcement_path_tracing(vec3* device_buffer, curandState* d_rand_state, Camera camera, AreaLight* light_planes, Surface* surfaces, int light_plane_count, int surfaces_count){
    
    // Populate the shared GPU/CPU screen buffer
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Path trace the ray to find the colour to paint the pixel
    device_buffer[x*(int)SCREEN_HEIGHT + y] = path_trace_reinforcement(d_rand_state, camera, x, y, surfaces, light_planes, light_plane_count, surfaces_count);

}

__device__
vec3 path_trace_reinforcement(curandState* d_rand_state, Camera camera, int pixel_x, int pixel_y, Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count){
    vec3 irradiance = vec3(0.f);
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++){
        
        Ray ray = Ray::sample_ray_through_pixel(d_rand_state, camera, pixel_x, pixel_y);

        // Trace the path of the ray
        irradiance += path_trace_reinforcement_iterative(d_rand_state, ray, surfaces, light_planes, light_plane_count, surfaces_count);
    }
    irradiance /= (float)SAMPLES_PER_PIXEL;
    return irradiance;
}

__device__
vec3 path_trace_reinforcement_iterative(curandState* d_rand_state, Ray ray, Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count){

    vec3 throughput = vec3(1.f);
    
    // RadianceVolume* current_radiance_volume;
    // int current_sector_x = -1;
    // int current_sector_y = -1;

    for (int i = 0; i < MAX_RAY_BOUNCES; i++){

        // Trace the path of the ray to find the closest intersection
        ray.closest_intersection(surfaces, light_planes, light_plane_count, surfaces_count);

        // We cannot update Q on the first bounce as it is the camera position,
        // not a point in the scene
        // if (i > 0){
        //     // Update Q
        //     // where x = ray.start, y = intersection.position
        //     // Check that a radiance volume has been found to update its sector
        //     if (current_radiance_volume && current_sector_x != -1 && current_sector_y != -1){
        //         radiance_map.temporal_difference_update_radiance_volume_sector(current_radiance_volume, current_sector_x, current_sector_y, ray.intersection, surfaces, light_planes);
        //         current_sector_x = -1;
        //         current_sector_y = -1;
        //     } 
        // }

        // Check what they ray intersected with...
        switch(ray.intersection.intersection_type){
            // Interescted with nothing, so no radiance
            case NOTHING:
                return vec3(0);
                break;
            
            // Intersected with light plane, so return its diffuse_p
            case AREA_LIGHT:
                return throughput * light_planes[ray.intersection.index].diffuse_p;
                break;

            // Intersected with a surface (diffuse)
            case SURFACE:

                // vec4 sampled_direction = vec4(0.f);
                // current_radiance_volume = radiance_map.importance_sample_ray_direction(ray.intersection, current_sector_x, current_sector_y, sampled_direction);
                // TODO: Remove for reinforcment learning
                float cos_theta;
                vec4 sampled_direction = sample_random_direction_around_intersection(d_rand_state, ray.intersection, cos_theta);

                vec3 BRDF = surfaces[ray.intersection.index].material.diffuse_c / (float)M_PI;
                // float cos_theta = dot(vec3(surfaces[ray.intersection.index].normal), vec3(sampled_direction));
                float rho = (1.f / (2.f * (float)M_PI));

                throughput *= (BRDF * cos_theta) / rho;
                
                vec4 start = ray.intersection.position + sampled_direction * 0.00001f;
                start.w = 1.f;
                ray = Ray(start, sampled_direction);
                break;
            }
    }
    return vec3(0);
}