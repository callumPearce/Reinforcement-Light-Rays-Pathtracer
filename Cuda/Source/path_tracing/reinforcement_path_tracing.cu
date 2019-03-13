#include "reinforcement_path_tracing.cuh"
//cuRand
#include <curand.h>
#include <curand_kernel.h>

__global__
void update_radiance_volume_distributions(RadianceMap* radiance_map){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    radiance_map->radiance_volumes[i].update_radiance_distribution();
}

__global__
void draw_reinforcement_path_tracing(vec3* device_buffer, curandState* d_rand_state, RadianceMap* radiance_map, Camera* camera, Scene* scene, int* device_path_lengths){
    
    // Populate the shared GPU/CPU screen buffer
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Path trace the ray to find the colour to paint the pixel
    device_buffer[x*(int)SCREEN_HEIGHT + y] = path_trace_reinforcement(d_rand_state, radiance_map, camera, x, y, scene, device_path_lengths);

}

__device__
vec3 path_trace_reinforcement(curandState* d_rand_state, RadianceMap* radiance_map, Camera* camera, int pixel_x, int pixel_y, Scene* scene, int* device_path_lengths){
    vec3 irradiance = vec3(0.f);
    int total_path_lengths = 0;
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++){

        // Trace the path of the ray
        int path_length;
        irradiance += path_trace_reinforcement_iterative(pixel_x, pixel_y, camera, d_rand_state, radiance_map, scene, path_length);
        total_path_lengths += path_length;
    }
    int avg_path_length = int(total_path_lengths/SAMPLES_PER_PIXEL);
    // printf("Path Length %d\n",avg_path_length);
    device_path_lengths[pixel_x*SCREEN_HEIGHT + pixel_y] = avg_path_length;
    irradiance /= (float)SAMPLES_PER_PIXEL;
    return irradiance;
}

__device__
vec3 path_trace_reinforcement_iterative(int pixel_x, int pixel_y, Camera* camera, curandState* d_rand_state, RadianceMap* radiance_map, Scene* scene, int& path_length){

    Ray ray = Ray::sample_ray_through_pixel(d_rand_state, *camera, pixel_x, pixel_y);

    vec3 throughput = vec3(1.f);
    
    RadianceVolume* current_radiance_volume;
    int current_sector_x = -1;
    int current_sector_y = -1;

    for (int i = 0; i < MAX_RAY_BOUNCES; i++){

        // Trace the path of the ray to find the closest intersection
        ray.closest_intersection(scene);

        // We cannot update Q on the first bounce as it is the camera position,
        // not a point in the scene. But we still need the closest radiance volume it intersects with
        if (i > 0){
            // Update Q
            // where x = ray.start, y = intersection.position
            // Check that a radiance volume has been found to update its sector
            if (current_radiance_volume && current_sector_x != -1 && current_sector_y != -1){
                current_radiance_volume = radiance_map->temporal_difference_update_radiance_volume_sector(current_radiance_volume, current_sector_x, current_sector_y, ray.intersection, scene);
                current_sector_x = -1;
                current_sector_y = -1;
            } 
        }
        // Get the radiance volume for the first iteration
        else{
            if (ray.intersection.intersection_type == SURFACE)
                current_radiance_volume = radiance_map->find_closest_radiance_volume_iterative(MAX_DIST, ray.intersection.position, ray.intersection.normal);
        }

        // Check what they ray intersected with...
        switch(ray.intersection.intersection_type){
            // Interescted with nothing, so no radiance
            case NOTHING:
                path_length = i+1;
                return throughput * vec3(ENVIRONMENT_LIGHT);
                break;
            
            // Intersected with light plane, so return its diffuse_p
            case AREA_LIGHT:
                path_length= i+1;
                return throughput * scene->area_lights[ray.intersection.index].diffuse_p;
                break;

            // Intersected with a surface (diffuse)
            case SURFACE:

                vec4 sampled_direction = vec4(0.f);
                radiance_map->importance_sample_ray_direction(d_rand_state, ray.intersection, current_sector_x, current_sector_y, pixel_x, pixel_y, sampled_direction, current_radiance_volume);

                vec3 BRDF = scene->surfaces[ray.intersection.index].material.diffuse_c / (float)M_PI;
                float cos_theta = dot(vec3(scene->surfaces[ray.intersection.index].normal), vec3(sampled_direction));
                float rho = (1.f / (2.f * (float)M_PI));

                throughput *= (BRDF * cos_theta) / rho;
                
                vec4 start = ray.intersection.position + sampled_direction * 0.00001f;
                start.w = 1.f;
                ray = Ray(start, sampled_direction);
                break;
            }
    }
    path_length = MAX_RAY_BOUNCES;
    return vec3(0);
}
