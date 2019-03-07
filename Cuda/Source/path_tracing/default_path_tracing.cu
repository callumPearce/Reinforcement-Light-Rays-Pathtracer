#include "default_path_tracing.cuh"
//cuRand
#include <curand.h>
#include <curand_kernel.h>

// global means running on GPU, callable from CPU -> global functions are kernels
__global__
void draw_default_path_tracing(vec3* device_buffer, curandState* d_rand_state, Camera camera, Scene* scene){

    // Populate the shared GPU/CPU screen buffer
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    device_buffer[x*(int)SCREEN_HEIGHT + y] = path_trace(d_rand_state, camera, x, y, scene);
}

__device__
vec3 path_trace(curandState* d_rand_state, Camera camera, int pixel_x, int pixel_y, Scene* scene){

    vec3 irradiance = vec3(0.f);
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++){
        
        Ray ray = Ray::sample_ray_through_pixel(d_rand_state, camera, pixel_x, pixel_y);

        // Trace the path of the ray
        irradiance += path_trace_iterative(d_rand_state, ray, scene);
    }
    irradiance /= (float)SAMPLES_PER_PIXEL;
    return irradiance;
}

__device__
vec3 path_trace_iterative(curandState* d_rand_state, Ray ray, Scene* scene){

    // Factor to multiply to output upon light intersection
    vec3 throughput = vec3(1.f);
    float rho = (1.f / (2.f * (float)M_PI));

    // Trace the path of the ray up to the maximum number of bounces
    for (int i = 0; i < MAX_RAY_BOUNCES; i++){
    
        ray.closest_intersection(scene);

        // Take the according action based on intersection type
        switch(ray.intersection.intersection_type){

            // Interescted with nothing, so no radiance
            case NOTHING:
                return vec3(0.f);
                break;
            
            // Intersected with light plane, so return its diffuse_p
            case AREA_LIGHT:
                return throughput * scene->area_lights[ray.intersection.index].diffuse_p;
                break;

            // Intersected with a surface (diffuse)
            case SURFACE:

                // Sample a direction to bounce the ray in
                float cos_theta;
                vec4 sampled_direction = sample_random_direction_around_intersection(d_rand_state, ray.intersection, cos_theta);
            
                // BRDF = reflectance / M_PI (equal from all angles for diffuse)
                // rho = 1/(2*M_PI) (probabiltiy of sampling a ray in the given direction)
                vec3 BRDF = scene->surfaces[ray.intersection.index].material.diffuse_c / (float)M_PI;
                
                // Approximate the rendering equation
                throughput = (throughput * BRDF * cos_theta) / rho;

                // Update rays direction
                vec4 start = ray.intersection.position + (0.00001f * sampled_direction);
                start[3] = 1.f;
                ray = Ray(start, sampled_direction);

                break;
        }
    }
    return vec3(0.f);
}