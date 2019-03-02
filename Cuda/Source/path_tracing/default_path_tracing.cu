#include "default_path_tracing.cuh"
#include <iostream>
//cuRand
#include <curand.h>
#include <curand_kernel.h>

// global means running on GPU, callable from CPU -> global functions are kernels
__global__
void draw_default_path_tracing(vec3* device_buffer, curandState* d_rand_state, Camera camera, AreaLight* light_planes, Surface* surfaces, int light_plane_count, int surfaces_count){

    // Populate the shared GPU/CPU screen buffer
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    device_buffer[x*(int)SCREEN_HEIGHT + y] = path_trace(d_rand_state, camera, x, y, surfaces, light_planes, light_plane_count, surfaces_count);
}

__device__
vec3 path_trace(curandState* d_rand_state, Camera camera, int pixel_x, int pixel_y, Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count){

    vec3 irradiance = vec3(0.f);
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++){
        
        // Generate the random point within a pixel for the ray to pass through
        float x = (float)pixel_x + curand_uniform(&d_rand_state[pixel_x*SCREEN_HEIGHT + pixel_y]);
        float y = (float)pixel_y + curand_uniform(&d_rand_state[pixel_x*SCREEN_HEIGHT + pixel_y]);

        // Set direction to pass through pixel (pixel space -> Camera space)
        vec4 dir((x - (float)SCREEN_WIDTH / 2.f) , (y - (float)SCREEN_HEIGHT / 2.f) , (float)FOCAL_LENGTH , 1);
        
        // Create a ray that we will change the direction for below
        Ray ray(camera.position, dir);
        ray.rotate_ray(camera.yaw);

        // Trace the path of the ray
        irradiance += path_trace_iterative(d_rand_state, ray, surfaces, light_planes, light_plane_count, surfaces_count);
    }
    irradiance /= (float)SAMPLES_PER_PIXEL;
    return irradiance;
}

__device__
vec3 path_trace_iterative(curandState* d_rand_state, Ray ray, Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count){

    // Factor to multiply to output upon light intersection
    vec3 throughput = vec3(1.f);
    float rho = (1.f / (2.f * (float)M_PI));

    // Trace the path of the ray up to the maximum number of bounces
    for (int i = 0; i < MAX_RAY_BOUNCES; i++){
    
        ray.closest_intersection(surfaces, light_planes, light_plane_count, surfaces_count);

        // Take the according action based on intersection type
        switch(ray.intersection.intersection_type){

            // Interescted with nothing, so no radiance
            case NOTHING:
                return vec3(0.f);
                break;
            
            // Intersected with light plane, so return its diffuse_p
            case AREA_LIGHT:
                return throughput * light_planes[ray.intersection.index].diffuse_p;
                break;

            // Intersected with a surface (diffuse)
            case SURFACE:

                // Sample a direction to bounce the ray in
                float cos_theta;
                vec4 sampled_direction = sample_random_direction_around_intersection(d_rand_state, ray.intersection, cos_theta);
            
                // BRDF = reflectance / M_PI (equal from all angles for diffuse)
                // rho = 1/(2*M_PI) (probabiltiy of sampling a ray in the given direction)
                vec3 BRDF = surfaces[ray.intersection.index].material.diffuse_c / (float)M_PI;
                
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

// Traces the path of a ray following monte carlo path tracer in order to estimate the radiance for a ray shot
// from its angle and starting position
__device__
vec3 path_trace_recursive(curandState* d_rand_state, Ray ray, Surface* surfaces, AreaLight* light_planes, int bounces, int light_plane_count, int surfaces_count){

    // printf("%d , %d\n", light_plane_count, surfaces_count);
    
    // Trace the path of the ray to find the closest intersection
    ray.closest_intersection(surfaces, light_planes, light_plane_count, surfaces_count);

    // Take the according action based on intersection type
    switch(ray.intersection.intersection_type){

        // Interescted with nothing, so no radiance
        case NOTHING:
            return vec3(0);
            break;
        
        // Intersected with light plane, so return its diffuse_p
        case AREA_LIGHT:
            return light_planes[ray.intersection.index].diffuse_p;
            break;

        // Intersected with a surface (diffuse)
        case SURFACE:
            if (bounces == MAX_RAY_BOUNCES){
                return vec3(0);
            } else{
                return indirect_irradiance(d_rand_state, ray, surfaces, light_planes, bounces, light_plane_count, surfaces_count);
            }
            break;
    }

    return vec3(0);
}

// For a given intersection point, return the radiance of the surface resulting
// from indirect illumination (i.e. other shapes in the scene) via the Monte Carlo Raytracing
__device__
vec3 indirect_irradiance(curandState* d_rand_state, const Ray incident_ray, Surface* surfaces, AreaLight* light_planes, int bounces, int light_plane_count, int surfaces_count){

    float cos_theta;
    vec4 sampled_direction = sample_random_direction_around_intersection(d_rand_state, incident_ray.intersection, cos_theta);
    
    // Create the new bounced ray
    vec4 start = incident_ray.intersection.position + (0.00001f * sampled_direction);
    start[3] = 1.f;
    Ray ray = Ray(start, sampled_direction);

    // 4) Get the radiance contribution for this ray and add to the sum
    vec3 radiance = path_trace_recursive(d_rand_state, ray, surfaces, light_planes, bounces+1, light_plane_count, surfaces_count);

    // BRDF = reflectance / M_PI (equal from all angles for diffuse)
    // rho = 1/(2*M_PI) (probabiltiy of sampling a ray in the given direction)
    vec3 BRDF = surfaces[incident_ray.intersection.index].material.diffuse_c / (float)M_PI;
    
    // Approximate the rendering equation
    vec3 irradiance = (radiance * BRDF * cos_theta) / (1.f / (2.f * (float)M_PI));

    return irradiance;
}
