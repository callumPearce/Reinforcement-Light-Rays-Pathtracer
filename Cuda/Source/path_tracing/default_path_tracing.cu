#include "default_path_tracing.cuh"
#include <iostream>
//cuRand
#include <curand.h>
#include <curand_kernel.h>

// global means running on GPU, callable from CPU -> global functions are kernels
__global__
void draw_default_path_tracing(vec3* device_buffer, curandState* d_rand_state, Camera& camera, AreaLight* light_planes, Surface* surfaces, int light_plane_count, int surfaces_count){
    
    printf("hello\n");

    // Populate the shared GPU/CPU screen buffer
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    printf("(%d,%d)\n",x,y);

    if (x < SCREEN_WIDTH && y < SCREEN_HEIGHT){
        device_buffer[x*(int)SCREEN_HEIGHT + y] = path_trace(d_rand_state, camera, x, y, surfaces, light_planes, light_plane_count, surfaces_count);
        // vec3 buff = shared_buffer[x*(int)SCREEN_HEIGHT + y];
        // printf("(%.3f,%.3f,%.3f)\n", buff.x, buff.y, buff.z);
    }
}

__device__
vec3 path_trace(curandState* d_rand_state, Camera& camera, int pixel_x, int pixel_y, Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count){

    vec3 irradiance = vec3(0.f);
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++){
        
        // Generate the random point within a pixel for the ray to pass through
        curandState x_rand_state = d_rand_state[pixel_x*(int)SCREEN_HEIGHT + pixel_y];
        curandState y_rand_state = d_rand_state[pixel_x*(int)SCREEN_HEIGHT + pixel_y];
        float x = (float)pixel_x + curand_uniform(&x_rand_state);
        float y = (float)pixel_y + curand_uniform(&y_rand_state);

        // Set direction to pass through pixel (pixel space -> Camera space)
        vec4 dir((x - (float)SCREEN_WIDTH / 2.f) , (y - (float)SCREEN_HEIGHT / 2.f) , (float)FOCAL_LENGTH , 1);
        
        // Create a ray that we will change the direction for below
        Ray ray(camera.get_position(), dir);
        ray.rotate_ray(camera.get_yaw());

        // Trace the path of the ray
        irradiance += path_trace_recursive(d_rand_state, ray, surfaces, light_planes, 0, light_plane_count, surfaces_count);
    }
    irradiance /= (float)SAMPLES_PER_PIXEL;
    return irradiance;
}


// Traces the path of a ray following monte carlo path tracer in order to estimate the radiance for a ray shot
// from its angle and starting position
__device__
vec3 path_trace_recursive(curandState* d_rand_state, Ray ray, Surface* surfaces, AreaLight* light_planes, int bounces, int light_plane_count, int surfaces_count){
    
    // Trace the path of the ray to find the closest intersection
    Intersection closest_intersection;
    ray.closest_intersection(surfaces, light_planes, closest_intersection, light_plane_count, surfaces_count);

    // Take the according action based on intersection type
    switch(closest_intersection.intersection_type){

        // Interescted with nothing, so no radiance
        case NOTHING:
            return vec3(0);
            break;
        
        // Intersected with light plane, so return its diffuse_p
        case AREA_LIGHT:
            return light_planes[closest_intersection.index].get_diffuse_p();
            break;

        // Intersected with a surface (diffuse)
        case SURFACE:
            if (bounces == MAX_RAY_BOUNCES){
                return vec3(0);
            } else{
                return indirect_irradiance(d_rand_state, closest_intersection, surfaces, light_planes, bounces, light_plane_count, surfaces_count);
            }
            break;
    }

    return vec3(0);
}

// For a given intersection point, return the radiance of the surface resulting
// from indirect illumination (i.e. other shapes in the scene) via the Monte Carlo Raytracing
__device__
vec3 indirect_irradiance(curandState* d_rand_state, const Intersection& intersection, Surface* surfaces, AreaLight* light_planes, int bounces, int light_plane_count, int surfaces_count){

    float cos_theta;
    vec4 sampled_direction = sample_random_direction_around_intersection(d_rand_state, intersection, cos_theta);
    
    // Create the new bounced ray
    vec4 start = intersection.position + (0.00001f * sampled_direction);
    start[3] = 1.f;
    Ray ray = Ray(start, sampled_direction);

    // 4) Get the radiance contribution for this ray and add to the sum
    vec3 radiance = path_trace_recursive(d_rand_state, ray, surfaces, light_planes, bounces+1, light_plane_count, surfaces_count);

    // BRDF = reflectance / M_PI (equal from all angles for diffuse)
    // rho = 1/(2*M_PI) (probabiltiy of sampling a ray in the given direction)
    vec3 BRDF = surfaces[intersection.index].get_material().get_diffuse_c() / (float)M_PI;
    
    // Approximate the rendering equation
    vec3 irradiance = (radiance * BRDF * cos_theta) / (1.f / (2.f * (float)M_PI));

    return irradiance;
}
