#include "default_path_tracing.h"
#include <iostream>

// global means running on GPU, callable from CPU -> global functions are kernels
__global__
void draw_default_path_tracing(vec3* shared_buffer, Camera& camera, AreaLight* light_planes, Surface* surfaces, int light_plane_count, int surfaces_count){
    
    // Populate the shared GPU/CPU screen buffer
    // TODO: Calculate x, y using cuda values rather then for loop
    for (int x = 0; x < SCREEN_WIDTH; x++){
        for (int y = 0; y < SCREEN_HEIGHT; y++){
            // Path trace the ray to find the colour to paint the pixel
            shared_buffer[x*(int)SCREEN_HEIGHT + y] = path_trace(camera, x, y, surfaces, light_planes, light_plane_count, surfaces_count);
        }
    }

}

__device__
vec3 path_trace(Camera& camera, int pixel_x, int pixel_y, Surface* surfaces, AreaLight* light_planes, int light_plane_count, int surfaces_count){

    vec3 irradiance = vec3(0.f);
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++){
        
        // Generate the random point within a pixel for the ray to pass through
        float x = (float)pixel_x + ((float) rand() / (RAND_MAX));
        float y = (float)pixel_y + ((float) rand() / (RAND_MAX));

        // Set direction to pass through pixel (pixel space -> Camera space)
        vec4 dir((x - (float)SCREEN_WIDTH / 2.f) , (y - (float)SCREEN_HEIGHT / 2.f) , (float)FOCAL_LENGTH , 1);
        
        // Create a ray that we will change the direction for below
        Ray ray(camera.get_position(), dir);
        ray.rotate_ray(camera.get_yaw());

        // Trace the path of the ray
        irradiance += path_trace_recursive(ray, surfaces, light_planes, 0, light_plane_count, surfaces_count);
    }
    irradiance /= (float)SAMPLES_PER_PIXEL;
    return irradiance;
}


// Traces the path of a ray following monte carlo path tracer in order to estimate the radiance for a ray shot
// from its angle and starting position
__device__
vec3 path_trace_recursive(Ray ray, Surface* surfaces, AreaLight* light_planes, int bounces, int light_plane_count, int surfaces_count){
    
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
                return indirect_irradiance(closest_intersection, surfaces, light_planes, bounces, light_plane_count, surfaces_count);
            }
            break;
    }

    return vec3(0);
}

// For a given intersection point, return the radiance of the surface resulting
// from indirect illumination (i.e. other shapes in the scene) via the Monte Carlo Raytracing
__device__
vec3 indirect_irradiance(const Intersection& intersection, Surface* surfaces, AreaLight* light_planes, int bounces, int light_plane_count, int surfaces_count){

    float cos_theta;
    vec4 sampled_direction = sample_random_direction_around_intersection(intersection, cos_theta);
    
    // Create the new bounced ray
    vec4 start = intersection.position + (0.00001f * sampled_direction);
    start[3] = 1.f;
    Ray ray = Ray(start, sampled_direction);

    // 4) Get the radiance contribution for this ray and add to the sum
    vec3 radiance = path_trace_recursive(ray, surfaces, light_planes, bounces+1, light_plane_count, surfaces_count);

    // BRDF = reflectance / M_PI (equal from all angles for diffuse)
    // rho = 1/(2*M_PI) (probabiltiy of sampling a ray in the given direction)
    vec3 BRDF = surfaces[intersection.index].get_material().get_diffuse_c() / (float)M_PI;
    
    // Approximate the rendering equation
    vec3 irradiance = (radiance * BRDF * cos_theta) / (1.f / (2.f * (float)M_PI));

    return irradiance;
}
