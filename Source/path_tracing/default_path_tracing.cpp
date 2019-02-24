#include "default_path_tracing.h"
#include <iostream>
#include "printing.h"

// Traces the path of a ray following monte carlo path tracer in order to estimate the radiance for a ray shot
// from its angle and starting position
vec3 path_trace(Ray ray, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces){
    
    // Trace the path of the ray to find the closest intersection
    Intersection closest_intersection;
    ray.closest_intersection(surfaces, light_planes, closest_intersection);

    // Take the according action based on intersection type
    switch(closest_intersection.intersection_type){

        // Interescted with nothing, so no radiance
        case NOTHING:
            return vec3(0);
            break;
        
        // Intersected with light plane, so return its diffuse_p
        case AREA_LIGHT_PLANE:
            return light_planes[closest_intersection.index]->get_diffuse_p();
            break;

        // Intersected with a surface (diffuse)
        case SURFACE:
            if (bounces == MAX_RAY_BOUNCES){
                return vec3(0);
            } else{
                return indirect_irradiance(closest_intersection, surfaces, light_planes, bounces);
            }
            break;
    }

    return vec3(0);
}

// For a given intersection point, return the radiance of the surface resulting
// from indirect illumination (i.e. other shapes in the scene) via the Monte Carlo Raytracing
vec3 indirect_irradiance(const Intersection& intersection, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces){

    float cos_theta;
    vec4 sampled_direction = sample_random_direction_around_intersection(intersection, cos_theta);

    // Create the new bounced ray
    vec4 start = intersection.position + (0.00001f * sampled_direction);
    start[3] = 1.f;
    Ray ray = Ray(start, sampled_direction);

    // 4) Get the radiance contribution for this ray and add to the sum
    vec3 radiance = path_trace(ray, surfaces, light_planes, bounces+1);

    // BRDF = reflectance / M_PI (equal from all angles for diffuse)
    // rho = 1/(2*M_PI) (probabiltiy of sampling a ray in the given direction)
    vec3 BRDF = surfaces[intersection.index]->get_material().get_diffuse_c() / (float)M_PI;
    
    // Approximate the rendering equation
    vec3 irradiance = (radiance * BRDF * cos_theta) / (1.f / (2.f * (float)M_PI));

    return irradiance;
}
