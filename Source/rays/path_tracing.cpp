#include "path_tracing.h"
#include "radiance_volumes_settings.h"
#include <iostream>
#include "printing.h"

// Traces the path of a ray following monte carlo path tracer in order to estimate the radiance for a ray shot
// from its angle and starting position
vec3 path_trace(bool radiance_volume, Ray ray, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes, int bounces){
    
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
                return indirect_radiance(radiance_volume, closest_intersection, surfaces, light_planes, bounces);
            }
            break;
    }

    return vec3(0);
}

// Traces the path of a ray following monte carlo path tracer in order to estimate the radiance for a ray shot
// from its angle and starting position
vec3 path_trace_radiance_map(RadianceMap& radiance_map, Ray ray, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes){

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
            return radiance_map.get_irradiance_estimate(closest_intersection, surfaces);
            break;
    }

    return vec3(0);
}

// For a given intersection point, return the radiance of the surface resulting
// from indirect illumination (i.e. other shapes in the scene) via the Monte Carlo Raytracing
vec3 indirect_radiance(bool radiance_volume, const Intersection& intersection, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes, int bounces){

    // 1) Create new coordinate system (tranformation matrix)
    vec3 normal = vec3(intersection.normal.x, intersection.normal.y, intersection.normal.z);
    vec3 normal_T = vec3(0);
    vec3 normal_B = vec3(0);
    create_normal_coordinate_system(normal, normal_T, normal_B);

    // Calculate the estiamted total radiance estimate
    // (\int L(w_i) * roh / pi * cos(w_i, N) dw)
    vec3 total_radiance = vec3(0);
    int sample_count = radiance_volume ? RADIANCE_SAMPLES_PER_BOUNCE : SAMPLES_PER_BOUNCE;
    for (int i = 0; i < sample_count; i++){
        
        // Generate random number for monte carlo sampling of theta and phi
        float cos_theta = ((float) rand() / (RAND_MAX)); //r1
        float r2 = ((float) rand() / (RAND_MAX));

        // 2) Sample uniformly coordinates on unit hemisphere
        vec3 sample = uniform_hemisphere_sample(cos_theta, r2);

        // 3) Transform random sampled direction into the world coordinate system
        vec3 sampled_direction = vec3(
            sample.x * normal_B.x + sample.y * normal.x + sample.z * normal_T.x, 
            sample.x * normal_B.y + sample.y * normal.y + sample.z * normal_T.y, 
            sample.x * normal_B.z + sample.y * normal.z + sample.z * normal_T.z
        );

        // Create the new bounced ray
        vec4 start = intersection.position + (0.00001f * vec4(sampled_direction, 1));
        start[3] = 1.f;
        Ray ray = Ray(start, vec4(sampled_direction, 1));

        // 4) Get the radiance contribution for this ray and add to the sum
        vec3 radiance = path_trace(radiance_volume, ray, surfaces, light_planes, bounces+1);

        // Note: we can multiply the BRDF to the final sum because it is 
        // constant for diffuse surfaces, so we omit it here
        total_radiance += cos_theta * radiance;
    }

    // Divide the sum by the number of samples (Monte Carlo) and apply BRDF
    // Note: 1/2pi comes from PDF being constant for sampled ray directions (Monte Carlo) 
    total_radiance /= ((float)sample_count * (1 / (2 * M_PI))); 
    total_radiance *= surfaces[intersection.index]->get_material().get_diffuse_c();

    return total_radiance;
}