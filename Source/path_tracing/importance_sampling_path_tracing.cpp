#include "importance_sampling_path_tracing.h"

/* Importance Sampling Path Tracing */
vec3 path_trace_importance_sampling(RadianceMap& radiance_map, Ray ray, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces){
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
                return importance_sample_ray(closest_intersection, radiance_map, surfaces, light_planes, bounces);
            }
            break;
    }

    return vec3(0);
}

vec3 importance_sample_ray(const Intersection& intersection, RadianceMap& radiance_map, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces){

    //1) Sample the rays direction via importance sampling from the closest RadianceVolume
    vec4 sampled_direction = radiance_map.importance_sample_ray_direction(intersection);

    // Create the new bounced ray
    vec4 start = intersection.position + (0.00001f * sampled_direction);
    start[3] = 1.f;
    Ray ray = Ray(start, sampled_direction);

    // 2) Get the radiance contribution for this ray
    vec3 radiance = path_trace_importance_sampling(radiance_map, ray, surfaces, light_planes, bounces+1);

    // 3) Solve the rendering equation
    // BRDF = reflectance / M_PI (equal from all angles for diffuse)
    // rho = 1/(2*M_PI) (probabiltiy of sampling a ray in the given direction)
    vec3 BRDF = surfaces[intersection.index]->get_material().get_diffuse_c() / (float)M_PI;
    
    // cos(theta): angle between the normal and the direction of the bounced ray
    float cos_theta = dot(vec3(surfaces[intersection.index]->getNormal()), vec3(sampled_direction));

    // Approximate the rendering equation
    vec3 irradiance = (radiance * BRDF * cos_theta) / (1.f / (2.f * (float)M_PI));

    return irradiance;
}