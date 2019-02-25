#include "reinforcement_path_tracing.h"

vec3 path_trace_reinforcement(Camera& camera, int pixel_x, int pixel_y, RadianceMap& radiance_map, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes){
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
        irradiance += path_trace_reinforcement_recursive(radiance_map, ray, surfaces, light_planes, 0);
    }
    irradiance /= (float)SAMPLES_PER_PIXEL;
    return irradiance;
}

vec3 path_trace_reinforcement_recursive(RadianceMap& radiance_map, Ray ray, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, int bounces){
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

/* TODOs:
    Sample direction inside pixel should be callable from both
    Importance sample ray should be callable from both importance sampling and reinforcement learning
*/
