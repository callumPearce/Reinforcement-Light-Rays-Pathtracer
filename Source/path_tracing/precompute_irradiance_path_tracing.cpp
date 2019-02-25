#include "precompute_irradiance_path_tracing.h"

// Traces the path of a ray following monte carlo path tracer in order to estimate the radiance for a ray shot
// from its angle and starting position
vec3 path_trace_radiance_map(Camera& camera, int pixel_x, int pixel_y, RadianceMap& radiance_map, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes){

    // Set direction to pass through pixel (pixel space -> Camera space)
    vec4 dir(((float)pixel_x - (float)SCREEN_WIDTH / 2.f) , ((float)pixel_y - (float)SCREEN_HEIGHT / 2.f) , (float)FOCAL_LENGTH , 1);
    
    // Create a ray that we will change the direction for below
    Ray ray(camera.get_position(), dir);
    ray.rotate_ray(camera.get_yaw());

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