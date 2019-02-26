#include "reinforcement_path_tracing.h"

void draw_reinforcement_path_tracing(SDLScreen screen, Camera& camera, RadianceMap& radiance_map, std::vector<AreaLightPlane *> light_planes, std::vector<Surface *> surfaces){
    // Reset the SDL screen to black
    memset(screen.buffer, 0, screen.height*screen.width*sizeof(uint32_t));

    #pragma omp parallel for
    for (int x = 0; x < SCREEN_WIDTH; x++){
        for (int y = 0; y < SCREEN_HEIGHT; y++){

            // Path trace the ray to find the colour to paint the pixel
            vec3 irradiance = path_trace_reinforcement(camera, x, y, radiance_map, surfaces, light_planes);
            screen.PutPixelSDL(x, y, irradiance);
        }
    }
}

vec3 path_trace_reinforcement(Camera& camera, int pixel_x, int pixel_y, RadianceMap& radiance_map, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes){
    vec3 irradiance = vec3(0.f);
    for (int i = 0; i < SAMPLES_PER_PIXEL; i++){
        
        Ray ray = Ray::sample_ray_through_pixel(camera, pixel_x, pixel_y);

        // Trace the path of the ray
        irradiance += path_trace_reinforcement_iterative(radiance_map, ray, surfaces, light_planes);
    }
    irradiance /= (float)SAMPLES_PER_PIXEL;
    return irradiance;
}

vec3 path_trace_reinforcement_iterative(RadianceMap& radiance_map, Ray ray, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes){

    vec3 throughput = vec3(1.f);
    
    RadianceVolume* current_radiance_volume;
    int current_sector_x = -1;
    int current_sector_y = -1;

    for (int i = 0; i < MAX_RAY_BOUNCES; i++){

        // Trace the path of the ray to find the closest intersection
        Intersection closest_intersection;
        ray.closest_intersection(surfaces, light_planes, closest_intersection);

        // We cannot update Q on the first bounce as it is the camera position,
        // not a point in the scene
        if (i > 0){
            // Update Q
            // where x = ray.start, y = intersection.position
            // Check that a radiance volume has been found to update its sector
            if (current_radiance_volume && current_sector_x != -1 && current_sector_y != -1){
                radiance_map.temporal_difference_update_radiance_volume_sector(current_radiance_volume, current_sector_x, current_sector_y, closest_intersection, surfaces, light_planes);
                current_sector_x = -1;
                current_sector_y = -1;
            } 
        }

        // Check what they ray intersected with...
        switch(closest_intersection.intersection_type){
            // Interescted with nothing, so no radiance
            case NOTHING:
                return vec3(0);
                break;
            
            // Intersected with light plane, so return its diffuse_p
            case AREA_LIGHT_PLANE:
                return throughput * light_planes[closest_intersection.index]->get_diffuse_p();
                break;

            // Intersected with a surface (diffuse)
            case SURFACE:
                vec4 sampled_direction = vec4(0.f);
                current_radiance_volume = radiance_map.importance_sample_ray_direction(closest_intersection, current_sector_x, current_sector_y, sampled_direction);

                vec3 BRDF = surfaces[closest_intersection.index]->get_material().get_diffuse_c() / (float)M_PI;
                float cos_theta = dot(vec3(surfaces[closest_intersection.index]->getNormal()), vec3(sampled_direction));
                float rho = (1.f / (2.f * (float)M_PI));

                throughput *= (BRDF * cos_theta) / rho;
                
                vec4 start = closest_intersection.position + sampled_direction * 0.00001f;
                start.w = 1.f;
                ray = Ray(start, sampled_direction);
                break;
            }
    }
    return vec3(0);
}