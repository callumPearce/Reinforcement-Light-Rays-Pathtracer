#include "radiance_map.cuh"
#include <algorithm>
#include <iostream>
#include <ctime>
#include "printing.h"

__host__
RadianceMap::RadianceMap(Surface* surfaces, int surfaces_count, std::vector<RadianceVolume>& temp_rvs){
    
    std::cout << "Sampling radiance volumes..." << std::endl;
    
    // Find the time
    time_t start_time;
    time_t end_time;
    time_t temp_time;
	start_time = time(NULL);

    // For every surface in the scene, uniformly sample N Radiance
    // Volumes where N is based on the triangles surface area
    get_radiance_volumes_count(surfaces, surfaces_count);
    size_t size = this->radiance_volumes_count * sizeof(RadianceVolume);
    printf("RadianceMap size: %zu bytes\n",size);
    temp_rvs =  std::vector<RadianceVolume>(this->radiance_volumes_count);
    uniformly_sample_radiance_volumes(surfaces, surfaces_count, temp_rvs);

    // Find the time
    end_time = time(NULL);
    temp_time = end_time - start_time; 
    std::cout << "Sampled " << temp_rvs.size() << " Radiance Volumes in " << temp_time << "s" << std::endl;

    // Create the RadianceTree (KDTree) from the radiance volumes
    // start_time = end_time;
    // std::cout << "Building Radiance Tree..." << std::endl; 
    // this->radiance_tree = new RadianceTree(radiance_vs, X_DIM);
    // end_time = time(NULL);
    // temp_time = end_time - start_time;
    // std::cout << "Radiance Tree constructed in " << temp_time << std::endl;
}


/*             Construction                */

__host__
void RadianceMap::get_radiance_volumes_count(Surface* surfaces, int surfaces_count){
    // Get sample count
    int total_sample_count = 0;
    for (int i = 0; i < surfaces_count; i++){
        total_sample_count += (int)floor(surfaces[i].compute_area() / AREA_PER_SAMPLE);
    }
    this->radiance_volumes_count = total_sample_count;
}

// Uniformly sample N Radiance volumes on a triangle based on the
// triangles surface area
__host__
void RadianceMap::uniformly_sample_radiance_volumes(Surface* surfaces, int surfaces_count, std::vector<RadianceVolume>& temp_rvs){
    int x = 0;
    for (int j = 0; j < surfaces_count; j++){
        // Calculate the number of radaince volumes to sample in that triangle
        int sample_count = (int)floor(surfaces[j].compute_area() / AREA_PER_SAMPLE);
        // Sample sample_count RadianceVolumes on the given surface
        for (int i = 0; i < sample_count; i++){
            vec4 sampled_position = surfaces[j].sample_position_on_plane();
            temp_rvs[x] = RadianceVolume(sampled_position, surfaces[j].normal);
            x++;
        }
    }
}

// Get the radiance estimate for all radiance volumes in the RadianceMap
__device__
void RadianceMap::get_radiance_estimates(curandState* volume_rand_state, Surface* surfaces, AreaLight* light_planes){
    for (int i = 0; i < this->radiance_volumes_count; i++){
        this->radiance_volumes[i].get_radiance_estimate_per_sector(volume_rand_state, surfaces, light_planes);
    }
}

// // Builds all RadianceVolumes which are part of the RadianceMap into the scene
// __host__
// void RadianceMap::build_radiance_map_shapes(std::vector<Surface>& surfaces){
//     for (int i = 0; i < this->radiance_volumes_count; i++){
//         this->radiance_volumes[i].build_radiance_volume_shapes(surfaces);
//     }
// }

// Normalizes all RadianceVolumes radiance values i.e. their grid values
// all sum to 1 (taking the length of each vec3)
__device__
void RadianceMap::update_radiance_distributions(){
    for (int i = 0; i < this->radiance_volumes_count; i++){
        this->radiance_volumes[i].update_radiance_distribution();
    }
}

/*              Querying                */
// Get the estimated radiance for a given intersection point within the scene
// based on interpolation of the radiance map stored estimates 
// vec3 RadianceMap::get_irradiance_estimate(const Intersection& intersection, Surface* surfaces){

//     // Get the closest n points by maintaining a heap of values
//     std::vector<RadianceVolume*> closest_volumes = this->radiance_tree->find_closest_radiance_volumes(CLOSEST_QUERY_COUNT, MAX_DIST, intersection.position, intersection.normal);
    
//     // For each index, get the radiance and average the total radiance
//     vec3 radiance = vec3(0.f);
//     int volumes = closest_volumes.size();

//     // Use barycentric interpolation if three radiance volumes have been found
//     if (volumes == 3){
//         float u, v;
//         // Check that P lies in the triangle defined
//         if (compute_barycentric(closest_volumes[0]->get_position(), closest_volumes[1]->get_position(), closest_volumes[2]->get_position(), intersection.position, u, v)){
//             vec3 u_colour = closest_volumes[0]->get_irradiance(intersection, surfaces);
//             vec3 v_colour = closest_volumes[1]->get_irradiance(intersection, surfaces);
//             vec3 w_colour = closest_volumes[2]->get_irradiance(intersection, surfaces);
//             radiance = u_colour * u + v_colour * v + w_colour * (1 - u - v);
//         }
//         // Else just take an average
//         else{
//             for (int i = 0; i < volumes; i++){
//                 radiance += closest_volumes[i]->get_irradiance(intersection, surfaces);
//             }
//             radiance /= (float)volumes;
//         }
//     }
//     // Otherwise just take an average
//     else if(volumes > 0){
//         for (int i = 0; i < volumes; i++){
//             radiance += closest_volumes[i]->get_irradiance(intersection, surfaces);
//         }
//         radiance /= (float)volumes;
//     }
//     return radiance;
// }

// Calculates a gaussian filter constant for the passed in radiance volume distance and max radiance volume distance
__device__
float RadianceMap::calculate_gaussian_filter(float volume_distance, float furthest_volume_distance){
    float alpha = 0.918f;
    float beta = 1.953f;
    float numerator = 1.0f - exp( -beta * ((pow(volume_distance,2)) / (2 * pow(furthest_volume_distance,2))) );
    float denominator = 1.0f - exp(-beta);
    float w_pc = alpha * ( numerator / denominator);
    return w_pc;
}

// Given an intersection point, importance sample a ray direction according to the
// cumulative distribution formed by the closest RadianceVolume's radiance_map
__device__
RadianceVolume* RadianceMap::importance_sample_ray_direction(curandState* volume_rand_state, const Intersection& intersection, int& sector_x, int& sector_y, vec4& sampled_direction){

    // 1) Find the closest RadianceVolume
    // RadianceVolume* closest_volume = this->radiance_tree->find_closest_radiance_volume(MAX_DIST, intersection.position, intersection.normal);
    RadianceVolume* closest_volume = this->get_closest_radiance_volume_linear(MAX_DIST, intersection.position, intersection.normal);

    // If a radiance volume is not found, just sample randomly 
    if (closest_volume == NULL){
        float cos_theta;
        // sampled_direction = sample_random_direction_around_intersection(intersection, cos_theta); TODO FIX
        return NULL;
    }
    else{
        // 2) Generate a random float uniformly between [0,1] and find which 
        //    part of the cumulative distribution this number falls in range
        //    of i.e. sample from the inverse of the cumulative distribution.
        //    This gives the location on the grid we sample our direction from
        //    Update the radiance distribution before attempting to sample
        closest_volume->update_radiance_distribution();
        sampled_direction = closest_volume->sample_direction_from_radiance_distribution(volume_rand_state, sector_x, sector_y);
        return closest_volume;
    }
}

// Performs the temporal difference update for the radiance volume passed in given the sampled ray direction lead to the intersection
__device__
void RadianceMap::temporal_difference_update_radiance_volume_sector(RadianceVolume* current_radiance_volume, int current_sector_x, int current_sector_y, Intersection& intersection, Surface* surfaces, AreaLight* light_planes){

    switch (intersection.intersection_type){

        case NOTHING:
            current_radiance_volume->temporal_difference_update(vec3(0.f), current_sector_x, current_sector_y);
            break;
        
        case AREA_LIGHT:
            current_radiance_volume->temporal_difference_update(light_planes[intersection.index].diffuse_p, current_sector_x, current_sector_y);
            break;
        
        case SURFACE:
            // Get the radiance volume closest to the intersection point
            // RadianceVolume* closest_volume = this->radiance_tree->find_closest_radiance_volume(MAX_DIST, intersection.position, intersection.normal);
            RadianceVolume* closest_volume = this->get_closest_radiance_volume_linear(MAX_DIST, intersection.position, intersection.normal);

            if (closest_volume == NULL){
                return;
            }
            else{
                // Get the radiance incident from all directions for the next position and perform temporal diff update
                vec3 next_pos_irradiance = closest_volume->get_irradiance(intersection, surfaces);
                current_radiance_volume->temporal_difference_update(next_pos_irradiance, current_sector_x, current_sector_y);
            }
            break;
    }
}

// Set the voronoi colours of all radiance volumes in the scene in the first entry of the radiance_grid[0][0]
__host__
void RadianceMap::set_voronoi_colours(std::vector<RadianceVolume>& temp_rvs){
    for (int i = 0; i < this->radiance_volumes_count; i++){
        temp_rvs[i].set_voronoi_colour();
    }
}

// Get the voronoi colour of the closest radiance volume
__device__
vec3 RadianceMap::get_voronoi_colour(const Intersection& intersection){
    // RadianceVolume* closest_volume = this->radiance_tree->find_closest_radiance_volume(MAX_DIST, intersection.position, intersection.normal);
    RadianceVolume* closest_volume = this->get_closest_radiance_volume_linear(MAX_DIST, intersection.position, intersection.normal);
    vec4 pos = closest_volume->position;
    if (closest_volume != NULL){
        return closest_volume->get_voronoi_colour();   
    }
    else{
        return vec3(0.f);
    }
}

// Find the closest radiance volume in linear time by traversing the list of radiance volumes
__device__
RadianceVolume* RadianceMap::get_closest_radiance_volume_linear(float max_dist, vec4 position, vec4 normal){
    RadianceVolume* current_closest = &(this->radiance_volumes[0]);
    // vec4 pos = current_closest->position;
    // printf("%f,%f,%f,%f\n", pos.x, pos.y, pos.z, pos.w);
    float closest_distance = glm::distance(this->radiance_volumes[0].position, position);

    for (int i = 1; i < this->radiance_volumes_count; i++){
        float temp_dist = glm::distance(this->radiance_volumes[i].position, position);
        if ( temp_dist < closest_distance && vec3(normal) == this->radiance_volumes[i].normal){
            current_closest = &this->radiance_volumes[i];
            closest_distance = temp_dist;
        }
    }
    return current_closest;
}