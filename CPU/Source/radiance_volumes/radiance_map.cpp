#include "radiance_map.h"
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <ctime>

RadianceMap::RadianceMap(bool precompute, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, std::vector<Surface>& surfaces_builder){
    
    std::cout << "Sampling radiance volumes..." << std::endl;
    
    // Find the time
    time_t start_time;
    time_t end_time;
    time_t temp_time;
	start_time = time(NULL);

    // For every surface in the scene, uniformly sample N Radiance
    // Volumes where N is based on the triangles surface area
    int surfaces_count = surfaces.size();
    for (int i = 0; i < surfaces_count; i++){
        uniformly_sample_radiance_volumes(*surfaces[i]);
    }
    
    // Find the time
    end_time = time(NULL);
    temp_time = end_time - start_time; 
    std::cout << "Sampled " << this->radiance_volumes.size() << " Radiance Volumes in " << temp_time << "s" << std::endl;
    
    // Get the radiance estimate for every radiance volume
    if (precompute){
         start_time = end_time;
        std::cout << "Getting radiance estimate for the radiance volumes..." << std::endl;
        get_radiance_estimates(surfaces, light_planes);
        end_time = time(NULL);
        temp_time = end_time - start_time;
        std::cout << "Radiance Volume Estimates Found in " << temp_time << "s" << std::endl;   
    }

    // Create the RadianceTree (KDTree) from the radiance volumes
    start_time = end_time;
    std::cout << "Building Radiance Tree..." << std::endl; 
    this->radiance_tree = std::unique_ptr<RadianceTree>(new RadianceTree(this->radiance_volumes, X_DIM));
    end_time = time(NULL);
    temp_time = end_time - start_time;
    std::cout << "Radiance Tree constructed in " << temp_time << std::endl;
}

/*             Construction                */

// Uniformly sample N Radiance volumes on a triangle based on the
// triangles surface area
void RadianceMap::uniformly_sample_radiance_volumes(Surface surface){
    // Calculate the number of radaince volumes to sample in that triangle
    int sample_count = (int)floor(surface.compute_area() / AREA_PER_SAMPLE);
    // Sample sample_count RadianceVolumes on the given surface
    for (int i = 0; i < sample_count; i++){
        vec4 sampled_position = surface.sample_position_on_plane();
        RadianceVolume* rv = new RadianceVolume(sampled_position, surface.getNormal());
        this->radiance_volumes.push_back(rv);
    }
}

// Get the radiance estimate for all radiance volumes in the RadianceMap
void RadianceMap::get_radiance_estimates(std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes){
    int volumes = this->radiance_volumes.size();
    #pragma omp parallel for
    for (int i = 0; i < volumes; i++){
        this->radiance_volumes[i]->get_radiance_estimate(surfaces, light_planes);
    }
}

// Builds all RadianceVolumes which are part of the RadianceMap into the scene
void RadianceMap::build_radiance_map_shapes(std::vector<Surface>& surfaces){
    int volumes = this->radiance_volumes.size();
    for (int i = 0; i < volumes; i++){
        this->radiance_volumes[i]->build_radiance_volume_shapes(surfaces);
    }
}

// Normalizes all RadianceVolumes radiance values i.e. their grid values
// all sum to 1 (taking the length of each vec3)
void RadianceMap::update_radiance_distributions(){
    int volumes = this->radiance_volumes.size();
    #pragma omp parallel for
    for (int i = 0; i < volumes; i++){
        this->radiance_volumes[i]->update_radiance_distribution();
    }
}

/*              Querying                */
// Get the estimated radiance for a given intersection point within the scene
// based on interpolation of the radiance map stored estimates 
vec3 RadianceMap::get_irradiance_estimate(const Intersection& intersection, std::vector<Surface *> surfaces){

    // Get the closest n points by maintaining a heap of values
    std::vector<RadianceVolume*> closest_volumes = this->radiance_tree->find_closest_radiance_volumes(CLOSEST_QUERY_COUNT, MAX_DIST, intersection.position, intersection.normal);
    
    // For each index, get the radiance and average the total radiance
    vec3 radiance = vec3(0.f);
    int volumes = closest_volumes.size();

    // Use barycentric interpolation if three radiance volumes have been found
    if (volumes == 3){
        float u, v;
        // Check that P lies in the triangle defined
        if (compute_barycentric(closest_volumes[0]->get_position(), closest_volumes[1]->get_position(), closest_volumes[2]->get_position(), intersection.position, u, v)){
            vec3 u_colour = closest_volumes[0]->get_irradiance(intersection, surfaces);
            vec3 v_colour = closest_volumes[1]->get_irradiance(intersection, surfaces);
            vec3 w_colour = closest_volumes[2]->get_irradiance(intersection, surfaces);
            radiance = u_colour * u + v_colour * v + w_colour * (1 - u - v);
        }
        // Else just take an average
        else{
            for (int i = 0; i < volumes; i++){
                radiance += closest_volumes[i]->get_irradiance(intersection, surfaces);
            }
            radiance /= (float)volumes;
        }
    }
    // Otherwise just take an average
    else if(volumes > 0){
        for (int i = 0; i < volumes; i++){
            radiance += closest_volumes[i]->get_irradiance(intersection, surfaces);
        }
        radiance /= (float)volumes;
    }
    return radiance;
}

// Calculates a gaussian filter constant for the passed in radiance volume distance and max radiance volume distance
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
RadianceVolume* RadianceMap::importance_sample_ray_direction(const Intersection& intersection, int& sector_x, int& sector_y, vec4& sampled_direction){

    // 1) Find the closest RadianceVolume
    std::vector<RadianceVolume*> closest_volumes = this->radiance_tree->find_closest_radiance_volumes(1, MAX_DIST, intersection.position, intersection.normal);

    // If a radiance volume is not found, just sample randomly 
    if (closest_volumes.size() < 1){
        float cos_theta;
        sampled_direction = sample_random_direction_around_intersection(intersection, cos_theta);
        return NULL;
    }
    else{
        // 2) Generate a random float uniformly between [0,1] and find which 
        //    part of the cumulative distribution this number falls in range
        //    of i.e. sample from the inverse of the cumulative distribution.
        //    This gives the location on the grid we sample our direction from
        //    Update the radiance distribution before attempting to sample
        closest_volumes[0]->update_radiance_distribution();
        sampled_direction = closest_volumes[0]->sample_direction_from_radiance_distribution(sector_x, sector_y);
        return closest_volumes[0];
    }
}

// Performs the temporal difference update for the radiance volume passed in given the sampled ray direction lead to the intersection
void RadianceMap::temporal_difference_update_radiance_volume_sector(RadianceVolume* current_radiance_volume, int current_sector_x, int current_sector_y, Intersection& intersection, std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes){

    switch (intersection.intersection_type){

        case NOTHING:
            current_radiance_volume->temporal_difference_update(vec3(0.f), current_sector_x, current_sector_y);
            break;
        
        case AREA_LIGHT_PLANE:
            current_radiance_volume->temporal_difference_update(light_planes[intersection.index]->get_diffuse_p(), current_sector_x, current_sector_y);
            break;
        
        case SURFACE:
            // Get the radiance volume closest to the intersection point
            std::vector<RadianceVolume*> closest_volumes = this->radiance_tree->find_closest_radiance_volumes(1, MAX_DIST, intersection.position, intersection.normal);

            if (closest_volumes.size() < 1){
                return;
            }
            else{
                // Get the radiance incident from all directions for the next position and perform temporal diff update
                vec3 next_pos_irradiance = closest_volumes[0]->get_irradiance(intersection, surfaces);
                current_radiance_volume->temporal_difference_update(next_pos_irradiance, current_sector_x, current_sector_y);
            }
            break;
    }
}

// Set the voronoi colours of all radiance volumes in the scene in the first entry of the radiance_grid[0][0]
void RadianceMap::set_voronoi_colours(){
    int volumes = this->radiance_volumes.size();
    for (int i = 0; i < volumes; i++){
        this->radiance_volumes[i]->set_voronoi_colour();
    }
}

// Get the voronoi colour of the closest radiance volume
vec3 RadianceMap::get_voronoi_colour(const Intersection& intersection){
    std::vector<RadianceVolume*> closest_volumes = this->radiance_tree->find_closest_radiance_volumes(1, 5.f , intersection.position, intersection.normal);
    return closest_volumes[0]->get_voronoi_colour();
}