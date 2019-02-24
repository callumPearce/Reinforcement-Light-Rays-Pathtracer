#include "radiance_map.h"
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <ctime>

RadianceMap::RadianceMap(std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, std::vector<Surface>& surfaces_builder){
    
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
    start_time = end_time;
    std::cout << "Getting radiance estimate for the radiance volumes..." << std::endl;
    get_radiance_estimates(surfaces, light_planes);
    end_time = time(NULL);
    temp_time = end_time - start_time;
    std::cout << "Radiance Volume Found in " << temp_time << "s" << std::endl;

    
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
void RadianceMap::normalize_radiance_volumes(){
    int volumes = this->radiance_volumes.size();
    #pragma omp parallel for
    for (int i = 0; i < volumes; i++){
        this->radiance_volumes[i]->normalize_radiance_volume();
    }
}

/*              Querying                */
// Get the estimated radiance for a given intersection point within the scene
// based on interpolation of the radiance map stored estimates 
vec3 RadianceMap::get_irradiance_estimate(const Intersection& intersection, std::vector<Surface *> surfaces){

    // Get the closest n points by maintaining a heap of values
    std::vector<RadianceVolume*> closest_volumes = this->radiance_tree->find_closest_radiance_volumes(CLOSEST_QUERY_COUNT, MAX_DIST, intersection.position);
    
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
vec4 RadianceMap::importance_sample_ray_direction(const Intersection& intersection){

    // 1) Find the closest RadianceVolume
    std::vector<RadianceVolume*> closest_volumes = this->radiance_tree->find_closest_radiance_volumes(1, MAX_DIST, intersection.position);

    // If a radiance volume is not found, just sample randomly 
    if (closest_volumes.size() < 1){
        float cos_theta;
        return sample_random_direction_around_intersection(intersection, cos_theta);
    }

    // 2) Calculate the cumulative sum of the radiance_map stored in the volume


    // 3) Generate a random float uniformly between [0,1]

    // 4) Find which part of the cumulative distribution this number
    //    falls in range of i.e. sample from the inverse of the cumulative
    //    distribution. This gives the location on the grid we sample
    //    our direction from
}