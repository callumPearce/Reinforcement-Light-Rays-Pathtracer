#include "radiance_map.h"
#include "printing.h"
#include "radiance_volumes_settings.h"
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <ctime>

RadianceMap::RadianceMap(vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes, vector<Surface>& surfaces_builder){
    cout << "Sampling radiance volumes..." << endl;
    
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
    cout << "Sampled " << this->radiance_volumes.size() << " Radiance Volumes in " << temp_time << "s" << endl;
    
    // Get the radiance estimate for every radiance volume
    start_time = end_time;
    cout << "Getting radiance estimate for the radiance volumes..." << endl;
    get_radiance_estimates(surfaces, light_planes);
    end_time = time(NULL);
    temp_time = end_time - start_time;
    cout << "Radiance Volume Found in " << temp_time << "s" << endl;

    
    // Create the RadianceTree (KDTree) from the radiance volumes
    start_time = end_time;
    cout << "Building Radiance Tree..." << endl; 
    this->radiance_tree = std::unique_ptr<RadianceTree>(new RadianceTree(this->radiance_volumes, X_DIM));
    end_time = time(NULL);
    temp_time = end_time - start_time;
    cout << "Radiance Tree constructed in " << temp_time << endl;
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
void RadianceMap::get_radiance_estimates(vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes){
    int volumes = this->radiance_volumes.size();
    #pragma omp parallel for
    for (int i = 0; i < volumes; i++){
        this->radiance_volumes[i]->get_radiance_estimate(surfaces, light_planes);
    }
}

// Builds all RadianceVolumes which are part of the RadianceMap into the scene
void RadianceMap::build_radiance_map_shapes(vector<Surface>& surfaces){
    int volumes = this->radiance_volumes.size();
    for (int i = 0; i < volumes; i++){
        this->radiance_volumes[i]->build_radiance_volume_shapes(surfaces);
    }
}

/*              Querying                */
// Get the estimated radiance for a given intersection point within the scene
// based on interpolation of the radiance map stored estimates 
vec3 RadianceMap::get_irradiance_estimate(const Intersection& intersection, vector<Surface *> surfaces){

    // Get the closest n points by maintaining a heap of values
    vector<RadianceVolume*> closest_volumes = this->radiance_tree->find_closest_radiance_volumes(CLOSEST_QUERY_COUNT, MAX_DIST, intersection.position);
    
    // For each index, get the radiance and average the total radiance
    vec3 radiance = vec3(0.f);
    int volumes = closest_volumes.size();

    // Get the largest distance between nearest radiance volumes (closest being the first one in the returned list)
    // and calculate the radiance whilst applying the filter
    if (volumes > 0){
        float furthest_distance = distance(vec3(closest_volumes[0]->get_position()), vec3(intersection.position));
        // Calculate the gaussian coefficients and ensure that their total adds up volumes
        float gaussian_sum = 0.f;
        vector<float> gaussian_coeffs;
        for (int i = 0; i < volumes; i++){
            float dist = distance(vec3(closest_volumes[i]->get_position()), vec3(intersection.position));
            float gaussian_coeff = calculate_gaussian_filter(dist, furthest_distance);
            gaussian_coeffs.push_back(gaussian_coeff);
            gaussian_sum += gaussian_coeff;
        }
        // Calculate the radiance with the scaled gaussian coefficient applied
        float gaussian_scale = volumes/gaussian_sum;
        for (int i = 0; i < volumes; i++){
            radiance += closest_volumes[i]->get_total_irradiance(intersection, surfaces) * gaussian_coeffs[i] * gaussian_scale;
        }
        radiance /= (float)volumes;
    }
    return radiance;
}

//Calculates a gaussian filter constant for the passed in radiance volume distance and max radiance volume distance
float RadianceMap::calculate_gaussian_filter(float volume_distance, float furthest_volume_distance){
    float alpha = 0.918f;
    float beta = 1.953f;
    float numerator = 1.0f - exp( -beta * ((pow(volume_distance,2)) / (2 * pow(furthest_volume_distance,2))) );
    float denominator = 1.0f - exp(-beta);
    float w_pc = alpha * ( numerator / denominator);
    return w_pc;
}