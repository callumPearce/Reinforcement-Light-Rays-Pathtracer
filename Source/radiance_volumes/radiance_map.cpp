#include "radiance_map.h"
#include "printing.h"
#include "radiance_volumes_settings.h"
#include <algorithm>

RadianceMap::RadianceMap(vector<Surface *> surfaces){
    // For every surface in the scene, uniformly sample N Radiance
    // Volumes where N is based on the triangles surface area 
    for (int i = 0; i < surfaces.size(); i++){
        uniformly_sample_radiance_volumes(*surfaces[i]);
    }
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
        RadianceVolume rv = RadianceVolume(sampled_position, surface.getNormal());
        this->radiance_volumes.push_back(rv);
    }
}

// Get the radiance estimate for all radiance volumes in the RadianceMap
void RadianceMap::get_radiance_estimates(vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes){
    int volumes = this->radiance_volumes.size();
    for (int i = 0; i < volumes; i++){
        this->radiance_volumes[i].get_radiance_estimate(surfaces, light_planes);
    }
}

// Builds all RadianceVolumes which are part of the RadianceMap into the scene
void RadianceMap::build_radiance_map_shapes(vector<Surface>& surfaces){
    int volumes = this->radiance_volumes.size();
    for (int i = 0; i < volumes; i++){
        this->radiance_volumes[i].build_radiance_volume_shapes(surfaces);
    }
}

/*              Querying                */

// Updates the list of closest distances/closest indices with the new value
// if it is smaller than the largest element in the current list]
void update_list(float new_distance, vector<float>& closest_distances, int index, vector<int>& closest_indices){
    
    if (closest_distances.size() < CLOSEST_QUERY_COUNT){
        closest_distances.push_back(new_distance);
        closest_indices.push_back(index);
    }
    else{
        if (new_distance < closest_distances[0]){
            return;
        }
        else{
            // Delete the first element
            closest_distances.erase(closest_distances.begin());
            closest_indices.erase(closest_indices.begin());
            // Find the index to insert the element into 
            int j = 0;
            while (new_distance < closest_distances[j] && j < CLOSEST_QUERY_COUNT){
                j++;
            }
            closest_distances.insert(closest_distances.begin()+j-1, new_distance);
            closest_indices.insert(closest_indices.begin()+j-1, index);
        }
    }
}

// Get the estimated radiance for a given intersection point within the scene
// based on interpolation of the radiance map stored estimates 
vec3 RadianceMap::get_radiance_estimate(const Intersection& intersection, vector<Surface *> surfaces){

    // Get the closest n points by maintaining a heap of values
    vector<int> closest_indices;
    vector<float> closest_distances;
    int volumes = this->radiance_volumes.size();
    for (int i = 0; i < volumes; i++){
        float temp_dist = distance(this->radiance_volumes[i].get_position(), intersection.position);
        update_list(temp_dist, closest_distances, i, closest_indices);
    }

    // Weight the distances values
    float distances_sum = 0.f;
    vector<float> distance_weightings;
    for (int i = 0; i < CLOSEST_QUERY_COUNT; i++){
        distances_sum += closest_distances[i];
    }
    for (int i = 0; i < CLOSEST_QUERY_COUNT; i++){
        closest_distances[i] = distances_sum - closest_distances[i];
    }
    distances_sum = 0.f;
    for (int i = 0; i < CLOSEST_QUERY_COUNT; i++){
        distances_sum += closest_distances[i];
    }
    for (int i = 0; i < CLOSEST_QUERY_COUNT; i++){
        distance_weightings.push_back(closest_distances[i]/distances_sum);
    }
    
    // For each index, get the radiance, multiply it by the weighting and add to the total
    vec3 radiance = vec3(0.f);
    for (int i = 0; i < CLOSEST_QUERY_COUNT; i++){
        radiance += distance_weightings[i] * this->radiance_volumes[closest_indices[i]].get_total_radiance(intersection, surfaces);
    }

    return radiance;
}