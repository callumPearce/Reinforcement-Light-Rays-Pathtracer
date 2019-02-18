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

void insert_to_distance_vector(vector<float>& closest_distances, vector<int>& closest_indices, float distance, int index){
    if (closest_distances.size() < CLOSEST_QUERY_COUNT){
        int j = 0;
        while (j < closest_distances.size() && distance < closest_distances[j]){
            j++;
        }
        closest_distances.insert(closest_distances.begin()+j, distance);
        closest_indices.insert(closest_indices.begin()+j, index);
    }
    else if (distance < closest_distances[0] ){
        int j = 1;
        closest_distances[0] = distance;
        closest_indices[0] = index;
        while (j < CLOSEST_QUERY_COUNT && distance < closest_distances[j]){
            float temp_d = closest_distances[j];
            int temp_i = closest_indices[j];
            closest_distances[j] = closest_distances[j-1];
            closest_indices[j] = closest_indices[j-1];
            closest_distances[j-1] = temp_d;
            closest_indices[j-1] = temp_i;
            j++;
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
        insert_to_distance_vector(closest_distances, closest_indices, temp_dist, i);
    }
    
    // For each index, get the radiance and average the total radiance
    vec3 radiance = vec3(0.f);
    for (int i = 0; i < CLOSEST_QUERY_COUNT; i++){
        radiance += this->radiance_volumes[closest_indices[i]].get_total_radiance(intersection, surfaces);
    }
    radiance /= (float)CLOSEST_QUERY_COUNT;
    return radiance;
}