#include "radiance_map.h"
#include "printing.h"
#include "radiance_volumes_settings.h"
#include <algorithm>
#include <omp.h>

RadianceMap::RadianceMap(vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes){
    // For every surface in the scene, uniformly sample N Radiance
    // Volumes where N is based on the triangles surface area
    int surfaces_count = surfaces.size();
    vector<RadianceVolume> radiance_volumes;
    for (int i = 0; i < surfaces_count; i++){
        uniformly_sample_radiance_volumes(radiance_volumes, *surfaces[i]);
    }
    // Get the radiance estimate for every radiance volume
    get_radiance_estimates(radiance_volumes, surfaces, light_planes);
    // Create the RadianceTree (KDTree) from the radiance volumes 
    // (assign in to ensure it is not lost after function call)
    this->radiance_tree = new RadianceTree(radiance_volumes, X_DIM);
}

/*             Construction                */

// Uniformly sample N Radiance volumes on a triangle based on the
// triangles surface area
void RadianceMap::uniformly_sample_radiance_volumes(vector<RadianceVolume>& radiance_volumes, Surface surface){
    // Calculate the number of radaince volumes to sample in that triangle
    int sample_count = (int)floor(surface.compute_area() / AREA_PER_SAMPLE);
    // Sample sample_count RadianceVolumes on the given surface
    for (int i = 0; i < sample_count; i++){
        vec4 sampled_position = surface.sample_position_on_plane();
        RadianceVolume rv = RadianceVolume(sampled_position, surface.getNormal());
        radiance_volumes.push_back(rv);
    }
}

// Get the radiance estimate for all radiance volumes in the RadianceMap
void RadianceMap::get_radiance_estimates(vector<RadianceVolume>& radiance_volumes, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes){
    int volumes = radiance_volumes.size();
    for (int i = 0; i < volumes; i++){
        radiance_volumes[i].get_radiance_estimate(surfaces, light_planes);
    }
}

// Builds all RadianceVolumes which are part of the RadianceMap into the scene
void RadianceMap::build_radiance_map_shapes(vector<RadianceVolume>& radiance_volumes, vector<Surface>& surfaces){
    int volumes = radiance_volumes.size();
    for (int i = 0; i < volumes; i++){
        radiance_volumes[i].build_radiance_volume_shapes(surfaces);
    }
}

// Get global pointer for the tree
RadianceTree* RadianceMap::get_global_radiance_tree_pointer(){
    return this->radiance_tree;
}

/*              Querying                */
// Get the estimated radiance for a given intersection point within the scene
// based on interpolation of the radiance map stored estimates 
vec3 RadianceMap::get_irradiance_estimate(const Intersection& intersection, vector<Surface *> surfaces){

    // Get the closest n points by maintaining a heap of values
    vector<RadianceVolume> closest_volumes = this->radiance_tree->find_closest_radiance_volumes(CLOSEST_QUERY_COUNT, MAX_DIST, intersection.position);
    
    // For each index, get the radiance and average the total radiance
    vec3 radiance = vec3(0.f);
    int volumes = closest_volumes.size();
    for (int i = 0; i < volumes; i++){
        radiance += closest_volumes[i].get_total_irradiance(intersection, surfaces);
    }
    radiance /= (float)volumes;
    return radiance;
}