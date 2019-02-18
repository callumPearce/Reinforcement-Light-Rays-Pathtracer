#include "radiance_map.h"
#include "printing.h"
#include "radiance_volumes_settings.h"

RadianceMap::RadianceMap(vector<Surface *> surfaces){
    // For every surface in the scene, uniformly sample N Radiance
    // Volumes where N is based on the triangles surface area 
    for (int i = 0; i < surfaces.size(); i++){
        uniformly_sample_radiance_volumes(*surfaces[i]);
    }
}

// Uniformly sample N Radiance volumes on a triangle based on the
// triangles surface area
void RadianceMap::uniformly_sample_radiance_volumes(Surface surface){
    // Calculate the number of radaince volumes to sample in that triangle
    int sample_count = (int)floor(surface.compute_area() / AREA_PER_SAMPLE);
    cout << surface.compute_area() << endl;
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