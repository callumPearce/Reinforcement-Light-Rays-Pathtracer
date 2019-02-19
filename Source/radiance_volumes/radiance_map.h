#ifndef RADIANCE_MAP_H
#define RADIANCE_MAP_H

#include <vector>
#include <glm/glm.hpp>
#include "radiance_volume.h"
#include "ray.h"
#include "radiance_tree.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Defines a single radiance volume in the scene as described in:
    https://www.researchgate.net/publication/3208677_The_Irradiance_Volume
*/
class RadianceMap{

    private:
        RadianceTree* radiance_tree;
        void uniformly_sample_radiance_volumes(vector<RadianceVolume>& radiance_volumes, Surface surface);

    public:

        // Constructor
        RadianceMap(vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes);

        // Builds all RadianceVolumes which are part of the RadianceMap into the scene
        void build_radiance_map_shapes(vector<RadianceVolume>& radiance_volumes, vector<Surface>& surfaces);

        // Get the radiance estimate for every radiance volume in the RadianceMap
        void get_radiance_estimates(vector<RadianceVolume>& radiance_volumes, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes);

        // Get radiance estimate at the intersection point
        vec3 get_irradiance_estimate(const Intersection& intersection, vector<Surface *> surfaces);

        // Get global radiance tree pointer
        RadianceTree* get_global_radiance_tree_pointer();
};

#endif