#ifndef RADIANCE_MAP_H
#define RADIANCE_MAP_H

#include <vector>
#include <glm/glm.hpp>
#include "radiance_volume.h"
#include "ray.h"

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
        vector<RadianceVolume> radiance_volumes;

        void uniformly_sample_radiance_volumes(Surface surface);

    public:

        // Constructor
        RadianceMap(vector<Surface *> surfaces);

        // Builds all RadianceVolumes which are part of the RadianceMap into the scene
        void build_radiance_map_shapes(vector<Surface>& surfaces);

        // Get the radiance estimate for every radiance volume in the RadianceMap
        void get_radiance_estimates(vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes);

        // Get radiance estimate at the intersection point
        vec3 get_radiance_estimate(const Intersection& intersection, vector<Surface *> surfaces);
};

#endif