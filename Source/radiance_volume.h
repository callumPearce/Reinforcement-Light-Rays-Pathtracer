#ifndef RADIANCE_VOLUME_H
#define RADIANCE_VOLUME_H

#include <glm/glm.hpp>

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Defines a single radiance volume in the scene as described in:
    https://www.researchgate.net/publication/3208677_The_Irradiance_Volume
*/
class RadianceVolume{

    private:
        vec4 position;
        vec4 normal;

    public:
        // Constructor
        RadianceVolume(vec4 position, vec4 normal);

        // Getters
        vec4 get_position();
        vec4 get_normal();

        // Setters
        void set_position(vec4 position);
        void set_normal(vec4 normal);
};

#endif