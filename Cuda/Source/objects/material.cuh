#ifndef MATERIAL_H
#define MATERIAL_H

#include <glm/glm.hpp>

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Defines a material which every object in the scene will own.
    This describes rendering attributes such as diffuse colour,
    specular colour and many more.
*/

class Material{

    private:
    
    public:
        vec3 diffuse_c;

        //Constructor
        Material(vec3 diffuse_c);
};

#endif