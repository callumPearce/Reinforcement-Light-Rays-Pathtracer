#ifndef LIGHT_H
#define LIGHT_H

#include <glm/glm.hpp>
#include <vector>

#include "shape.h"
#include "ray.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

class Light{

    private:
        vec3 diffuse_p;
        vec4 position;

    public:
        // Constructor
        Light(vec4 position, vec3 diffuse_p);

        // Light Functions
        vec3 direct_light(const Intersection& i, vector<Shape *> shapes);

        // Movement

        // Getters
        vec4 get_position();
        vec3 get_diffuse_p();

        // Setters
        void set_position(vec4 position);
        void set_diffuse_p(vec3 diffuse_p);
};

#endif