#ifndef SCENE_H
#define SCENE_H

#include <glm/glm.hpp>
#include "surface.cuh"
#include "area_light.cuh"
#include "cornell_box_scene.cuh"
#include "ray.cuh"

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;


/*
    Defines a scene of geometry. This makes it easier to
    pass the scene around the engine as a single entity 
    which can be accessed.
*/
class Scene{

    private:

    public:
        Surface* surfaces;
        int surfaces_count;
        AreaLight* area_lights;
        int area_light_count;

        Scene();

        void load_cornell_box_scene();

};

#endif