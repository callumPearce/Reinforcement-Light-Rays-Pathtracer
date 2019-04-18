#ifndef RAY_H
#define RAY_H

#include <glm/glm.hpp>
#include <string>
#include <memory>
#include <vector>
#include "camera.cuh"
#include "image_settings.h"
//cuRand
#include <curand.h>
#include <curand_kernel.h>


class AreaLight;
class Surface;
class Triangle;
class Scene;
// #include "scene.cuh"

using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Define all the object types a ray can intersect with
*/
enum IntersectionType {
    NOTHING,
    AREA_LIGHT,
    SURFACE
};

/*
    Defines an intersection point for a ray
*/
struct Intersection {
    vec4 position;
    float distance;
    vec4 normal;
    int index;
    IntersectionType intersection_type = NOTHING;
};

/*
    Defines a ray, which is a 4D std::vector (Homogeneous coordinates)
    with a direcion and starting position
*/
class Ray {

    private:
        __device__
        static bool cramer(mat3 A, vec3 b, vec3& solution);

        __device__
        bool intersects(int index, Surface* surfaces);

        __device__
        bool intersects(int index, AreaLight* area_lights);

    public:
        vec4 start;
        vec4 direction;
        Intersection intersection;

        // Constructor
        __device__
        Ray(vec4 start, vec4 direction);

        // Find the closest intersection for the given ray with an shape in the scene
        __device__
        void closest_intersection(Scene* scene);

        // Cramers rule for solving a system of linear equations
        __device__
        bool cramers(mat3 A, vec3 b, vec3& solution);
        
        // Sample a ray which passes through the pixel at the specified coordinates from the camera
        __device__
        static Ray sample_ray_through_pixel(curandState* d_rand_state, Camera& camera, int pixel_x, int pixel_y);

        // Rotate a ray by "yaw"
        __device__
        void rotate_ray(float yaw_y, float yaw_x);

};

#endif