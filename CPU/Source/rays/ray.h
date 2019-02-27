#ifndef RAY_H
#define RAY_H

#include <glm/glm.hpp>
#include <string>
#include <memory>
#include <vector>
#include "camera.h"
#include "image_settings.h"

class Shape;
class Triangle;
class Surface;
class AreaLightPlane;

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
    AREA_LIGHT_PLANE,
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
        vec4 start;
        vec4 direction;

    public:
        // Constructor
        Ray(vec4 start, vec4 direction);

        // Find the closest intersection for the given ray with an shape in the scene
        void closest_intersection(std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes, Intersection& closest_intersection);

        // Cramers rule for solving a system of linear equations
        bool cramers(mat3 A, vec3 b, vec3& solution);

        // Rotate a ray by "yaw"
        void rotate_ray(float yaw);

        // Sample a ray which passes through the pixel at the specified coordinates from the camera
        static Ray sample_ray_through_pixel(Camera& camera, int pixel_x, int pixel_y);

        // Getters
        vec4 get_start();
        vec4 get_direction();

        // Setters
        void set_start(vec4 start);
        void set_direction(vec4 direction);
};

#endif