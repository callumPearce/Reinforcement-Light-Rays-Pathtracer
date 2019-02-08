#ifndef RAY_H
#define RAY_H

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>

class Shape;
class Triangle;

using namespace std;
using glm::vec3;
using glm::vec2;
using glm::mat3;
using glm::vec4;
using glm::mat4;

/*
    Defines an intersection point for a ray
*/
struct Intersection {
    vec4 position;
    vec2 distances;
    vec4 normal;
    vec2 indices;
};


/*
    Defines a ray, which is a 4D vector (Homogeneous coordinates)
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
        bool closest_intersection(vector<Shape *> shapes, Intersection& closest_intersection);

        // Cramers rule for solving a system of linear equations
        bool cramers(mat3 A, vec3 b, vec3& solution);

        // Rotate a ray by "yaw"
        void rotate_ray(float yaw);

        // Getters
        vec4 get_start();
        vec4 get_direction();

        // Setters
        void set_start(vec4 start);
        void set_direction(vec4 direction);
};

#endif