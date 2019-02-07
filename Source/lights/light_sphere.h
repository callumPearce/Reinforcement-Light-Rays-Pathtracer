#ifndef LIGHTSPHERE_H
#define LIGHTSPHERE_H

#include <glm/glm.hpp>
#include <vector>

#include "ray.h"
#include "light.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

class LightSphere {

    private:
        vector<Light> point_lights;
        vec4 centre;
        float radius;
        vec3 diffuse_p;
        vec3 ambient_p;

        // Generate the point lights to go in the sphere
        void create_lights(int sample_count);

        // Test whether a point is in a sphere or not
        bool contained_in_sphere(vec4 point);

    public:
        // Constructor
        LightSphere(vec4 centre, float radius, int num_lights, vec3 diffuse_p, vec3 ambient_p);

        // Return the light for a given interesection point contributed to by the lightsphere
        vec3 get_intersection_radiance(Intersection& intersection, vector<Shape *> shapes);

        // Movement
        void translate_left(float distance);
        void translate_right(float distance);
        void translate_forwards(float distance);
        void translate_backwards(float distance);
        void translate_up(float distance);
        void translate_down(float distance);

        // Getters
        vector<Light> get_point_lights();
        vec4 get_centre();
        float get_radius();
        vec3 get_diffuse_p();
        vec3 get_ambient_p();

        // Setters
        void set_point_lights(vector<Light> point_lights);
        void set_centre(vec4 centre);
        void set_radius(float r);
        void set_diffuse_p(vec3 diffuse_p);
        void set_ambient_p(vec3 ambient_p);
};

#endif