#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

class Camera {

    private:
        vec4 position;
        mat4 R;
        float yaw;
        
        mat4 look_at(vec3 from, vec3 to);

    public:
        // Constructor
        Camera(vec4 position);

        // Camera rotation
        void rotate_left(float yaw);
        void rotate_right(float yaw);

        // Camera translation
        void move_forwards(float distance);
        void move_backwards(float distance);

        // Getters
        vec4 get_position();
        mat4 get_R();
        float get_yaw();

        // Setters
        void set_position(vec4 position);
        void set_R(mat4 R);
        void set_yaw(float yaw);

};
#endif