#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

class Camera {

    private:
        
        mat4 look_at(vec3 from, vec3 to);

    public:
        vec4 position;
        mat4 R;
        float yaw;

        // Constructor
        Camera(vec4 position);

        // Camera rotation
        void rotate_left(float y);
        void rotate_right(float y);

        // Camera translation
        void move_forwards(float distance);
        void move_backwards(float distance);

};
#endif