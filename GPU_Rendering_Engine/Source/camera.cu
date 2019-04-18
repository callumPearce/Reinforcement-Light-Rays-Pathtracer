#include "camera.cuh"

Camera::Camera(vec4 position) {
    this->position = position;
    this->yaw_y = 0.f;
    this->R = mat4(1.0);
}

void Camera::rotate_left(float y) {
    this->yaw_y = this->yaw_y + y;
    mat4 newR = this->R;
    newR[0] = vec4(cos(y), 0, sin(y), 0);
    newR[2] = vec4(-sin(y), 0, cos(y), 0);
    this->R = newR;
    this->position = this->R * this->position;
}

void Camera::rotate_right(float y) {
    this->yaw_y = this->yaw_y - y;
    mat4 newR = this->R;
    newR[0] = vec4(cos(-y), 0, sin(-y), 0);
    newR[2] = vec4(-sin(-y), 0, cos(-y), 0);
    this->R = newR;
    this->position = this->R * this->position;
}

void Camera::rotate_up(float x) {
    this->yaw_x = this->yaw_x - x;
    mat4 newR = this->R;
    newR[0] = vec4(1.f, 0, 0, 0);
    newR[1] = vec4(0, cos(-x), -sin(-x), 0);
    newR[2] = vec4(0, sin(-x), cos(-x), 0);
    this->R = newR;
    this->position = this->R * this->position;
}

void Camera::rotate_down(float x) {
    this->yaw_x = this->yaw_x + x;
    mat4 newR = this->R;
    newR[0] = vec4(1.f, 0, 0, 0);
    newR[1] = vec4(0, cos(x), -sin(x), 0);
    newR[2] = vec4(0, sin(x), cos(x), 0);
    this->R = newR;
    this->position = this->R * this->position;
}

void Camera::move_forwards(float distance) {
    vec4 pos = this->position;
    vec3 new_camera_pos(
        pos[0] - distance * sin(this->yaw_y),
        pos[1],
        pos[2] + distance * cos(this->yaw_y)
        );
    mat4 cam_to_world = look_at(new_camera_pos, vec3(0, 1.f, 0));
    this->position = cam_to_world * vec4(0, 0, 0, 1);
}

void Camera::move_backwards(float distance) {
    vec4 pos = this->position;
    vec3 new_camera_pos(
        pos[0] + distance * sin(this->yaw_y),
        pos[1],
        pos[2] - distance * cos(this->yaw_y)
        );
    mat4 cam_to_world = look_at(new_camera_pos, vec3(0, 1.f, 0));
    this->position = cam_to_world * vec4(0, 0, 0, 1);
}

mat4 Camera::look_at(vec3 from, vec3 to) {
    vec3 forward = normalize(from - to);
    vec3 temp(0, 1, 0);
    vec3 right = cross(normalize(temp), forward);
    vec3 up = cross(forward, right);

    vec4 forward4(forward.x, forward.y, forward.z, 0);
    vec4 right4(right.x, right.y, right.z, 0);
    vec4 up4(up.x, up.y, up.z, 0);
    vec4 from4(from.x, from.y, from.z, 1);

    mat4 cam_to_world(right4, up4, forward4, from4);

    return cam_to_world;
}