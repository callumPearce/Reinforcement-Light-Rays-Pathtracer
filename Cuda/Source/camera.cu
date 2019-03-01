#include "camera.cuh"

Camera::Camera(vec4 position) {
    set_position(position);
    set_yaw(0);
    set_R(mat4(1.0));
}

void Camera::rotate_left(float yaw) {
    set_yaw(get_yaw() - yaw);
    mat4 newR = get_R();
    newR[0] = vec4(cos(-yaw), 0, sin(-yaw), 0);
    newR[2] = vec4(-sin(-yaw), 0, cos(-yaw), 0);
    set_R(newR);
    set_position(get_R() * get_position());
}

void Camera::rotate_right(float yaw) {
    set_yaw(get_yaw() + yaw);
    mat4 newR = get_R();
    newR[0] = vec4(cos(yaw), 0, sin(yaw), 0);
    newR[2] = vec4(-sin(yaw), 0, cos(yaw), 0);
    set_R(newR);
    set_position(get_R() * get_position());
}   

void Camera::move_forwards(float distance) {
    vec4 position = get_position();
    vec3 new_camera_pos(
        position[0] - distance * sin(this->yaw),
        position[1],
        position[2] + distance * cos(this->yaw)
        );
    mat4 cam_to_world = look_at(new_camera_pos, vec3(0, 0, 0));
    set_position(cam_to_world * vec4(0, 0, 0, 1));
}

void Camera::move_backwards(float distance) {
    vec4 position = get_position();
    vec3 new_camera_pos(
        position[0] + distance * sin(this->yaw),
        position[1],
        position[2] - distance * cos(this->yaw)
        );
    mat4 cam_to_world = look_at(new_camera_pos, vec3(0, 0, 0));
    set_position(cam_to_world * vec4(0, 0, 0, 1));
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

// Getters
__host__ __device__
vec4 Camera::get_position() {
    return position;
}

mat4 Camera::get_R() {
    return this->R;
}

__host__ __device__
float Camera::get_yaw() {
    return this->yaw;
}

// Setters
void Camera::set_position(vec4 position) {
    this->position = position;
}

void Camera::set_R(mat4 R) {
    this->R = R;
}

void Camera::set_yaw(float yaw) {
    this->yaw = yaw;
}
