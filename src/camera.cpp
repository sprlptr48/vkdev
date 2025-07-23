#include "camera.h"
#include <SDL_scancode.h>
#include <glm/gtx/transform.hpp>
#include <algorithm>

// NEW: Implementation of the Camera class.

void Camera::process_keyboard(const uint8_t* state, float deltaTime) {
    const float moveAmount = speed * deltaTime;
    if (state[SDL_SCANCODE_W])
        position += _front * moveAmount;
    if (state[SDL_SCANCODE_S])
        position -= _front * moveAmount;
    if (state[SDL_SCANCODE_A])
        position -= _right * moveAmount;
    if (state[SDL_SCANCODE_D])
        position += _right * moveAmount;
    if (state[SDL_SCANCODE_SPACE])
        position += _worldUp * moveAmount;
    if (state[SDL_SCANCODE_LSHIFT])
        position -= _worldUp * moveAmount;
}

void Camera::process_mouse_movement(float xoffset, float yoffset, bool constrainPitch) {
    yaw += xoffset * mouseSensitivity;
    pitch += yoffset * mouseSensitivity;

    if (constrainPitch) {
        pitch = std::clamp(pitch, -89.0f, 89.0f);
    }
    update_camera_vectors();
}

glm::mat4 Camera::get_view_matrix() {
    // We call this here to ensure vectors are up-to-date before creating the matrix.
    // This is a safeguard if the camera is manipulated externally.
    update_camera_vectors();
    return glm::lookAt(position, position + _front, _up);
}

void Camera::update_camera_vectors() {
    glm::vec3 newFront;
    newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    newFront.y = sin(glm::radians(pitch));
    newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    _front = glm::normalize(newFront);

    _right = glm::normalize(glm::cross(_front, _worldUp));
    _up = glm::normalize(glm::cross(_right, _front));
}
