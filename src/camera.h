#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <cstdint>

// NEW: A self-contained camera class to handle view matrix and user input.
class Camera {
public:
    glm::vec3 position{0.0f, 1.0f, -5.0f};
    float speed = 5.0f; // units per second

    // Orientation in degrees
    float pitch = 0.0f; // Up/down
    float yaw = 90.0f;  // Left/right. 90 degrees starts facing towards +Z
    float mouseSensitivity = 0.1f;

    // Processes keyboard state to move the camera
    void process_keyboard(const uint8_t* state, float deltaTime);

    // Processes mouse movement to change camera orientation
    void process_mouse_movement(float xoffset, float yoffset, bool constrainPitch = true);

    // Returns the view matrix calculated from the camera's state
    glm::mat4 get_view_matrix();

private:
    // Recalculates front, right, and up vectors from pitch and yaw
    void update_camera_vectors();

    // Camera direction vectors
    glm::vec3 _front{0.0f, 0.0f, 1.0f};
    glm::vec3 _up{0.0f, 1.0f, 0.0f};
    glm::vec3 _right{1.0f, 0.0f, 0.0f};
    const glm::vec3 _worldUp{0.0f, 1.0f, 0.0f};
};
