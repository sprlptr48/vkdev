#version 460

// Input from vertex shader
layout(location = 0) in vec4 inColor;

// Output to the framebuffer
layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = inColor;
}
