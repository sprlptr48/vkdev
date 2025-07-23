#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// Output to fragment shader
layout(location = 0) out vec4 outColor;

// Matches the Vertex struct in vk_types.h
struct Vertex {
    vec3 position;
    float uvx;
    vec3 normal;
    float uvy;
    vec4 color;
};

// Define a buffer type using a buffer reference
layout(buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex vertices[];
};

// Matches GPUDrawPushConstants in vk_types.h
layout(push_constant) uniform constants {
    mat4 worldMatrix;
    VertexBuffer vertexBuffer; // The GPU address of the vertex buffer
} PushConstants;


void main() {
    // Pull the vertex data for the current vertex index from the buffer
    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

    // Standard model-view-projection transform
    gl_Position = PushConstants.worldMatrix * vec4(v.position, 1.0);

    // Visualize normals by remapping them from [-1, 1] to the [0, 1] color range
    outColor = vec4(v.normal * 0.5 + 0.5, 1.0);
}
