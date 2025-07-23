#pragma once

#include <vk_types.h>
#include <filesystem>
#include <unordered_map>

struct GeoSurface {
    uint32_t startIndex;
    uint32_t count;
};

struct MeshAsset {
    std::string name;

    // Added a transform matrix to store the mesh's world matrix.
    glm::mat4 transform;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

class VulkanEngine;

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, const std::filesystem::path& filePath);
