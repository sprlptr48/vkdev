#include <vk_pipelines.h>
#include <fstream>
#include <vk_initializers.h>

bool vkutil::load_shader_module(const char* filePath, VkDevice device, VkShaderModule* out) {
    // open the file. With cursor at the end
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return false;
    }
    // find what the size of the file is by looking up the location of the cursor
    // because the cursor is at the end, it gives the size directly in bytes
    const size_t fileSize = (size_t)file.tellg();
    // spirv expects the buffer to be on uint32, so make sure to reserve a int
    // vector big enough for the entire file
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);
    file.close();
    // create a new shader file with the blob.
    VkShaderModuleCreateInfo SMCreateInfo = {};
    SMCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    SMCreateInfo.pNext = nullptr;

    SMCreateInfo.codeSize = sizeof(uint32_t) * buffer.size();
    SMCreateInfo.pCode = buffer.data();

    VkShaderModule shaderModule;
    if(vkCreateShaderModule(device, &SMCreateInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return false;
    }
    *out = shaderModule;
    return true;
}

