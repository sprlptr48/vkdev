// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <glm/fwd.hpp>

#include "vk_descriptors.h"
#include "vk_loader.h"
#include "camera.h"

#define VK_DEBUG


struct DeletionQueue    // TODO: use array or somn mabye idk this is ugly
{
    std::deque<std::function<void()>> deletors;
    void push_function(std::function<void()>&& function) {
        deletors.push_back(function);
    }
    void flush() {
        // reverse iterate the deletion queue to execute all the functions
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)(); //call functors
        }
        deletors.clear();
    }
};

struct FrameData {
    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;
    VkSemaphore _swapchainSemaphore, _renderSemaphore;
    VkFence _renderFence;
    DeletionQueue _deletionQueue;
    DescriptorAllocatorGrowable _frameDescriptors;
};

struct ComputePushConstants {
    glm::mat4 viewMatrix;
    glm::uint32 time;
};

struct ComputeEffect {
    const char* name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    ComputePushConstants data;
};

struct GPUSceneData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection; // w for sun power
    glm::vec4 sunlightColor;
};

constexpr unsigned int FRAME_OVERLAP = 2;


class VulkanEngine {
public:

    bool _isInitialized{false};
    bool stop_rendering{false};
    int _frameNumber{0};
    struct SDL_Window *_window{nullptr};
    VkExtent2D _windowExtent{1920, 1080};
    VkExtent2D _requestedExtent{800, 600};
    DeletionQueue _mainDeletionQueue;
    VkInstance _instance;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;                          // Vulkan Device for commands
#ifdef VK_DEBUG
    VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
#endif
    VkSurfaceKHR _surface;                     // Vulkan window surface
    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;
    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent;

    FrameData _frames[FRAME_OVERLAP];
    FrameData& get_current_frame() {  return _frames[_frameNumber % FRAME_OVERLAP]; };

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily = -1;

    VmaAllocator _allocator;
    //draw resources
    AllocatedImage _drawImage;
    AllocatedImage _depthImage;
    VkExtent2D _drawExtent;
    float _renderScale = 1.f;
    DescriptorAllocator globalDescriptorAllocator;
    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;
    VkPipeline _gradientPipeline;
    VkPipelineLayout _gradientPipelineLayout;

    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    std::vector<ComputeEffect> _backgroundEffects;
    int currentBackgroundEffect{0};

    VkPipelineLayout _trianglePipelineLayout;
    VkPipeline _trianglePipeline;

    VkPipelineLayout _meshPipelineLayout;
    VkPipeline _meshPipeline;
    // Dynamically load meshes
    std::vector<std::shared_ptr<MeshAsset>> _loadedMeshes;
    int _selectedMeshIndex = -1;

    std::chrono::high_resolution_clock::time_point lastTime;
    double totalTime = 0; // seconds
    uint64_t totalFrames = 0;

    GPUSceneData sceneData;

    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    VkSampler _defaultSamplerLinear;
    VkSampler _defaultSamplerNearest;

    VkDescriptorSetLayout _singleImageDescriptorLayout;
    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    Camera _camera;
    bool _mouseCaptured = false;

    bool resize_requested{false};

    void init_default_data();
    void init_mesh_pipeline();


    static VulkanEngine &Get();

    //initializes everything in the engine
    void init();

    //shuts down the engine
    void cleanup();

    //draw loop
    void draw();
    // additional draw stuff
    void draw_background(VkCommandBuffer cmd);

    //run main loop
    void run();

    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
    void draw_geometry(VkCommandBuffer cmd);

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroy_buffer(const AllocatedBuffer& buffer) const;
    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped=false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped=false);
    void destroy_image(const AllocatedImage& img);

private:
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();
    void create_swapchain(uint32_t width, uint32_t height);
    void resize_swapchain();
    void destroy_swapchain();
    void init_descriptors();
    void init_pipelines();
    void init_background_pipelines();
    void init_imgui();


};
