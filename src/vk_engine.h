// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

#include "vk_descriptors.h"

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
};

struct ComputePushConstants {
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect {
    const char* name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    ComputePushConstants data;
};


constexpr unsigned int FRAME_OVERLAP = 2;


class VulkanEngine {
public:

    bool _isInitialized{false};
    int _frameNumber{0};
    bool stop_rendering{false};
    VkExtent2D _windowExtent{900, 550};
    struct SDL_Window *_window{nullptr};
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
    uint32_t _graphicsQueueFamily;

    VmaAllocator _allocator;
    //draw resources
    AllocatedImage _drawImage;
    VkExtent2D _drawExtent;
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
    float _dpiScale = 1.0;


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

    void draw_imgui(VkCommandBuffer cmd, VkImageView targetView);


private:
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    void init_desciptors();
    void init_pipelines();
    void init_background_pipelines();
    void init_imgui();


};
