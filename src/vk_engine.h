// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

#define VK_DEBUG

class VulkanEngine {
public:

    bool _isInitialized{false};
    int _frameNumber{0};
    bool stop_rendering{false};
    VkExtent2D _windowExtent{900, 550};
    struct SDL_Window *_window{nullptr};


    VkInstance _instance;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;                          // Vulkan Device for commands
    VkSurfaceKHR _surface;                     // Vulkan window surface
    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent;
#ifdef VK_DEBUG
    VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
#endif

    static VulkanEngine &Get();

    //initializes everything in the engine
    void init();

    //shuts down the engine
    void cleanup();

    //draw loop
    void draw();

    //run main loop
    void run();

private:
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();


};
