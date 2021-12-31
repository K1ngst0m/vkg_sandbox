// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vulkan/vulkan_core.h>

#include <deque>
#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "vk_mesh.h"
#include "vk_types.h"


struct Material {
    VkDescriptorSet textureSet{VK_NULL_HANDLE};
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct Texture{
    AllocatedImage image;
    VkImageView imageView;
};

struct IndirectBatch{
    Mesh* mesh;
    Material* material;
    uint32_t first;
    uint32_t count;
};

struct UploadContext{
    VkFence _uploadFence;
    VkCommandPool _commandPool;
};

struct GPUSceneData {
    glm::vec4 fogColor;
    glm::vec4 forgDistances;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection;
    glm::vec4 sunlightColor;
};

struct GPUCameraData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
};

struct GPUObjectData {
    glm::mat4 modelMatrix;
    glm::vec4 color;
};

constexpr unsigned int FRAME_OVERLAP = 2;
constexpr unsigned int MAX_COMMANDS = 10000;
struct FrameData {
    VkSemaphore _presentSemaphore, _renderSemaphore;
    VkFence _renderFence;

    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    VkDescriptorSet globalDescriptor;

    AllocatedBuffer objectTransformBuffer;
    AllocatedBuffer objectColorBuffer;
    VkDescriptorSet objectDescriptor;

    AllocatedBuffer indirectBuffer;
};

struct RenderObject {
    Mesh* mesh;
    Material* material;
    glm::mat4 transformMatrix;
};

struct MeshPushConstants {
    glm::vec4 data;
    glm::mat4 render_matrix;
};

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()>&& function) {
        deletors.push_back(function);
    }

    void flush() {
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)();
        }

        deletors.clear();
    }
};

class VulkanEngine {
public:
    void init();
    void cleanup();
    void draw();
    void run();

public:
    bool _isInitialized{false};
    uint32_t _frameNumber{0};
    VkExtent2D _windowExtent{1700, 900};
    struct SDL_Window* _window{nullptr};
    DeletionQueue _mainDeletionQueue;

public:
    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debug_messenger;
    VkPhysicalDevice _chosenGPU;
    VkPhysicalDeviceProperties _gpuProperties;
    VkDevice _device;
    VkSurfaceKHR _surface;

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    VmaAllocator _allocator;

public:
    // swap chains
    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;
    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;

    // depth buffer
    VkImageView _depthImageView;
    AllocatedImage _depthImage;
    VkFormat _depthFormat;

    VkRenderPass _renderPass;
    std::vector<VkFramebuffer> _framebuffers;

public:
    glm::vec3 _camPos = {0.f, -6.f, -10.f};
    std::vector<RenderObject> _renderables;
    std::unordered_map<std::string, Material> _materials;
    std::unordered_map<std::string, Mesh> _meshes;
    std::unordered_map<std::string, Texture> _textures;

    FrameData _frames[FRAME_OVERLAP];
    FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; }

public:
    VkDescriptorSetLayout _globalSetLayout;
    VkDescriptorSetLayout _objectSetLayout;
    VkDescriptorSetLayout _singleTextureSetLayout;
    VkDescriptorPool _descriptorPool;
    GPUSceneData _sceneParameters;
    AllocatedBuffer _sceneBuffer;
    UploadContext _uploadContext;

public:
    void load_meshes();
    void load_images();
    void upload_mesh(Mesh& mesh);
    Material* create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);
    Material* get_material(const std::string& name);
    Mesh* get_mesh(const std::string& name);

    void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);
    void immediate_submit(std::function<void(VkCommandBuffer cmd)> && function);
    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, bool addDeletionQueue = true);

private:
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_default_renderpass();
    void init_framebuffers();
    void init_sync_structures();
    void init_pipelines();
    void init_scene();
    void init_descriptors();

private:
    bool load_shader_module(const char* filePath, VkShaderModule* outShaderModule);
    size_t pad_uniform_buffer_size(size_t originalSize);

};

class PipelineBuilder {
public:
    std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
    VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
    VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
    VkViewport _viewport;
    VkRect2D _scissor;
    VkPipelineRasterizationStateCreateInfo _rasterizer;
    VkPipelineColorBlendAttachmentState _colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo _multisampling;
    VkPipelineLayout _pipelineLayout;
    VkPipelineDepthStencilStateCreateInfo _depthStencil;

    VkPipeline build_pipeline(VkDevice, VkRenderPass pass);
};

namespace vkutil {
    std::vector<IndirectBatch> compact_draws(RenderObject* objects, int count);
}
