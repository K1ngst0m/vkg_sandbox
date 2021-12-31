#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>

#include <fstream>
#include <glm/gtx/transform.hpp>
#include <iostream>

#include "vk_initializers.h"
#include "vk_textures.h"
#include "vk_types.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            std::cout << "Detected Vulkan error: " << err << std::endl; \
            abort();                                                    \
        }                                                               \
    } while (0)

std::vector<IndirectBatch> vkutil::compact_draws(RenderObject* objects, int count) {
    std::vector<IndirectBatch> draws;
    IndirectBatch firstDraw;
    firstDraw.mesh = objects[0].mesh;
    firstDraw.material = objects[0].material;
    firstDraw.first = 0;
    firstDraw.count = 1;

    draws.push_back(firstDraw);

    for (int i = 0; i < count; i++) {
        bool sameMesh = objects[i].mesh == draws.back().mesh;
        bool sameMaterial = objects[i].material == draws.back().material;

        if (sameMesh && sameMaterial) {
            draws.back().count++;
        } else {
            // add new draw
            IndirectBatch newDraw;
            newDraw.mesh = objects[i].mesh;
            newDraw.material = objects[i].material;
            newDraw.first = i;
            newDraw.count = 1;

            draws.push_back(newDraw);
        }
    }

    return draws;
}

void VulkanEngine::init() {
    // init SDL
    SDL_Init(SDL_INIT_VIDEO);
    SDL_WindowFlags window_flags = (SDL_WindowFlags) (SDL_WINDOW_VULKAN);
    _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                               _windowExtent.width, _windowExtent.height,
                               window_flags);

    // init vulkan
    init_vulkan();
    init_swapchain();
    init_commands();
    init_default_renderpass();
    init_framebuffers();
    init_sync_structures();
    init_descriptors();
    init_pipelines();
    load_meshes();
    load_images();
    init_scene();

    _isInitialized = true;
}

void VulkanEngine::cleanup() {
    if (_isInitialized) {
        vkDeviceWaitIdle(_device);
        _mainDeletionQueue.flush();

        vkDestroyDevice(_device, nullptr);
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }
}

void VulkanEngine::draw() {
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    uint32_t swapchainImageIndex;
    VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._presentSemaphore, nullptr, &swapchainImageIndex));

    VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));
    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    // begin command recording
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // start the main renderpass
    VkRenderPassBeginInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.pNext = nullptr;
    rpInfo.renderPass = _renderPass;
    rpInfo.renderArea.offset.x = 0;
    rpInfo.renderArea.offset.y = 0;
    rpInfo.renderArea.extent = _windowExtent;
    rpInfo.framebuffer = _framebuffers[swapchainImageIndex];
    VkClearValue colorClear, depthClear;
    float flash = abs(sin(_frameNumber / 120.f));
    colorClear.color = {{0.2f, 0.2f, flash, 1.0f}};
    depthClear.depthStencil.depth = 1.0f;
    VkClearValue clearValues[] = {colorClear, depthClear};
    rpInfo.clearValueCount = 2;
    rpInfo.pClearValues = clearValues;
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    // draw objects
    draw_objects(cmd, _renderables.data(), _renderables.size());

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    // prepare the submission to the queue
    VkSubmitInfo submitInfo = vkinit::submit_Info(&cmd, 1, &get_current_frame()._presentSemaphore, &get_current_frame()._renderSemaphore);
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    submitInfo.pWaitDstStageMask = &waitStage;
    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submitInfo, get_current_frame()._renderFence));

    // present image
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
    presentInfo.pImageIndices = &swapchainImageIndex;
    VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

    // increase the number of frames drawn
    _frameNumber++;
}

void VulkanEngine::run() {
    SDL_Event e;
    bool bQuit = false;

    float camSpeed = 0.02f;
    // main loop
    while (!bQuit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;
            else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        bQuit = true;
                        std::cout << "quit" << std::endl;
                        break;
                    case SDLK_w:
                        _camPos += glm::vec3(0.0f, camSpeed, 0.0f);
                        break;
                    case SDLK_a:
                        _camPos += glm::vec3(camSpeed, 0.0f, 0.0f);
                        break;
                    case SDLK_s:
                        _camPos += glm::vec3(0.0f, -camSpeed, 0.0f);
                        break;
                    case SDLK_d:
                        _camPos += glm::vec3(-camSpeed, 0.0f, 0.0f);
                        break;
                    default:
                        break;
                }
            }
        }

        draw();
    }
}

void VulkanEngine::init_vulkan() {
    vkb::InstanceBuilder builder;
    auto inst_ret = builder.set_app_name("demo")
                            .request_validation_layers(true)
                            // .require_api_version(1, 2, 0)
                            .require_api_version(1, 1, 0)
                            .use_default_debug_messenger()
                            .build();

    vkb::Instance vkb_inst = inst_ret.value();
    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    // get the surface of the window we opened with SDL
    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    // use vkbootstrap to select a GPU
    // We want a GPU that can write to the SDL surface and supports Vulkan 1.1
    vkb::PhysicalDeviceSelector selector{vkb_inst};
    VkPhysicalDeviceFeatures gpuFeatures = {};
    gpuFeatures.multiDrawIndirect = VK_TRUE;
    selector.set_required_features(gpuFeatures);
    selector.add_required_extension("VK_KHR_shader_draw_parameters");
    vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(1, 1).set_surface(_surface).select().value();

    // create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    // get the vkdevice handle used in the rest of a vulkan application
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();

    // initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([=]() {
        vmaDestroyAllocator(_allocator);
    });

    vkGetPhysicalDeviceProperties(_chosenGPU, &_gpuProperties);
    // VkPhysicalDeviceFeatures gpuFeatures;
    // vkGetPhysicalDeviceFeatures(_chosenGPU, &gpuFeatures);
    // if(gpuFeatures.multiDrawIndirect == VK_TRUE){
    //     std::cout << "The GPU supports multi draw indirect features" << std::endl;
    //     exit(EXIT_SUCCESS);
    // }
    // std::cout << "The GPU supports " << _gpuProperties.limits.maxDrawIndirectCount << " maxDrawIndirectCount." << std::endl;
    // std::cout << "The GPU has a minimum buffer alignment of " << _gpuProperties.limits.minUniformBufferOffsetAlignment << std::endl;
}

void VulkanEngine::init_swapchain() {
    vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};

    vkb::Swapchain vkbSwapchain = swapchainBuilder
                                          .use_default_format_selection()
                                          // use vsync present mode
                                          .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                                          .set_desired_extent(_windowExtent.width, _windowExtent.height)
                                          .build()
                                          .value();

    // store swapchian and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();

    _swapchainImageFormat = vkbSwapchain.image_format;

    _mainDeletionQueue.push_function([=]() {
        vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    });

    // depth buffer
    VkExtent3D depthImageExtent = {
            _windowExtent.width,
            _windowExtent.height,
            1};

    _depthFormat = VK_FORMAT_D32_SFLOAT;
    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

    VmaAllocationCreateInfo dimg_allocinfo = {};
    dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image, &_depthImage._allocation, nullptr);

    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);
    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _depthImageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
    });
}

void VulkanEngine::init_commands() {
    // main command pool
    VkCommandPoolCreateInfo mainCommandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateCommandPool(_device, &mainCommandPoolInfo, nullptr, &_frames[i]._commandPool));
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
        _mainDeletionQueue.push_function([=]() {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
        });
    }

    // upload command pool
    VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
    VK_CHECK(vkCreateCommandPool(_device, &uploadCommandPoolInfo, nullptr, &_uploadContext._commandPool));
    _mainDeletionQueue.push_function([=]() {
        vkDestroyCommandPool(_device, _uploadContext._commandPool, nullptr);
    });

    // indirect command buffer
    for(int i = 0; i < FRAME_OVERLAP; i++){
        _frames[i].indirectBuffer = create_buffer(MAX_COMMANDS * sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    }
}

void VulkanEngine::init_default_renderpass() {
    // the renderpass will use this color attachment;
    VkAttachmentDescription color_attachment = {};

    color_attachment.format = _swapchainImageFormat;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref = {};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depth_attachment = {};
    depth_attachment.flags = 0;
    depth_attachment.format = _depthFormat;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_attachment_ref = {};
    depth_attachment_ref.attachment = 1;
    depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;

    VkAttachmentDescription attachments[2] = {color_attachment, depth_attachment};

    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

    // connect the color attachment to the info
    render_pass_info.attachmentCount = 2;
    render_pass_info.pAttachments = &attachments[0];
    // connect the subpass to the info
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyRenderPass(_device, _renderPass, nullptr);
    });
}

void VulkanEngine::init_framebuffers() {
    VkFramebufferCreateInfo fb_info = {};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;

    fb_info.renderPass = _renderPass;
    fb_info.attachmentCount = 1;
    fb_info.width = _windowExtent.width;
    fb_info.height = _windowExtent.height;
    fb_info.layers = 1;

    // grab how many images we have in the swapchain
    const uint32_t swapchain_imagecount = _swapchainImages.size();
    _framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

    // create framebuffers for each of the swapchain image views
    for (int i = 0; i < swapchain_imagecount; i++) {
        VkImageView attachments[2];
        attachments[0] = _swapchainImageViews[i];
        attachments[1] = _depthImageView;

        fb_info.attachmentCount = 2;
        fb_info.pAttachments = attachments;
        VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

        _mainDeletionQueue.push_function([=]() {
            vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
            vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
        });
    }
}

void VulkanEngine::init_sync_structures() {
    // render fence and semaphore
    VkFenceCreateInfo renderFenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();
    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateFence(_device, &renderFenceCreateInfo, nullptr, &_frames[i]._renderFence));

        _mainDeletionQueue.push_function([=]() {
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
        });

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._presentSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));

        _mainDeletionQueue.push_function([=]() {
            vkDestroySemaphore(_device, _frames[i]._presentSemaphore, nullptr);
            vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
        });
    }

    // upload fence
    VkFenceCreateInfo uploadFenceCreateInfo = vkinit::fence_create_info();
    VK_CHECK(vkCreateFence(_device, &uploadFenceCreateInfo, nullptr, &_uploadContext._uploadFence));
    _mainDeletionQueue.push_function([=]() {
        vkDestroyFence(_device, _uploadContext._uploadFence, nullptr);
    });
}

bool VulkanEngine::load_shader_module(const char* filePath, VkShaderModule* outShaderModule) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return false;
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

    file.close();

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;

    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode = buffer.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return false;
    }

    *outShaderModule = shaderModule;
    return true;
}

void VulkanEngine::init_pipelines() {
    PipelineBuilder pipelineBuilder;

    // vertex input
    pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();
    VertexInputDescription vertexDescription = Vertex::get_vertex_description();
    pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
    pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();
    pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
    pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();
    pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    // viewport
    pipelineBuilder._viewport.x = 0.0f;
    pipelineBuilder._viewport.y = 0.0f;
    pipelineBuilder._viewport.width = static_cast<float>(_windowExtent.width);
    pipelineBuilder._viewport.height = static_cast<float>(_windowExtent.height);
    pipelineBuilder._viewport.minDepth = 0.0f;
    pipelineBuilder._viewport.maxDepth = 1.0f;
    pipelineBuilder._scissor.offset = {0, 0};
    pipelineBuilder._scissor.extent = _windowExtent;

    // rasterization
    pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);
    pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
    pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();
    pipelineBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();

    // shader stages
    VkShaderModule meshVertShader;
    if (!load_shader_module("shaders/tri_mesh.vert.spv", &meshVertShader)) {
        std::cout << "Error when building the triangle vertex shader module" << std::endl;
    } else {
        std::cout << "Triangle mesh vertex shader successfully loaded" << std::endl;
    }
    VkShaderModule triangleFragShader;
    if (!load_shader_module("shaders/colored_triangle.frag.spv", &triangleFragShader)) {
        std::cout << "Error when building the triangle fragment shader module" << std::endl;
    } else {
        std::cout << "Triangle fragment shader successfully loaded" << std::endl;
    }
    VkShaderModule colorMeshShader;
    if (!load_shader_module("shaders/default_lit.frag.spv", &colorMeshShader)) {
        std::cout << "Error when building the colored mesh shader" << std::endl;
    }
    VkShaderModule texturedMeshShader;
    if (!load_shader_module("shaders/textured_lit.frag.spv", &texturedMeshShader)) {
        std::cout << "Error when building the textured mesh shader" << std::endl;
    }

    // pipeline layout
    VkPipelineLayout defaultMeshPipelineLayout;
    VkDescriptorSetLayout setLayouts[] = {_globalSetLayout, _objectSetLayout};
    VkPushConstantRange push_constant;
    push_constant.offset = 0;
    push_constant.size = sizeof(MeshPushConstants);
    push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkPipelineLayoutCreateInfo mesh_pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 2, &push_constant, 1);
    VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_info, nullptr, &defaultMeshPipelineLayout));
    pipelineBuilder._pipelineLayout = defaultMeshPipelineLayout;

    // default mesh pipeline
    pipelineBuilder._shaderStages.clear();
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));
    VkPipeline defaultMeshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);
    create_material(defaultMeshPipeline, defaultMeshPipelineLayout, "defaultMesh");

    // default light pipeline
    pipelineBuilder._shaderStages.clear();
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, colorMeshShader));
    VkPipeline defaultLightPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);
    create_material(defaultLightPipeline, defaultMeshPipelineLayout, "defaultLightMesh");

    // textured pipeline
    VkPipelineLayoutCreateInfo textured_pipeline_layout_info = mesh_pipeline_layout_info;
    VkDescriptorSetLayout texturedSetLayouts[] = {_globalSetLayout, _objectSetLayout, _singleTextureSetLayout};
    textured_pipeline_layout_info.setLayoutCount = 3;
    textured_pipeline_layout_info.pSetLayouts = texturedSetLayouts;
    VkPipelineLayout texturedPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(_device, &textured_pipeline_layout_info, nullptr, &texturedPipelineLayout));
    pipelineBuilder._pipelineLayout = texturedPipelineLayout;
    pipelineBuilder._shaderStages.clear();
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, texturedMeshShader));
    VkPipeline texPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);
    create_material(texPipeline, texturedPipelineLayout, "texturedMesh");

    // destroy all shader modules, outside of the queue
    vkDestroyShaderModule(_device, meshVertShader, nullptr);
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, colorMeshShader, nullptr);
    vkDestroyShaderModule(_device, texturedMeshShader, nullptr);

    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipeline(_device, defaultMeshPipeline, nullptr);
        vkDestroyPipeline(_device, defaultLightPipeline, nullptr);
        vkDestroyPipeline(_device, texPipeline, nullptr);

        vkDestroyPipelineLayout(_device, defaultMeshPipelineLayout, nullptr);
        vkDestroyPipelineLayout(_device, texturedPipelineLayout, nullptr);
    });
}

void VulkanEngine::load_meshes() {
    Mesh triangleMesh;

    // triangle
    // triangleMesh._vertices.resize(3);
    // triangleMesh._vertices[0].position = {1.f, 1.f, 0.0f};
    // triangleMesh._vertices[1].position = {-1.f, 1.f, 0.0f};
    // triangleMesh._vertices[2].position = {0.f, -1.f, 0.0f};
    // triangleMesh._vertices[0].color = {0.f, 1.f, 0.0f};//pure green
    // triangleMesh._vertices[1].color = {1.f, 0.f, 0.0f};//pure green
    // triangleMesh._vertices[2].color = {0.f, 0.f, 1.0f};//pure green
    // upload_mesh(triangleMesh);
    // _meshes["triangle"] = triangleMesh;

    // monkey
    // Mesh monkeyMesh;
    // monkeyMesh.load_from_obj("assets/monkey_smooth.obj");
    // upload_mesh(monkeyMesh);
    // _meshes["monkey"] = monkeyMesh;

    // lost empire
    Mesh lostEmpireMesh;
    lostEmpireMesh.load_from_obj("assets/lost_empire.obj");
    upload_mesh(lostEmpireMesh);
    _meshes["lostempire"] = lostEmpireMesh;

    // cottage
    // Mesh cottageMesh;
    // cottageMesh.load_from_obj("assets/cottage_obj.obj");
    // upload_mesh(cottageMesh);
    // _meshes["cottage"] = cottageMesh;
}

void VulkanEngine::upload_mesh(Mesh& mesh) {
    const size_t bufferSize = mesh._vertices.size() * sizeof(Vertex);

    // staging buffer
    AllocatedBuffer stagingBuffer = create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, false);
    void* data;
    vmaMapMemory(_allocator, stagingBuffer._allocation, &data);
    memcpy(data, mesh._vertices.data(), mesh._vertices.size() * sizeof(Vertex));
    vmaUnmapMemory(_allocator, stagingBuffer._allocation);

    // vertex buffer
    mesh._vertexBuffer = create_buffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    immediate_submit([=](VkCommandBuffer cmd) {
        VkBufferCopy copy;
        copy.dstOffset = 0;
        copy.srcOffset = 0;
        copy.size = bufferSize;
        vkCmdCopyBuffer(cmd, stagingBuffer._buffer, mesh._vertexBuffer._buffer, 1, &copy);
    });

    vmaDestroyBuffer(_allocator, stagingBuffer._buffer, stagingBuffer._allocation);
}

Material* VulkanEngine::create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name) {
    Material mat;
    mat.pipeline = pipeline;
    mat.pipelineLayout = layout;
    _materials[name] = mat;
    return &_materials[name];
}

Material* VulkanEngine::get_material(const std::string& name) {
    if (_materials.count(name)) {
        return &_materials[name];
    } else {
        return nullptr;
    }
}

Mesh* VulkanEngine::get_mesh(const std::string& name) {
    if (_meshes.count(name)) {
        return &_meshes[name];
    } else {
        return nullptr;
    }
}

void VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject* first, int count) {
    glm::mat4 view = glm::translate(glm::mat4(1.f), _camPos);
    glm::mat4 projection = glm::perspective(glm::radians(70.0f), (float) _windowExtent.width / _windowExtent.height, 0.1f, 200.0f);
    projection[1][1] *= -1;

    float framed = (_frameNumber / 120.0f);
    _sceneParameters.ambientColor = {sin(framed), 0, cos(framed), 1};

    GPUCameraData cameraData;
    cameraData.proj = projection;
    cameraData.view = view;
    cameraData.viewproj = projection * view;

    char* sceneData;
    vmaMapMemory(_allocator, _sceneBuffer._allocation, (void**) &sceneData);
    int frameIndex = _frameNumber % FRAME_OVERLAP;
    sceneData += pad_uniform_buffer_size(sizeof(GPUSceneData) + sizeof(GPUCameraData)) * frameIndex;
    memcpy(sceneData, &cameraData, sizeof(GPUCameraData));
    sceneData += pad_uniform_buffer_size(sizeof(GPUCameraData));
    memcpy(sceneData, &_sceneParameters, sizeof(GPUSceneData));
    vmaUnmapMemory(_allocator, _sceneBuffer._allocation);

    void* objectSSBOptr;
    vmaMapMemory(_allocator, get_current_frame().objectTransformBuffer._allocation, &objectSSBOptr);
    GPUObjectData* objectSSBO = (GPUObjectData*) objectSSBOptr;
    for (int i = 0; i < count; i++) {
        RenderObject& object = first[i];
        objectSSBO[i].modelMatrix = object.transformMatrix;
        objectSSBO[i].color = glm::vec4(1.0f, float(i % 255) / 255, 1.0f, 1.0f);
    }
    vmaUnmapMemory(_allocator, get_current_frame().objectTransformBuffer._allocation);

    // draw commands
    std::vector<IndirectBatch> draws = vkutil::compact_draws(first, count);

    void* drawCommandsPtr;
    vmaMapMemory(_allocator, get_current_frame().indirectBuffer._allocation, &drawCommandsPtr);
    VkDrawIndirectCommand* drawCommands = (VkDrawIndirectCommand*)drawCommandsPtr;
    for(int i = 0; i < count; i++){
        RenderObject& object = first[i];
        drawCommands[i].vertexCount = object.mesh->_vertices.size();
        drawCommands[i].instanceCount = 1;
        drawCommands[i].firstVertex = 0;
        drawCommands[i].firstInstance = i;
    }
    vmaUnmapMemory(_allocator, get_current_frame().indirectBuffer._allocation);

    for (auto& draw : draws) {
        // pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline);

        uint32_t uniform_offsets[] = {
                uint32_t(pad_uniform_buffer_size(sizeof(GPUSceneData) + sizeof(GPUCameraData)) * frameIndex),
                uint32_t(pad_uniform_buffer_size(sizeof(GPUSceneData) + sizeof(GPUCameraData)) * frameIndex + sizeof(GPUCameraData))};
        // set 0: global data (scene, camera...)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipelineLayout,
                                0, 1, &get_current_frame().globalDescriptor, 2, uniform_offsets);

        // set 1: object data (transform, color...)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipelineLayout,
                                1, 1, &get_current_frame().objectDescriptor, 0, nullptr);

        // set 2: object textures data
        if (draw.material->textureSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipelineLayout, 2, 1, &draw.material->textureSet, 0, nullptr);
        }

        // mesh
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &draw.mesh->_vertexBuffer._buffer, &offset);

        // direct command
        // for (int i = draw.first; i < draw.count; i++) {
        //     vkCmdDraw(cmd, draw.mesh->_vertices.size(), 1, 0, i);
        // }

        // indirect command
        VkDeviceSize indirect_offset = draw.first * sizeof(VkDrawIndirectCommand);
        uint32_t draw_stride = sizeof(VkDrawIndirectCommand);
        vkCmdDrawIndirect(cmd, get_current_frame().indirectBuffer._buffer, indirect_offset, draw.count, draw_stride);
    }
}

void VulkanEngine::init_scene() {
    VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);
    // VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);
    VkSampler blockySampler;
    vkCreateSampler(_device, &samplerInfo, nullptr, &blockySampler);

    _mainDeletionQueue.push_function([=]() {
        vkDestroySampler(_device, blockySampler, nullptr);
    });

    Material* texturedMat = get_material("texturedMesh");

    // allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.pNext = nullptr;
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &_singleTextureSetLayout;
    vkAllocateDescriptorSets(_device, &allocInfo, &texturedMat->textureSet);

    // update descriptor set
    VkDescriptorImageInfo imageWrite;
    imageWrite.sampler = blockySampler;
    imageWrite.imageView = _textures["empire_diffuse"].imageView;
    imageWrite.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet texture1 = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texturedMat->textureSet, &imageWrite, 0);
    vkUpdateDescriptorSets(_device, 1, &texture1, 0, nullptr);

    // RenderObject monkey;
    // monkey.mesh = get_mesh("monkey");
    // monkey.material = get_material("defaultLightMesh");
    // monkey.transformMatrix = glm::mat4{1.0f};
    // _renderables.push_back(monkey);

    RenderObject map;
    map.mesh = get_mesh("lostempire");
    map.material = get_material("texturedMesh");
    map.transformMatrix = glm::translate(glm::vec3{5, -10, 0});
    _renderables.push_back(map);

    // RenderObject cottage;
    // cottage.mesh = get_mesh("cottage");
    // cottage.material = get_material("defaultmesh");
    // glm::mat4 translation = glm::translate(glm::mat4{1.0f}, glm::vec3(0.0f, 2.0f, 1.0f));
    // cottage.transformMatrix = translation;
    // _renderables.push_back(cottage);

    // for (int x = -20; x <= 20; x++) {
    //     for (int y = -20; y <= 20; y++) {
    //         RenderObject tri;
    //         tri.mesh = get_mesh("triangle");
    //         tri.material = get_material("defaultLightMesh");
    //         glm::mat4 translation = glm::translate(glm::mat4{1.0}, glm::vec3(x, 0, y));
    //         glm::mat4 scale = glm::scale(glm::mat4{1.0}, glm::vec3(0.2, 0.2, 0.2));
    //         tri.transformMatrix = translation * scale;
    //         _renderables.push_back(tri);
    //     }
    // }

    // sort objects with pipeline

    std::sort(_renderables.begin(), _renderables.end(),
              [&](RenderObject& r1, RenderObject& r2) -> bool {
                  return r1.material->pipeline < r2.material->pipeline;
              });
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass) {
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.pNext = nullptr;

    viewportState.viewportCount = 1;
    viewportState.pViewports = &_viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &_scissor;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.pNext = nullptr;

    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &_colorBlendAttachment;

    // build the actual pipeline
    // we now use all of the info structs we have been writing into this one to create the pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = nullptr;

    pipelineInfo.stageCount = _shaderStages.size();
    pipelineInfo.pStages = _shaderStages.data();
    pipelineInfo.pVertexInputState = &_vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &_inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &_rasterizer;
    pipelineInfo.pMultisampleState = &_multisampling;
    pipelineInfo.pDepthStencilState = &_depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = _pipelineLayout;
    pipelineInfo.renderPass = pass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    VkPipeline newPipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
        std::cout << "failed to create pipeline\n";
        return VK_NULL_HANDLE;
    } else {
        return newPipeline;
    }
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, bool addDeletionQueue) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memoryUsage;

    AllocatedBuffer newBuffer;
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo,
                             &newBuffer._buffer,
                             &newBuffer._allocation,
                             nullptr));
    if (addDeletionQueue) {
        _mainDeletionQueue.push_function([=]() {
            vmaDestroyBuffer(_allocator, newBuffer._buffer, newBuffer._allocation);
        });
    }

    return newBuffer;
}

void VulkanEngine::init_descriptors() {
    std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 20},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10}};

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = 0;
    pool_info.maxSets = 30;
    pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pool_info.pPoolSizes = sizes.data();
    vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptorPool);

    _mainDeletionQueue.push_function([=]() {
        vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
    });

    VkDescriptorSetLayoutBinding cameraBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT, 0);
    VkDescriptorSetLayoutBinding sceneBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1);
    VkDescriptorSetLayoutBinding bindings[] = {cameraBind, sceneBind};

    VkDescriptorSetLayoutCreateInfo setInfo = {};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setInfo.pNext = nullptr;
    setInfo.bindingCount = 2;
    setInfo.flags = 0;
    setInfo.pBindings = bindings;
    vkCreateDescriptorSetLayout(_device, &setInfo, nullptr, &_globalSetLayout);

    VkDescriptorSetLayoutBinding objectBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
    VkDescriptorSetLayoutCreateInfo set2info = {};
    set2info.bindingCount = 1;
    set2info.flags = 0;
    set2info.pNext = nullptr;
    set2info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set2info.pBindings = &objectBind;
    vkCreateDescriptorSetLayout(_device, &set2info, nullptr, &_objectSetLayout);

    const size_t sceneBufferSize = FRAME_OVERLAP * pad_uniform_buffer_size(sizeof(GPUSceneData) + sizeof(GPUCameraData));
    _sceneBuffer = create_buffer(sceneBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        const int MAX_OBJECTS = 10000;
        _frames[i].objectTransformBuffer = create_buffer(sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

        VkDescriptorSetAllocateInfo globalSetAlloc = vkinit::descriptor_set_alloc(_descriptorPool, &_globalSetLayout);
        vkAllocateDescriptorSets(_device, &globalSetAlloc, &_frames[i].globalDescriptor);

        VkDescriptorSetAllocateInfo objectSetAlloc = vkinit::descriptor_set_alloc(_descriptorPool, &_objectSetLayout);
        vkAllocateDescriptorSets(_device, &objectSetAlloc, &_frames[i].objectDescriptor);

        VkDescriptorBufferInfo objectInfo;
        objectInfo.buffer = _frames[i].objectTransformBuffer._buffer;
        objectInfo.offset = 0;
        objectInfo.range = sizeof(GPUObjectData) * MAX_OBJECTS;

        VkDescriptorBufferInfo cameraInfo;
        cameraInfo.buffer = _sceneBuffer._buffer;
        cameraInfo.offset = 0;
        cameraInfo.range = sizeof(GPUCameraData);

        VkDescriptorBufferInfo sceneInfo;
        sceneInfo.buffer = _sceneBuffer._buffer;
        sceneInfo.offset = 0;
        sceneInfo.range = sizeof(GPUSceneData);

        VkWriteDescriptorSet cameraWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, _frames[i].globalDescriptor, &cameraInfo, 0);
        VkWriteDescriptorSet sceneWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, _frames[i].globalDescriptor, &sceneInfo, 1);
        VkWriteDescriptorSet objectWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _frames[i].objectDescriptor, &objectInfo, 0);
        VkWriteDescriptorSet setWrites[] = {cameraWrite, sceneWrite, objectWrite};

        vkUpdateDescriptorSets(_device, 3, setWrites, 0, nullptr);
    }

    VkDescriptorSetLayoutBinding textureBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
    VkDescriptorSetLayoutCreateInfo set3info = {};
    set3info.bindingCount = 1;
    set3info.flags = 0;
    set3info.pNext = nullptr;
    set3info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set3info.pBindings = &textureBind;

    vkCreateDescriptorSetLayout(_device, &set3info, nullptr, &_singleTextureSetLayout);
    _mainDeletionQueue.push_function([=]() {
        vkDestroyDescriptorSetLayout(_device, _globalSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _objectSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _singleTextureSetLayout, nullptr);
    });
}

size_t VulkanEngine::pad_uniform_buffer_size(size_t originalSize) {
    size_t minUboAlignment = _gpuProperties.limits.minUniformBufferOffsetAlignment;
    size_t alignedSize = originalSize;
    if (minUboAlignment > 0) {
        alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
    }
    return alignedSize;
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function) {
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_uploadContext._commandPool, 1);
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &cmd));

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    function(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit = vkinit::submit_Info(&cmd);
    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _uploadContext._uploadFence));
    vkWaitForFences(_device, 1, &_uploadContext._uploadFence, true, 9999999999);
    vkResetFences(_device, 1, &_uploadContext._uploadFence);

    vkResetCommandPool(_device, _uploadContext._commandPool, 0);
}

void VulkanEngine::load_images() {
    Texture lostEmpire;
    vkutil::load_image_from_file(*this, "assets/lost_empire-RGBA.png", lostEmpire.image);
    VkImageViewCreateInfo imageInfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_SRGB, lostEmpire.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
    vkCreateImageView(_device, &imageInfo, nullptr, &lostEmpire.imageView);

    _textures["empire_diffuse"] = lostEmpire;
    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, lostEmpire.imageView, nullptr);
    });
}
