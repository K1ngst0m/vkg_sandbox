// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include "vk_types.h"

namespace vkinit {
    //vulkan init code goes here
    VkCommandPoolCreateInfo command_pool_create_info(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags = 0);

    VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info(VkShaderStageFlagBits stage, VkShaderModule shaderModule);

    VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info();

    VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info(VkPrimitiveTopology topology);

    VkPipelineRasterizationStateCreateInfo rasterization_state_create_info(VkPolygonMode polygonMode);

    VkPipelineMultisampleStateCreateInfo multisampling_state_create_info();

    VkPipelineColorBlendAttachmentState color_blend_attachment_state();

    VkPipelineLayoutCreateInfo pipeline_layout_create_info(VkDescriptorSetLayout *layout = nullptr, uint32_t layout_count = 0, VkPushConstantRange *pc_range = nullptr, uint32_t pc_range_count = 0);

    VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent);

    VkImageViewCreateInfo imageview_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags);

    VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info(bool bDepthTest, bool bDepthWrite, VkCompareOp compareOp);

    VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags = 0);

    VkSemaphoreCreateInfo semaphore_create_info();

    VkDescriptorSetLayoutBinding descriptorset_layout_binding(VkDescriptorType type, VkShaderStageFlags stageFlags, uint32_t binding);

    VkWriteDescriptorSet write_descriptor_buffer(VkDescriptorType type, VkDescriptorSet dstSet, VkDescriptorBufferInfo* bufferInfo , uint32_t binding);

    VkDescriptorSetAllocateInfo descriptor_set_alloc(VkDescriptorPool pool, VkDescriptorSetLayout * layout);

    VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags, VkCommandBufferInheritanceInfo * info = nullptr);

    VkSubmitInfo submit_Info(VkCommandBuffer* cmd, uint32_t cb_count = 1, VkSemaphore* wait = nullptr, VkSemaphore* sig = nullptr);

    VkSamplerCreateInfo sampler_create_info(VkFilter filters, VkSamplerAddressMode samplerAddressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT);

    VkWriteDescriptorSet write_descriptor_image(VkDescriptorType type, VkDescriptorSet dstSet, VkDescriptorImageInfo* imageInfo, uint32_t binding);
} // namespace vkinit
