#include "vk_initializers.h"

namespace vkinit {
    //vulkan init code goes here
    VkCommandPoolCreateInfo command_pool_create_info(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags) {
        VkCommandPoolCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.pNext = nullptr;

        info.queueFamilyIndex = queueFamilyIndex;
        info.flags = flags;
        return info;
    }

    VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool,
                                                             uint32_t count,
                                                             VkCommandBufferLevel level) {
        VkCommandBufferAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.pNext = nullptr;

        // commands will be made from our _commandPool
        info.commandPool = pool;
        // we will allocate 1 command buffer
        info.commandBufferCount = count;
        // command level is Primary
        info.level = level;

        return info;
    }

    VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info(VkShaderStageFlagBits stage,
                                                                      VkShaderModule shaderModule) {
        VkPipelineShaderStageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.pNext = nullptr;

        // shader stage
        info.stage = stage;
        // module contianing the code for this shader stage
        info.module = shaderModule;
        // the entry point of the shader
        info.pName = "main";

        return info;
    }

    VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info() {
        VkPipelineVertexInputStateCreateInfo info = {};

        info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        info.pNext = nullptr;

        info.vertexBindingDescriptionCount = 0;
        info.vertexAttributeDescriptionCount = 0;

        return info;
    }

    VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info(VkPrimitiveTopology topology) {
        VkPipelineInputAssemblyStateCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        info.pNext = nullptr;

        info.topology = topology;
        info.primitiveRestartEnable = VK_FALSE;

        return info;
    }

    VkPipelineRasterizationStateCreateInfo rasterization_state_create_info(VkPolygonMode polygonMode) {
        VkPipelineRasterizationStateCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        info.pNext = nullptr;

        info.depthClampEnable = VK_FALSE;
        //discard all primitives before the rasterization stage if enabled which we don't want
        info.rasterizerDiscardEnable = VK_FALSE;

        info.polygonMode = polygonMode;
        info.lineWidth = 1.0f;
        // no backface cull
        info.cullMode = VK_CULL_MODE_NONE;
        info.frontFace = VK_FRONT_FACE_CLOCKWISE;
        // no depth bias
        info.depthBiasEnable = VK_FALSE;
        info.depthBiasConstantFactor = 0.0f;
        info.depthBiasClamp = 0.0f;
        info.depthBiasSlopeFactor = 0.0f;

        return info;
    }

    VkPipelineMultisampleStateCreateInfo multisampling_state_create_info() {
        VkPipelineMultisampleStateCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        info.pNext = nullptr;

        info.sampleShadingEnable = VK_FALSE;
        // multisampling defaulted to no multisampling (1 sample per pixel)
        info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        info.minSampleShading = 1.0f;
        info.pSampleMask = nullptr;
        info.alphaToCoverageEnable = VK_FALSE;
        info.alphaToOneEnable = VK_FALSE;

        return info;
    }

    VkPipelineColorBlendAttachmentState color_blend_attachment_state() {
        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                              VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT |
                                              VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        return colorBlendAttachment;
    }

    VkPipelineLayoutCreateInfo pipeline_layout_create_info(VkDescriptorSetLayout *layout, uint32_t layout_count, VkPushConstantRange *pc_range, uint32_t pc_range_count) {
        VkPipelineLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.pNext = nullptr;

        info.flags = 0;
        info.setLayoutCount = layout_count;
        info.pSetLayouts = layout;
        info.pushConstantRangeCount = pc_range_count;
        info.pPushConstantRanges = pc_range;

        return info;
    }

    VkImageCreateInfo image_create_info(VkFormat format,
                                        VkImageUsageFlags usageFlags,
                                        VkExtent3D extent) {
        VkImageCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.pNext = nullptr;

        info.imageType = VK_IMAGE_TYPE_2D;

        info.format = format;
        info.extent = extent;
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usageFlags;

        return info;
    }

    VkImageViewCreateInfo imageview_create_info(VkFormat format,
                                                VkImage image,
                                                VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.pNext = nullptr;

        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.image = image;
        info.format = format;
        info.subresourceRange.baseMipLevel = 0;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.baseArrayLayer = 0;
        info.subresourceRange.layerCount = 1;
        info.subresourceRange.aspectMask = aspectFlags;

        return info;
    }

    VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info(bool bDepthTest, bool bDepthWrite, VkCompareOp compareOp) {
        VkPipelineDepthStencilStateCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        info.pNext = nullptr;

        info.depthTestEnable = bDepthTest ? VK_TRUE : VK_FALSE;
        info.depthWriteEnable = bDepthWrite ? VK_TRUE : VK_FALSE;
        info.depthCompareOp = bDepthTest ? compareOp : VK_COMPARE_OP_ALWAYS;
        info.depthBoundsTestEnable = VK_FALSE;
        info.minDepthBounds = 0.0f;
        info.maxDepthBounds = 1.0f;
        info.stencilTestEnable = VK_FALSE;

        return info;
    }

    VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags)
    {
        VkFenceCreateInfo info;
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        info.pNext = nullptr;
        info.flags = flags;

        return info;
    }

    VkSemaphoreCreateInfo semaphore_create_info(){
        VkSemaphoreCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        info.pNext = nullptr;
        info.flags = 0;
        return info;
    }

    VkDescriptorSetLayoutBinding descriptorset_layout_binding(VkDescriptorType type, VkShaderStageFlags stageFlags, uint32_t binding){
        VkDescriptorSetLayoutBinding setbind = {};
        setbind.binding = binding;
        setbind.descriptorCount= 1;
        setbind.descriptorType = type;
        setbind.pImmutableSamplers = nullptr;
        setbind.stageFlags = stageFlags;

        return setbind;
    }

    VkWriteDescriptorSet write_descriptor_buffer(VkDescriptorType type, VkDescriptorSet dstSet, VkDescriptorBufferInfo* bufferInfo , uint32_t binding)
    {
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.pNext = nullptr;

        write.dstBinding = binding;
        write.dstSet = dstSet;
        write.descriptorCount = 1;
        write.descriptorType = type;
        write.pBufferInfo = bufferInfo;

        return write;
    }

    VkDescriptorSetAllocateInfo descriptor_set_alloc(VkDescriptorPool pool, VkDescriptorSetLayout * layout){
        VkDescriptorSetAllocateInfo info;
        info.pNext = nullptr;
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = pool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = layout;
        return info;
    }

    VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags, VkCommandBufferInheritanceInfo * inheritance_info){
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.pNext = nullptr;
        info.pInheritanceInfo = inheritance_info;
        info.flags = flags;
        return info;
    }

    VkSubmitInfo submit_Info(VkCommandBuffer* cmd, uint32_t cb_count, VkSemaphore* wait, VkSemaphore* sig){
        VkSubmitInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.pNext = nullptr;
        info.waitSemaphoreCount = (wait == nullptr ? 0 : 1);
        info.pWaitSemaphores = wait;
        info.signalSemaphoreCount = (sig == nullptr ? 0 : 1);
        info.pSignalSemaphores = sig;
        info.commandBufferCount = 1;
        info.pCommandBuffers = cmd;

        return info;
    }

    VkSamplerCreateInfo sampler_create_info(VkFilter filters, VkSamplerAddressMode samplerAddressMode){
        VkSamplerCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.pNext = nullptr;
        info.magFilter = filters;
        info.minFilter = filters;
        info.addressModeU = samplerAddressMode;
        info.addressModeV = samplerAddressMode;
        info.addressModeW = samplerAddressMode;

        return info;
    }

    VkWriteDescriptorSet write_descriptor_image(VkDescriptorType type, VkDescriptorSet dstSet, VkDescriptorImageInfo *imageInfo, uint32_t binding){
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.pNext = nullptr;

        write.dstBinding = binding;
        write.dstSet = dstSet;
        write.descriptorCount = 1;
        write.descriptorType = type;
        write.pImageInfo = imageInfo;

        return write;
    }
}// namespace vkinit
