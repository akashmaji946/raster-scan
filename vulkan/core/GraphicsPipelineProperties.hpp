// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VkInclude.hpp>
#include <vector>

#include <core/VulkanDevice.hpp>

namespace vkcore {

enum class BlendFunc : int {
    BLEND_NONE = 0,
    BLEND_DEFAULT_ALPHA,
    BLEND_ADD,
    BLEND_MIN,
    BLEND_MAX,
    BLEND_OVERWRITE
};

class GraphicsPipelineProperties
{
public:
    GraphicsPipelineProperties();
    vk::UniquePipeline createPipeline(PVkDevice vd, vk::UniqueRenderPass &renderPass, vk::PipelineRenderingCreateInfo *renderingPipelineCreateInfo = nullptr);

    void enableDepthTest();
    void updateBlendProperties(vk::PipelineColorBlendAttachmentState blendState);
    void updateBlendAttachmentCount(int attachmentCount);

    void setShaderStageFlag();
    void setInputBindingFlag();
    void setInputAttrFlag();
    void setInputAssemblyFlag();
    void setBlendFunction(BlendFunc blendFunc);
    void setLogicalOp(vk::LogicOp lop);

public:
    // To be filled by the code -- required
    bool shaderStageFlag, inputBindingFlag, inputAttrFlag, inputAssemblyFlag;
    std::vector<vk::PipelineShaderStageCreateInfo> pipelineShaderStageCreateInfos;
    std::vector<vk::VertexInputBindingDescription> vertexInputBindingDescriptions;
    std::vector<vk::VertexInputAttributeDescription> vertexInputAttributeDescriptions;
    vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo;

    // To be filled by the code -- optional
    std::vector<vk::PushConstantRange> pushConstantRange;
    std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
    std::vector<vk::DescriptorPoolSize> poolSizes;

public:
    // filled with default values
    vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo;
    vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo;
    vk::PipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo;
    vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo;
    vk::ColorComponentFlags colorComponentFlags;
    vk::PipelineColorBlendAttachmentState pipelineColorBlendAttachmentState;
    std::vector<vk::DynamicState> dynamicStates;
    vk::PipelineTessellationStateCreateInfo *pipelineTessellationStateCreateInfo;
    vk::LogicOp logicalOp; vk::Bool32 logicalOpEnable;
    int blendAttachmentCount;

public:
    // derived from above properties
    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;
    vk::PipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo;
    vk::PipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo;
    vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorPool descriptorPool;
    vk::UniqueDescriptorSet descriptorSet;

    vk::UniquePipelineLayout pipelineLayout;
};

} // namespace

