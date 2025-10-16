// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VkInclude.hpp>
#include <vector>

#include <core/VulkanDevice.hpp>

namespace vkcore {

class ComputePipelineProperties
{
public:
    ComputePipelineProperties();
    void setShader(vk::ShaderModule &shaderModule, vk::SpecializationInfo *specializationInfo = nullptr);
    vk::UniquePipeline createPipeline( PVkDevice vd);
    vk::UniquePipeline createPipeline(VulkanDevice *vd);

public:
    // To be filled by the code -- optional
    std::vector<vk::PushConstantRange> pushConstantRange;
    std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
    std::vector<vk::DescriptorPoolSize> poolSizes;

public:
    // derived from above properties
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorPool descriptorPool;
    vk::UniqueDescriptorSet descriptorSet;

    vk::UniquePipelineLayout pipelineLayout;

protected:
    bool shaderStageFlag;
    vk::PipelineShaderStageCreateInfo pipelineShaderStageCreateInfo;
};

} // namespace


