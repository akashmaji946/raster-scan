// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ComputePipelineProperties.hpp"

#include <iostream>

namespace vkcore {

ComputePipelineProperties::ComputePipelineProperties() {
    this->shaderStageFlag = false;
}

void ComputePipelineProperties::setShader(vk::ShaderModule &shaderModule, vk::SpecializationInfo *specializationInfo) {
    pipelineShaderStageCreateInfo = vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, shaderModule, "main");
    pipelineShaderStageCreateInfo.pSpecializationInfo = specializationInfo;
    shaderStageFlag = true;
}

vk::UniquePipeline ComputePipelineProperties::createPipeline(PVkDevice vd) {
    return this->createPipeline(vd.get());
}

vk::UniquePipeline ComputePipelineProperties::createPipeline(VulkanDevice *vd) {
    if(!shaderStageFlag) {
        std::cerr << "compute shader not set" << std::endl;
        exit(-3);
    }

    descriptorSetLayout = vd->device->createDescriptorSetLayoutUnique({
        {},
        uint32_t(setLayoutBindings.size()),
        setLayoutBindings.data()
    });
    if(setLayoutBindings.size() > 0) {
        descriptorPool = vd->device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo{ vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });;
        std::vector<vk::UniqueDescriptorSet> descriptorSets = vd->device->allocateDescriptorSetsUnique({ descriptorPool.get(), 1, &descriptorSetLayout.get() });
        descriptorSet = std::move(descriptorSets[0]);
    }

    // TODO can there be more than 1 descriptor set??
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
        vk::PipelineLayoutCreateFlags(),
        1, &descriptorSetLayout.get(),    // descriptorSetLayout
        (uint32_t)pushConstantRange.size(), pushConstantRange.data()      // constantRange
    );

    pipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutCreateInfo);

    vk::ComputePipelineCreateInfo computePipelineCreateInfo;
    computePipelineCreateInfo.layout = pipelineLayout.get();
    computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
    return vd->device->createComputePipelineUnique(vd->pipelineCache.get(), computePipelineCreateInfo).value;
}

} // namespace
