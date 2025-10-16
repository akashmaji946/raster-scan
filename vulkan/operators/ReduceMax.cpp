// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ReduceMax.hpp"

#include <iostream>
#include <cmath>

#include <common/utils.h>
#include <common/constants.h>
#include <common/ShaderDefines.h>

#include <core/VkEngine.hpp>

namespace vkcore {

ReduceMax::ReduceMax(PVkDevice vd) {
    this->vd = vd;
}

void ReduceMax::initialize() {
//    std::cerr << std::endl << "creating reduce max pipeline" << std::endl;

    {
        std::vector<uint32_t> cshader;
        validate(readShader(SHADER_FOLDER + "/reduce-max.comp.spv",cshader),"read image reduce max shader");
        vk::ShaderModuleCreateInfo computeShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), cshader.size() * sizeof(uint32_t), cshader.data());
        maxShader = vd->device->createShaderModuleUnique(computeShaderModuleCreateInfo);

        // compute pipeline
        std::array<vk::SpecializationMapEntry, 1> specializationMapEntries;
        specializationMapEntries[0].constantID = 0;
        specializationMapEntries[0].size = sizeof(int32_t);
        specializationMapEntries[0].offset = 0;

        vk::SpecializationInfo specializationInfo;
        specializationInfo.dataSize = sizeof(int32_t);
        specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());;
        specializationInfo.pMapEntries = specializationMapEntries.data();
        specializationInfo.pData = &(vd->subgroupSize);

        maxPipelineProps.setShader(maxShader.get(), &specializationInfo);


        maxPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };
        maxPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
        };

        maxPipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute,0,sizeof(int32_t) * 1)
        };

        maxPipeline = maxPipelineProps.createPipeline(vd);
    }

    //    std::cerr << std::endl << "finished creating reduce max pipeline" << std::endl;
}

void ReduceMax::cmdReduce(PBuffer ipBuf, PBuffer opBuf, uint32_t size) {
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
    this->reduce(ipBuf,opBuf,size);
    vd->commandBuffer->end();

    vk::UniqueFence fence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->submit(submitInfo,fence.get(),false);
    vd->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
}

void ReduceMax::reduce(PBuffer ipBuf, PBuffer opBuf, uint32_t size) {
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, maxPipeline.get());

    vk::DescriptorBufferInfo ipDescriptor{ ipBuf->buf, 0, VK_WHOLE_SIZE };
    vk::DescriptorBufferInfo opDescriptor{ opBuf->buf, 0, VK_WHOLE_SIZE };

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{ maxPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &ipDescriptor},
        vk::WriteDescriptorSet{ maxPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &opDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, maxPipelineProps.pipelineLayout.get(), 0, maxPipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t,1> consts = {size};
    vd->commandBuffer->pushConstants<uint32_t>(maxPipelineProps.pipelineLayout.get(),vk::ShaderStageFlagBits::eCompute,0,consts);

    int dsize = int (std::ceil(double(size) / GROUP_SIZE));
    vd->commandBuffer->dispatch(dsize,1,1);
}

} // namespace
