// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "SinglePassScan.hpp"

#include <common/utils.h>

#include <cmath>
#include <iostream>

namespace vkcore {

SinglePassScan::SinglePassScan(PVkDevice device) {
    this->vd = device;
}

SinglePassScan::~SinglePassScan() {

}

void SinglePassScan::initialize() {
    maxWorkGroupSize[0] = vd->props.limits.maxComputeWorkGroupSize[0];
    maxWorkGroupSize[1] = vd->props.limits.maxComputeWorkGroupSize[1];
    maxWorkGroupSize[2] = vd->props.limits.maxComputeWorkGroupSize[2];
    threadsPerGroup = vd->subgroupSize;
    if(vd->props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
        threadsPerGroup = 16;
    }

    this->initPrefixSum();

    // initializing with a relatively large value
    sumBuffer.reset(new Buffer(vd));
    int numGroups = 1024*1024;
    sumBuffer->create(sizeof(int32_t) * numGroups, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst,MemoryType::Internal);
}

void SinglePassScan::initPrefixSum() {
    // Setup shader
    std::vector<uint32_t> cshader;
    validate(readShader(SHADER_FOLDER + "/prefixsum.comp.spv",cshader),"read prefix sum shader");
    vk::ShaderModuleCreateInfo computeShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), cshader.size() * sizeof(uint32_t), cshader.data());
    psShader = vd->device->createShaderModuleUnique(computeShaderModuleCreateInfo);

    std::array<vk::SpecializationMapEntry, 5> specializationMapEntries;
    int offset = 0;
    for(int i = 0;i < 5;i ++) {
        specializationMapEntries[i].constantID = i;
        specializationMapEntries[i].size = sizeof(int32_t);
        specializationMapEntries[i].offset = offset;
        offset += sizeof(int32_t);
    }
    psGroupSize = (vd->props.limits.maxComputeSharedMemorySize + 16) / (4 * threadsPerGroup + 8);
    // should be a power of 2
    if(psGroupSize > 256) {
        psGroupSize = 256;
    } else {
        psGroupSize = 128;
    }
    int32_t constData[5] = {psGroupSize,threadsPerGroup,psGroupSize / threadsPerGroup,int32_t(std::log2(threadsPerGroup)),psGroupSize/2};

    vk::SpecializationInfo specializationInfo;
    specializationInfo.dataSize = specializationMapEntries.size() * sizeof(int32_t);
    specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
    specializationInfo.pMapEntries = specializationMapEntries.data();
    specializationInfo.pData = constData;

    psPipelineProps.setShader(psShader.get(), &specializationInfo);

    psPipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
    };

    psPipelineProps.poolSizes = {
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
    };

    psPipelineProps.pushConstantRange = {
        vk::PushConstantRange (vk::ShaderStageFlagBits::eCompute,0,sizeof(int32_t))
    };

    psPipeline = psPipelineProps.createPipeline(vd);
}

void SinglePassScan::cmdPrefixSum(vk::Buffer &buffer, size_t size) {
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
    this->prefixSum(buffer,size);
    vd->commandBuffer->end();

    vk::UniqueFence fence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->submit(submitInfo,fence.get(),false);
    vd->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
}


void SinglePassScan::prefixSum(vk::Buffer &buffer, size_t size) {
    if(size % (psGroupSize * threadsPerGroup) != 0) {
        std::cerr << "size should be divisible by " << (psGroupSize * threadsPerGroup) << std::endl;
        exit(2);
    }

    int32_t workGroupSize = int32_t(size / (psGroupSize * threadsPerGroup));
    int32_t wg[3];
    if(workGroupSize > maxWorkGroupSize[0]) {
        wg[0] = maxWorkGroupSize[0];
        wg[1] = int(std::ceil(double(workGroupSize) / maxWorkGroupSize[0]));
        if(wg[1] > maxWorkGroupSize[1]) {
            int tmp = wg[1];
            wg[1] = maxWorkGroupSize[1];
            wg[2] = int(std::ceil(double(tmp) / maxWorkGroupSize[1]));
        } else {
            wg[2] = 1;
        }
    } else {
        wg[0] = workGroupSize;
        wg[1] = wg[2] = 1;
    }

    if(sumBuffer->size < (workGroupSize + 2) * sizeof(int32_t)) {
        size_t newSize = int(std::ceil(double((workGroupSize + 2)) / 4)) * 4;
        sumBuffer->create(newSize * sizeof(int32_t),sumBuffer->flags,sumBuffer->type);
    }
    sumBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eComputeShader,-1);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, psPipeline.get());

    vk::DescriptorBufferInfo bufDescriptor{ buffer, 0, VK_WHOLE_SIZE };
    vk::DescriptorBufferInfo sumBufDescriptor{ sumBuffer->buf, 0, VK_WHOLE_SIZE };
    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{ psPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &bufDescriptor },
        vk::WriteDescriptorSet{ psPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sumBufDescriptor },
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, psPipelineProps.pipelineLayout.get(), 0, psPipelineProps.descriptorSet.get(), nullptr);

    std::array<int32_t,1> consts = {workGroupSize};
    vd->commandBuffer->pushConstants<int32_t>(psPipelineProps.pipelineLayout.get(),vk::ShaderStageFlagBits::eCompute,0,consts);

    vd->commandBuffer->dispatch(wg[0], wg[1], wg[2]);
}

uint32_t SinglePassScan::getBufSizeDivisor() {
    return (psGroupSize * threadsPerGroup);
}

uint32_t SinglePassScan::getPrefixSumGroupSize() {
    return psGroupSize;
}

}
