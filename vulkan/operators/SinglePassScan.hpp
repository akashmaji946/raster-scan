// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VkInclude.hpp>

#include <core/VulkanDevice.hpp>
#include <core/VkData.hpp>
#include <core/ComputePipelineProperties.hpp>

namespace vkcore {

class SinglePassScan
{
public:
    SinglePassScan(PVkDevice vd);
    ~SinglePassScan();

public:
    void initialize();
    void cmdPrefixSum(vk::Buffer &buffer, size_t size);
    void prefixSum(vk::Buffer &buffer, size_t size);

public:
    uint32_t getBufSizeDivisor();
    uint32_t getPrefixSumGroupSize();

protected:
    void initPrefixSum();

protected:
    PVkDevice vd;

    vk::UniqueShaderModule psShader;
    ComputePipelineProperties psPipelineProps;
    vk::UniquePipeline psPipeline;

    PBuffer sumBuffer;
    int threadsPerGroup, psGroupSize;

public:
    int maxWorkGroupSize[3];
};

} // namespace

