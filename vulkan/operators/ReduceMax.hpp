// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VulkanDevice.hpp>
#include <core/VkData.hpp>
#include <core/ComputePipelineProperties.hpp>

namespace vkcore {

class ReduceMax
{
public:
    ReduceMax(PVkDevice vd);

public:
    void initialize();
    void cmdReduce(PBuffer ipBuf, PBuffer opBuf, uint32_t size);
    void reduce(PBuffer ipBuf, PBuffer opBuf, uint32_t size);

protected:
    PVkDevice vd;

    vk::UniqueShaderModule maxShader;
    ComputePipelineProperties maxPipelineProps;
    vk::UniquePipelineLayout maxPipelineLayout;
    vk::UniquePipeline maxPipeline;

public:
    int32_t n, nTiles;
    int32_t resx, resy;
    vk::DeviceSize size;
};

} // namespace

