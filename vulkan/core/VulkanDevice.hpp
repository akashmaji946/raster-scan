// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VkInclude.hpp>
#include <vma/vma.hpp>

#include <common/constants.h>
#include <memory>
#include <mutex>

namespace vkcore {

class VulkanDevice
{
public:
    VulkanDevice(vk::PhysicalDevice &pdevice, vk::PhysicalDeviceProperties &props, int devId);
    ~VulkanDevice();

public:
    vk::UniqueRenderPass createRenderPass(vk::Format colorFormat, int noAttachments, bool clear = true);
    int getNextTimer();
    void resetTimer(int timerIndex);

public:
    void submit(vk::SubmitInfo &submitInfo, vk::Fence &fence, bool transfer);
    void waitForFences(vk::Fence &fence, vk::Bool32 block, uint64_t timeout);

protected:
    bool initDevice();
    void setupVMA();
    void createSharedImagePool(vk::Format colorFormat);
    void createSharedBufferPool();
    void setupCommonVariables();

protected:
    vk::UniqueCommandPool graphicsCommandPool, transferCommandPool[NUM_STAGING_ARRAYS + 1];
    uint32_t computeQueueIndex, graphicsQueueIndex, transferQueueIndex;

    vk::Queue queue, tqueue;
    std::mutex cmut, tmut;

public:
    vk::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceProperties props;
    vk::PhysicalDeviceMemoryProperties memProps;
    vk::UniqueDevice device;
    vk::UniqueCommandBuffer commandBuffer, transferCommandBuffer[NUM_STAGING_ARRAYS + 1];
    vk::UniquePipelineCache pipelineCache;
    vk::UniqueSampler nearestClampSampler;
    vk::UniqueQueryPool timerPool;
    bool rebarEnabled;
    int32_t maxRenderings;
    vk::DeviceSize heapSize;

    VmaAllocator allocator;

    vk::ExportMemoryAllocateInfo rgba32fExportAllocInfo, bufferExportAllocInfo;
    VmaPool rgba32fImagePool, bufferPool;

protected:
    bool timer[N_TIMERS];
    int curTimer;

public:
    int id;
    uint32_t subgroupSize;
};

typedef std::shared_ptr<VulkanDevice> PVkDevice;

} // namespace

