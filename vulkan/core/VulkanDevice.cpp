// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "VulkanDevice.hpp"

#include <common/utils.h>
#include <core/VkEngine.hpp>

#include <iostream>

namespace vkcore {

VulkanDevice::VulkanDevice(vk::PhysicalDevice &pdevice, vk::PhysicalDeviceProperties &props, int devId)
    :physicalDevice(pdevice), props(props), id(devId) {
    validate(this->initDevice(), "initializing physical device");
    this->setupVMA();
    this->setupCommonVariables();

    this->curTimer = 0;
    for(int i = 0;i < N_TIMERS;i ++) {
        this->timer[i] = true;
    }
}

VulkanDevice::~VulkanDevice() {
    timerPool.reset();
    nearestClampSampler.reset();

    if(props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        vmaDestroyPool(this->allocator,rgba32fImagePool);
        vmaDestroyPool(this->allocator,bufferPool);
    }
    vmaDestroyAllocator(this->allocator);

    commandBuffer.reset();
    graphicsCommandPool.reset();

    for(int i = 0;i < NUM_STAGING_ARRAYS + 1;i ++) {
        transferCommandBuffer[i].reset();
        transferCommandPool[i].reset();
    }

    pipelineCache.reset();
    device.reset();

    std::cerr << "successfully destroyed device: " << props.deviceName << std::endl;
}

vk::UniqueRenderPass VulkanDevice::createRenderPass(vk::Format colorFormat, int noAttachments, bool clear) {
    std::vector<vk::AttachmentDescription> attachmentDescriptions(noAttachments);

    vk::AttachmentLoadOp clearOp = clear?vk::AttachmentLoadOp::eClear:vk::AttachmentLoadOp::eLoad;
    vk::ImageLayout beginLayout = clear?vk::ImageLayout::eUndefined:vk::ImageLayout::eGeneral;

    std::vector<vk::AttachmentReference> colorReference(noAttachments);
    for(int i = 0;i < noAttachments;i ++) {

        attachmentDescriptions[i] =  {vk::AttachmentDescriptionFlags(), colorFormat, vk::SampleCountFlagBits::e1, clearOp,
                                         vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, beginLayout, vk::ImageLayout::eGeneral};

        colorReference[i].attachment = i;
        colorReference[i].layout = vk::ImageLayout::eColorAttachmentOptimal;
    }

    vk::SubpassDescription subpass(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, 0, nullptr, noAttachments, colorReference.data(), nullptr, nullptr);
    return device->createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), noAttachments, attachmentDescriptions.data(), 1, &subpass));
}

// TODO not thread safe
int VulkanDevice::getNextTimer() {
    int st = curTimer;
    while(!timer[curTimer]) {
        curTimer = (curTimer + 1) % N_TIMERS;
        if(curTimer == st) {
            std::cerr << "Error!!! timer pool overflow!!!" << std::endl;
            exit(-1);
        }
    }
    timer[curTimer] = false;
    int ret = curTimer;
    curTimer = (curTimer + 1) % N_TIMERS;
    return ret;
}

void VulkanDevice::resetTimer(int timerIndex) {
    timer[timerIndex] = true;
}

void VulkanDevice::submit(vk::SubmitInfo &submitInfo, vk::Fence &fence, bool transfer) {
    if(transfer) {
        tmut.lock();
        this->tqueue.submit(submitInfo,fence);
        tmut.unlock();
    } else {
        cmut.lock();
        this->queue.submit(submitInfo,fence);
        cmut.unlock();
    }
}

void VulkanDevice::waitForFences(vk::Fence &fence, vk::Bool32 block, uint64_t timeout) {
    vk::Result res = this->device->waitForFences(fence, block, timeout);
    if(res != vk::Result::eSuccess) {
        LOG << "Error occured while waiting for fence!! error code: " << vk::to_string(res);
    }
}

bool VulkanDevice::initDevice() {
    std::cerr << "\nInitializing device: " << (props.deviceName) << std::endl;

    // get the QueueFamilyProperties of the PhysicalDevice
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    // get the first index into queueFamiliyProperties which supports graphics
    std::cerr << "No. of queues: " << queueFamilyProperties.size() << std::endl;

    int qindex = -1;
    std::vector<int> tqueues;
    transferQueueIndex = -1;
    for(int i = 0;i < queueFamilyProperties.size();i ++) {
        int ct = 0;
        if(queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
            std::cerr << "Found compute queue at index: " << i << std::endl;
            ct ++;
        }
        if(queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            std::cerr << "Found graphics queue at index: " << i << std::endl;
            ct ++;
        }
        if(queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eTransfer) {
            std::cerr << "Found transfer queue at index: " << i << std::endl;
            if(ct == 0) {
                transferQueueIndex = i;
            }
            tqueues.push_back(i);
        }

        if((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) && (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) && (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eTransfer)) {
            if(queueFamilyProperties[i].timestampValidBits == 0) {
                std::cerr << "Selected queue does not support time stamps!" << std::endl;
            } else {
                std::cerr << "Selected queue supports time stamps..." << std::endl;
            }
            qindex = i;
        }
    }
    if(qindex == -1) {
        std::cerr << "ERROR: could not find combined graphics, compute, and transfer queue!" << std::endl;
        return false;
    }
    std::cerr << "using combined graphics, compute, and transfer queue that was found at: " << qindex << std::endl;
    this->computeQueueIndex = this->graphicsQueueIndex = qindex;

    if(transferQueueIndex == -1) {
        for(int tq: tqueues) {
            if(tq != qindex) {
                transferQueueIndex = tq;
            }
        }
    }
    if(transferQueueIndex == -1) {
        transferQueueIndex = qindex;
    }
    std::cerr << "using transfer queue that was found at: " << transferQueueIndex << std::endl;

    // create a logical device. using UniqueDevice so it gets destroyed automatically
    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo[2];
    deviceQueueCreateInfo[0] = {vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(qindex), 1, &queuePriority};
    if(transferQueueIndex != qindex) {
        deviceQueueCreateInfo[0] = {vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(qindex), 1, &queuePriority};
        deviceQueueCreateInfo[1] = {vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(transferQueueIndex), 1, &queuePriority};
    }

    vk::PhysicalDeviceFeatures features;
    features.fragmentStoresAndAtomics = VK_TRUE;
    features.logicOp = VK_TRUE;
    features.vertexPipelineStoresAndAtomics = VK_TRUE;
    features.independentBlend = VK_TRUE;
    features.shaderInt64 = VK_TRUE;
    features.geometryShader = VK_TRUE;

    vk::PhysicalDeviceVulkan12Features features12;
    features12.shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE;
    features12.shaderBufferInt64Atomics = VK_TRUE;
    features12.shaderSharedInt64Atomics = VK_TRUE;

    vk::DeviceCreateInfo info;
    info.pEnabledFeatures = &features;
    info.flags = vk::DeviceCreateFlags();
    if(transferQueueIndex == qindex) {
        info.queueCreateInfoCount = 1;
    } else {
        info.queueCreateInfoCount = 2;
    }

    info.pQueueCreateInfos = deviceQueueCreateInfo;
    info.pEnabledFeatures = &features;

    // TODO check if some of these extensions are even supported
    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
//    extensions.push_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
#ifdef WIN32
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
    info.enabledExtensionCount = (uint32_t)extensions.size();
    info.ppEnabledExtensionNames = extensions.data();

//    if(this->props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
//        std::vector<const char *> requiredExtensions{};
//        requiredExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

//        vk::PhysicalDeviceSynchronization2Features sync2;
//        sync2.synchronization2 = VK_TRUE;

//        std::vector<const char *> enabled_extensions{};
//        std::vector<vk::ExtensionProperties> deviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
//        for(vk::ExtensionProperties e : deviceExtensions) {
//            for(auto s : requiredExtensions) {
//                if(std::string(e.extensionName.data()).compare(s) == 0) {
//                    enabled_extensions.emplace_back(s);
//                }
//            }
//        }
//        if(enabled_extensions.size() != requiredExtensions.size()) {
//            std::cerr << "sync 2 extension not supported!!!";
//            exit(0);
//        }
//        info.enabledExtensionCount = enabled_extensions.size();
//        info.ppEnabledExtensionNames = enabled_extensions.data();
//        info.pNext = (void *) &sync2;
//    }

    info.pNext = &features12;

    vk::PhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeature;
    dynamicRenderingFeature.dynamicRendering = true;
    features12.pNext = &dynamicRenderingFeature;

    device = physicalDevice.createDeviceUnique(info);

    std::cerr << "successfully created logical device" << std::endl;

    queue = device->getQueue(qindex,0);
    if(transferQueueIndex == qindex) {
        tqueue = queue;
    } else {
        tqueue = device->getQueue(transferQueueIndex,0);
    }
    std::cerr << "successfully obtained queue" << std::endl;

    pipelineCache = device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo(vk::PipelineCacheCreateFlags()));

    // graphics queue
    {
        this->graphicsCommandPool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsQueueIndex));
        std::cerr << "created graphics + transfer command pool" << std::endl;
        std::vector<vk::UniqueCommandBuffer> commandBuffers = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(graphicsCommandPool.get(), vk::CommandBufferLevel::ePrimary, 1));
        if(commandBuffers.size() == 0) {
            std::cerr << "ERROR: could not create command buffer!" << std::endl;
            exit(-2);
        }
        this->commandBuffer = std::move(commandBuffers[0]);
    }

    // transfer queue
    for(int i = 0;i < NUM_STAGING_ARRAYS + 1;i ++) {
        this->transferCommandPool[i] = device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, transferQueueIndex));
        std::cerr << "created transfer command pool " << i << std::endl;

        std::vector<vk::UniqueCommandBuffer> tcommandBuffers = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(transferCommandPool[i].get(), vk::CommandBufferLevel::ePrimary, 1));
        if(tcommandBuffers.size() == 0) {
            std::cerr << "ERROR: could not create transfer command buffer!" << std::endl;
            exit(-2);
        }
        transferCommandBuffer[i] = std::move(tcommandBuffers[0]);
    }

    std::cerr << "successfully created command buffers" << std::endl;

    vk::QueryPoolCreateInfo timerPoolCreateInfo;
    timerPoolCreateInfo.queryType = vk::QueryType::eTimestamp;
    timerPoolCreateInfo.queryCount = N_TIMERS * 2;
    timerPool = device->createQueryPoolUnique(timerPoolCreateInfo);
    std::cerr << "created timer query pool" << std::endl;

    return true;
}

void VulkanDevice::setupVMA() {
    // first create cllocator: it needs to be done once only?
    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorCreateInfo.physicalDevice = physicalDevice;
    allocatorCreateInfo.device = device.get();
    allocatorCreateInfo.instance = VkEngine::getEngine()->getVulkanInstance();
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
    vmaCreateAllocator(&allocatorCreateInfo, &allocator);

    if(props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        createSharedImagePool(vk::Format::eR32G32B32A32Sfloat);
        createSharedBufferPool();
    }
}

void VulkanDevice::createSharedImagePool(vk::Format colorFormat) {
    // create pool for shared memory
    vk::ImageCreateInfo imgCreateInfo(
                vk::ImageCreateFlags(),
                vk::ImageType::e2D,
                colorFormat,
                vk::Extent3D(10,10,1),
                1,1,
                vk::SampleCountFlagBits::e1,
                vk::ImageTiling::eLinear,
                vk::ImageUsageFlagBits::eColorAttachment|vk::ImageUsageFlagBits::eSampled);

    VmaAllocationCreateInfo sampleAllocCreateInfo = {};
    sampleAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    // TODO
//    if(props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
//        sampleAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
//           VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
//           VMA_ALLOCATION_CREATE_MAPPED_BIT;
//    }

    uint32_t memTypeIndex;
    VkResult res = vmaFindMemoryTypeIndexForImageInfo(allocator,&static_cast<const VkImageCreateInfo &>(imgCreateInfo), &sampleAllocCreateInfo, &memTypeIndex);
    if(res != VK_SUCCESS) {
        LOG << "failed to get memory type when creating custom pool";
        exit(0);
    }

    if(colorFormat == vk::Format::eR32G32B32A32Sfloat) {
        rgba32fExportAllocInfo.handleTypes = ExternalMemoryHandleBit;
        VmaPoolCreateInfo poolCreateInfo = {};
        poolCreateInfo.memoryTypeIndex = memTypeIndex;
        poolCreateInfo.pMemoryAllocateNext = &rgba32fExportAllocInfo;

        res = vmaCreatePool(allocator, &poolCreateInfo, &rgba32fImagePool);
        if(res != VK_SUCCESS) {
            LOG << "failed to create custom pool";
            exit(0);
        }
    } else {
        LOG << "color format for vma pool not supported yet!";
        exit(0);
    }
}

void VulkanDevice::createSharedBufferPool() {
    VmaAllocationCreateInfo sampleAllocCreateInfo = {};
    sampleAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    vk::BufferCreateInfo bufCreateInfo;
    bufCreateInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst;

    bufferExportAllocInfo.handleTypes = ExternalMemoryHandleBit;
    uint32_t memTypeIndex;
    VkResult res = vmaFindMemoryTypeIndexForBufferInfo(allocator,&static_cast<const VkBufferCreateInfo &>(bufCreateInfo), &sampleAllocCreateInfo, &memTypeIndex);
    if(res != VK_SUCCESS) {
        LOG << "failed to get memory type when creating custom buffer pool";
        exit(0);
    }
    VmaPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.memoryTypeIndex = memTypeIndex;
    poolCreateInfo.pMemoryAllocateNext = &bufferExportAllocInfo;

    res = vmaCreatePool(allocator, &poolCreateInfo, &bufferPool);
    if(res != VK_SUCCESS) {
        LOG << "failed to create custom buffer pool";
        exit(0);
    }
}

void VulkanDevice::setupCommonVariables() {
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eNearest;
    samplerInfo.minFilter = vk::Filter::eNearest;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.unnormalizedCoordinates = VK_TRUE;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;

    nearestClampSampler = this->device->createSamplerUnique(samplerInfo);
}

} //namespace
