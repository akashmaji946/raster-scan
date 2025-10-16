// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VkInclude.hpp>

#include "VulkanDevice.hpp"

namespace vkcore {

class VkEngine
{
private:
    VkEngine();
    ~VkEngine();
    static VkEngine* engine;

public:
    static VkEngine* getEngine();
    static void destroy();

public:
    vk::Instance getVulkanInstance();
    int32_t getDefaultDeviceId(bool forceIntegrated=false);
    PVkDevice getDevice(int did);

protected:
    bool createInstance();
    bool createDevices();

protected:
    vk::Instance instance;
    std::vector<PVkDevice> devices;
    int32_t defaultDevice;

    /////////// vulkan debug
    vk::DebugUtilsMessengerEXT debugMessenger;
};

} // namespace

