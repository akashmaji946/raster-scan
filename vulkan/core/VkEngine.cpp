// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "VkEngine.hpp"

#include <common/utils.h>

#include <iostream>
#include <vector>


#if VK_HEADER_VERSION >= 301
using VulkanDynamicLoader = vk::detail::DynamicLoader;
#else
using VulkanDynamicLoader = vk::DynamicLoader;
#endif

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace vkcore {
#ifdef DEV_BUILD
//inline static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {

inline static vk::Bool32 debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageType,
        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData){

    if(messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        std::cerr << std::endl << "validation layer: " << pCallbackData->pMessage << std::endl << std::endl;
    }
    return VK_FALSE;
}
#endif

VkEngine* VkEngine::engine = NULL;

VkEngine::VkEngine() {
    validate(createInstance(),"createInstance");
    defaultDevice = -1;
}

VkEngine::~VkEngine() {
    devices.clear();
#ifdef DEV_BUILD
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) instance.getProcAddr("vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger,nullptr);
    } else {
        std::cerr << "ERROR: could not find debug destroy function!" << std::endl;
    }
#endif
    std::cerr << "successfully destroyed instance" << std::endl;

}

void VkEngine::destroy() {
    delete engine;
}

VkEngine *VkEngine::getEngine() {
    if(engine == NULL) {
        engine = new VkEngine();
        validate(engine->createDevices(),"createDevices");
    }
    return engine;
}

vk::Instance VkEngine::getVulkanInstance() {
    return this->instance;
}

// return device with maximum heap
int32_t VkEngine::getDefaultDeviceId(bool forceIntegrated) {
    if(defaultDevice != -1) {
        return defaultDevice;
    }
    int did = -1;
    vk::DeviceSize max = 0;
    if(!forceIntegrated) {
        for(int i = 0;i < devices.size();i ++) {
            if(devices[i]->props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                if(max < devices[i]->heapSize) {
                    did = i;
                    max = devices[i]->heapSize;
                }
            }
        }
    } else {
        for(int i = 0;i < devices.size();i ++) {
            if(devices[i]->props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
                if(max < devices[i]->heapSize) {
                    did = i;
                    max = devices[i]->heapSize;
                }
            }
        }
    }
    if(did == -1) {
        did = 0;
    }
    return did;
}

PVkDevice VkEngine::getDevice(int did) {
    return this->devices[did];
}

bool VkEngine::createInstance() {
    static VulkanDynamicLoader dl{};
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    std::vector<const char*> layers;
    std::vector<const char*> extensions;

    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);

#ifdef DEV_BUILD
    std::vector<const char*> validationLayers = {
        "VK_LAYER_LUNARG_standard_validation",
        "VK_LAYER_KHRONOS_validation"
    };

    auto installedLayers = vk::enumerateInstanceLayerProperties();

    for (auto &w : validationLayers) {
        for (auto &i : installedLayers) {
            if (std::string(i.layerName.data()).compare(w) == 0) {
                layers.emplace_back(w);
                break;
            }
        }
    }
    const char *validationExt = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    extensions.push_back(validationExt);
#endif

    vk::ApplicationInfo appInfo("App Name",1,"engineName",1,VK_MAKE_VERSION(1, 3, 0));
    vk::InstanceCreateInfo instanceInfo({},&appInfo,(uint32_t)layers.size(),layers.data(),(uint32_t)extensions.size(),extensions.data());
    vk::Result res = vk::createInstance(&instanceInfo, nullptr, &this->instance);
    if(res != vk::Result::eSuccess) {
        std::cerr << "*********************** could not create instance ***********************" << vk::to_string(res) << std::endl;
        return false;
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

#ifdef DEV_BUILD
    if(layers.size() > 0) {
        vk::DebugUtilsMessengerCreateInfoEXT debugInfo;
        debugInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        debugInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
        debugInfo.pfnUserCallback = debugCallback;

        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) instance.getProcAddr("vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr) {
            VkResult res = func(instance, reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&debugInfo), nullptr, reinterpret_cast<VkDebugUtilsMessengerEXT*>(&debugMessenger));
            if(res != VK_SUCCESS) {
                std::cerr << "*********************** could not setup debug messenger ***********************" << std::endl;
                return false;
            }
        } else {
            std::cerr << "ERROR: could not find debug function!" << std::endl;
            return false;
        }
    }
#endif
    return true;
}

bool VkEngine::createDevices() {
    std::vector<vk::PhysicalDevice> physicalDevices = this->instance.enumeratePhysicalDevices();
    std::cerr << "no. of devices: " << physicalDevices.size() << std::endl;
    if(physicalDevices.size() == 0) {
        std::cerr << "ERROR: found no vulkan devices!" << std::endl;
        return false;
    }

    for(int i = 0;i < physicalDevices.size();i ++) {
        LOG << std::endl << std::endl << "Setting up physical device";
        vk::PhysicalDeviceProperties props = physicalDevices[i].getProperties();

        VkPhysicalDeviceSubgroupProperties subgroupProperties;
        subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        subgroupProperties.pNext = NULL;

        VkPhysicalDeviceProperties2 physicalDeviceProperties;
        physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        physicalDeviceProperties.pNext = &subgroupProperties;

        vkGetPhysicalDeviceProperties2((VkPhysicalDevice)physicalDevices[i], &physicalDeviceProperties);
        uint32_t subgroupSize = subgroupProperties.subgroupSize;
        LOG << "subgroup size: " << subgroupProperties.subgroupSize;

        vk::PhysicalDeviceMemoryProperties memProps = physicalDevices[i].getMemoryProperties();
        vk::DeviceSize hsize = 0;
        bool rebar = false;
        uint32_t localHeapCt = 0;
        uint32_t rebarEnablesCt = 0;
        for(uint32_t h = 0;h < memProps.memoryHeapCount;h ++) {
            if(memProps.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                localHeapCt ++;
                hsize += memProps.memoryHeaps[h].size;
                LOG << "Device local heap " << h << ": " << memProps.memoryHeaps[h].size;
                bool rebarFlags = false;
                for(uint32_t t = 0;t < memProps.memoryTypeCount;t ++) {
                    if(memProps.memoryTypes[t].heapIndex == h) {
                        if((memProps.memoryTypes[t].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)
                                && (memProps.memoryTypes[t].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
                            rebarFlags = true;
                        }
                    }
                }
                if(rebarFlags) {
                    rebarEnablesCt ++;
                }
            }
        }

        // TODO is this check correct?
        if(localHeapCt == rebarEnablesCt) {
            rebar = true;
        }

        double memsize = hsize / (1024.0 * 1024 * 1024);
        LOG << "Device " << props.deviceName << " has ID: " << i <<  " and Heap size: " << memsize << " GB";

        PVkDevice dev(new VulkanDevice(physicalDevices[i],props,i));
        dev->subgroupSize = subgroupSize;
        dev->memProps = memProps;
        dev->rebarEnabled = rebar;
        dev->heapSize = hsize;
        this->devices.push_back(dev);
    }
    return true;
}

} //namespace
