#include "vulkan_common.hpp"

namespace vk_raster {

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {

  std::string msg = pCallbackData->pMessage;

  // Filter spam
  if (msg.find("Copying old device") != std::string::npos ||
      msg.find("Searching for ICD") != std::string::npos ||
      msg.find("Loading layer") != std::string::npos ||
      msg.find("terminator_CreateInstance") != std::string::npos) {
    return VK_FALSE;
  }

  std::cerr << "[Validation Layer]: " << msg << std::endl;
  return VK_FALSE;
}

VulkanContext::VulkanContext(bool enableValidation)
    : enableValidationLayers(enableValidation) {
  createInstance();
  setupDebugMessenger();
  pickPhysicalDevice();
  createLogicalDevice();
  createCommandPool();
}

VulkanContext::~VulkanContext() {
  device.destroyCommandPool(commandPool);
  device.destroy();
  if (enableValidationLayers) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
      func(instance, debugMessenger, nullptr);
    }
  }
  instance.destroy();
}

void VulkanContext::createInstance() {
  vk::ApplicationInfo appInfo("RasterDB Phase 2", VK_MAKE_VERSION(1, 0, 0),
                              "No Engine", VK_MAKE_VERSION(1, 0, 0),
                              VK_API_VERSION_1_2);

  std::vector<const char *> layers;
  if (enableValidationLayers) {
    layers.push_back("VK_LAYER_KHRONOS_validation");
  }

  std::vector<const char *> extensions;
  if (enableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  vk::InstanceCreateInfo createInfo({}, &appInfo, layers.size(), layers.data(),
                                    extensions.size(), extensions.data());

  // Enable debug utils (validation messages)
  vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo;
  if (enableValidationLayers) {
    debugCreateInfo.messageSeverity =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    debugCreateInfo.messageType =
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    debugCreateInfo.pfnUserCallback = debugCallback;
    createInfo.pNext = &debugCreateInfo;
  }

  instance = vk::createInstance(createInfo);
}

void VulkanContext::setupDebugMessenger() {
  if (!enableValidationLayers)
    return;

  vk::DebugUtilsMessengerCreateInfoEXT createInfo;
  createInfo.messageSeverity =
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
  createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
  createInfo.pfnUserCallback = debugCallback;

  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    VkDebugUtilsMessengerEXT messenger;
    func(instance, reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&createInfo), nullptr,
         &messenger);
    debugMessenger = messenger;
  } else {
    std::cerr << "Failed to set up debug messenger!" << std::endl;
  }
}

void VulkanContext::pickPhysicalDevice() {
  auto devices = instance.enumeratePhysicalDevices();
  for (const auto &dev : devices) {
    auto props = dev.getProperties();
    if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
      physicalDevice = dev;
      std::cout << "Picked Discrete GPU: " << props.deviceName << std::endl;
      return;
    }
  }
  if (devices.size() > 0) {
    physicalDevice = devices[0];
    std::cout << "Picked Fallback GPU: "
              << physicalDevice.getProperties().deviceName << std::endl;
  } else {
    throw std::runtime_error("No Vulkan physical devices found!");
  }
}

void VulkanContext::createLogicalDevice() {
  auto queueFamilies = physicalDevice.getQueueFamilyProperties();
  int i = 0;
  int computeFamily = -1;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) {
      computeFamily = i;
      break;
    }
    i++;
  }

  if (computeFamily == -1)
    throw std::runtime_error("No compute queue found!");
  computeQueueFamilyIndex = computeFamily;

  float queuePriority = 1.0f;
  vk::DeviceQueueCreateInfo queueCreateInfo({}, computeFamily, 1,
                                            &queuePriority);

  std::vector<const char *> deviceLayers;
  if (enableValidationLayers) {
    deviceLayers.push_back("VK_LAYER_KHRONOS_validation");
  }

  vk::PhysicalDeviceFeatures deviceFeatures;
  deviceFeatures.shaderFloat64 = VK_TRUE;

  vk::DeviceCreateInfo createInfo({}, 1, &queueCreateInfo, deviceLayers.size(),
                                  deviceLayers.data(), 0, nullptr,
                                  &deviceFeatures);
  device = physicalDevice.createDevice(createInfo);
  computeQueue = device.getQueue(computeFamily, 0);
}

void VulkanContext::createCommandPool() {
  vk::CommandPoolCreateInfo poolInfo(
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      computeQueueFamilyIndex);
  commandPool = device.createCommandPool(poolInfo);
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter,
                                       vk::MemoryPropertyFlags properties) {
  vk::PhysicalDeviceMemoryProperties memProperties =
      physicalDevice.getMemoryProperties();
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find suitable memory type!");
}

BufferResource VulkanContext::createBuffer(vk::DeviceSize size,
                                           vk::BufferUsageFlags usage,
                                           vk::MemoryPropertyFlags properties) {
  BufferResource res;
  res.size = size;

  vk::BufferCreateInfo bufferInfo({}, size, usage, vk::SharingMode::eExclusive);
  res.buffer = device.createBuffer(bufferInfo);

  vk::MemoryRequirements memRequirements =
      device.getBufferMemoryRequirements(res.buffer);

  vk::MemoryAllocateInfo allocInfo(
      memRequirements.size,
      findMemoryType(memRequirements.memoryTypeBits, properties));

  // Enable buffer address feature if needed? Not for SSBO.
  res.memory = device.allocateMemory(allocInfo);
  device.bindBufferMemory(res.buffer, res.memory, 0);

  if (properties & vk::MemoryPropertyFlagBits::eHostVisible) {
    res.mapped = device.mapMemory(res.memory, 0, size);
  }

  return res;
}

void VulkanContext::destroyBuffer(BufferResource &res) {
  if (res.buffer) {
    device.destroyBuffer(res.buffer);
  }
  if (res.memory) {
    if (res.mapped) {
      device.unmapMemory(res.memory);
    }
    device.freeMemory(res.memory);
  }
}

void VulkanContext::runCommandBuffer(
    std::function<void(vk::CommandBuffer)> recorder) {
  vk::CommandBufferAllocateInfo allocInfo(commandPool,
                                          vk::CommandBufferLevel::ePrimary, 1);
  auto commandBuffers = device.allocateCommandBuffers(allocInfo);
  vk::CommandBuffer &cmd = commandBuffers[0];

  vk::CommandBufferBeginInfo beginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  cmd.begin(beginInfo);
  recorder(cmd);
  cmd.end();

  vk::SubmitInfo submitInfo;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;

  computeQueue.submit(submitInfo, nullptr);
  computeQueue.waitIdle();

  device.freeCommandBuffers(commandPool, commandBuffers);
}

vk::ShaderModule
VulkanContext::createShaderModule(const std::vector<uint32_t> &code) {
  vk::ShaderModuleCreateInfo createInfo({}, code.size() * 4, code.data());
  return device.createShaderModule(createInfo);
}

std::vector<uint32_t> VulkanContext::readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file! " + filename);
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<uint32_t> buffer(fileSize / 4);
  file.seekg(0);
  file.read((char *)buffer.data(), fileSize);
  file.close();

  return buffer;
}

} // namespace vk_raster
