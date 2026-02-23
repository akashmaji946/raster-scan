#pragma once

#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

#ifndef VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#endif

namespace vk_raster {

struct BufferResource {
  vk::Buffer buffer;
  vk::DeviceMemory memory;
  void *mapped = nullptr;
  vk::DeviceSize size = 0;
};

class VulkanContext {
public:
  VulkanContext(bool enableValidation = true);
  ~VulkanContext();

  // Copy constructor deleted
  VulkanContext(const VulkanContext &) = delete;
  VulkanContext &operator=(const VulkanContext &) = delete;

  // Public members for easy access in this prototype
  vk::Instance instance;
  vk::PhysicalDevice physicalDevice;
  vk::Device device;
  vk::Queue computeQueue;
  uint32_t computeQueueFamilyIndex;
  vk::CommandPool commandPool;
  vk::PipelineCache pipelineCache;

  // Helper to create a buffer
  BufferResource createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                              vk::MemoryPropertyFlags properties);

  // Helper to cleanup buffer
  void destroyBuffer(BufferResource &res);

  // Helper to run a single time command
  void runCommandBuffer(std::function<void(vk::CommandBuffer)> recorder);

  // Shader loading
  vk::ShaderModule createShaderModule(const std::vector<uint32_t> &code);
  static std::vector<uint32_t> readFile(const std::string &filename);

private:
  vk::DebugUtilsMessengerEXT debugMessenger;
  bool enableValidationLayers;

  void createInstance();
  void setupDebugMessenger();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createCommandPool();
  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties);
};

} // namespace vk_raster
