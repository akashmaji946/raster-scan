// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "VulkanDevice.hpp"

#include <core/VkInclude.hpp>
#include <vma/vma.hpp>
#include <memory>

namespace vkcore {

enum class MemoryType : int {
    Internal = 0,
    LocalHostVisible,
    LocalHostVisibleForce,
    WriteOnly,
    ReadOnly,
    ReadWrite
};

struct Buffer;
typedef std::shared_ptr<Buffer> PBuffer;

struct Image {
    Image(PVkDevice vd);
    ~Image();
    void create(int width, int height, vk::Flags<vk::ImageUsageFlagBits> usageFlags, MemoryType memType, vk::ImageTiling tiling=vk::ImageTiling::eLinear, vk::Format imgFormat=vk::Format::eR32G32B32A32Sfloat, bool share=false);
    void createFromSharedHandle(int width, int height, vk::Flags<vk::ImageUsageFlagBits> usageFlags, vk::ImageTiling tiling, vk::Format imgFormat, Handle handle);
    void destroy();
    void *map();
    void unmap();
    void copyToBuffer(PBuffer buf, vk::Extent3D size, vk::Offset3D offset = vk::Offset3D(0,0,0), int32_t transferQueue=-1);
    void convertLayout(vk::PipelineStageFlags srcStage, vk::AccessFlags srcAccessMask, vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccessMask, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);

public:
    vk::Image img;
    VmaAllocation alloc;
    vk::UniqueDeviceMemory memory;
    VkMemoryPropertyFlags memPropFlags;

    bool valid, isExternal, isShared;
    vk::Flags<vk::ImageUsageFlagBits> usageFlags;
    Handle memHandle;

    PVkDevice vd;
};

typedef std::shared_ptr<Image> PImage;

class FrameBuffer {
public:
    PImage colorImg, colorImg2;
    vk::ImageView colorView, colorView2;

//    Image depthImg;
//    vk::ImageView depthView;

    vk::UniqueFramebuffer fbo;

    PVkDevice vd;
    int wd, ht;
    vk::Format colorFormat;
    bool depth, internal, valid;
    int noColAttachments;
    vk::ImageTiling tiling;

public:
    FrameBuffer(PVkDevice vd);
    ~FrameBuffer();
    void create(vk::Format colorFormat, int width, int height, int noColAttachments, MemoryType memType, bool nocopy=true, bool sampler=false, bool depth=false, bool share=false);
    void bind(vk::RenderPass &pass);
    void readData(char *data, size_t size);
    void destroy();
};
typedef std::shared_ptr<FrameBuffer> PFrameBuffer;


struct Buffer {
public:
    Buffer(PVkDevice vd);
    ~Buffer();

    void create(vk::DeviceSize size, vk::BufferUsageFlags usage, MemoryType type, bool share = false);
    void createFromSharedHandle(vk::DeviceSize size, vk::BufferUsageFlags usage, MemoryType type, Handle handle);

    virtual void destroy();
    void map();
    void unmap();
    void loadData(char *data, size_t size, size_t offset = 0);
    void readData(char *data, size_t size, size_t offset = 0);
    void clearBuffer(uint32_t fillVal = 0);
    void barrier(vk::PipelineStageFlagBits srcStage, vk::PipelineStageFlagBits dstStage, vk::Flags<vk::AccessFlagBits> srcAccessMask, vk::Flags<vk::AccessFlagBits> dstAccessMask);
    void clearBufferWithBarrier(vk::PipelineStageFlagBits dstStage, uint32_t fillVal=0);

    void copyFrom(size_t size, size_t srcOffset, size_t dstOffset, PBuffer buf, int32_t transferQueue = -1);
    
    // Get device address for buffer device address feature (bypasses 4GB maxStorageBufferRange limit)
    uint64_t getDeviceAddress();

public:
    vk::Buffer buf;
    VmaAllocation alloc;
    vk::UniqueDeviceMemory memory;
    VkMemoryPropertyFlags memPropFlags;
    Handle memHandle;

    PVkDevice vd;
    vk::DeviceSize size;
    vk::BufferUsageFlags flags;
    bool valid, mapped, isExternal, isShared;
    MemoryType type;

protected:
    void *ptr;
};

struct ImageBuffer : public Buffer {
public:
    ImageBuffer(PVkDevice vd);
    ~ImageBuffer();

    void create(vk::DeviceSize size, vk::Format format, vk::BufferUsageFlags usage, MemoryType type);
    void destroy();

public:
    vk::BufferView view;
};
typedef std::shared_ptr<ImageBuffer> PImageBuffer;


} // namespace

