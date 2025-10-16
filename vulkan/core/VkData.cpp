// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "VkData.hpp"

#include <common/utils.h>
#include <iostream>

namespace vkcore {

inline VmaAllocationCreateFlags getVmaFlags(PVkDevice vd, MemoryType memType) {
    VmaAllocationCreateFlags flags = 0;
    if(memType == MemoryType::LocalHostVisible && vd->rebarEnabled) {
        flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else if(memType == MemoryType::LocalHostVisibleForce) {
        flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    else if(memType == MemoryType::ReadOnly) {
        flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    } else if(memType == MemoryType::WriteOnly) {
        flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    } else if(memType == MemoryType::ReadWrite) {
        flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    }
    return flags;
}

Image::Image(PVkDevice vd) : valid(false), vd(vd) {
}

Image::~Image() {
    this->destroy();
}

void Image::create(int width, int height, vk::Flags<vk::ImageUsageFlagBits> usageFlags, MemoryType memType, vk::ImageTiling tiling, vk::Format imgFormat, bool share) {
    this->isShared = false;
    this->usageFlags = usageFlags;
    vk::ImageCreateInfo imgCreateInfo(
                vk::ImageCreateFlags(),
                vk::ImageType::e2D,
                imgFormat,
                vk::Extent3D(width,height,1),
                1,1,
                vk::SampleCountFlagBits::e1,
                tiling,
                usageFlags);
    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = getVmaFlags(vd,memType);

    vk::ExternalMemoryImageCreateInfo external_memory_image_create_info;
    if(share) {
        if(imgFormat == vk::Format::eR32G32B32A32Sfloat) {
            external_memory_image_create_info.handleTypes = ExternalMemoryHandleBit;
            imgCreateInfo.pNext = &external_memory_image_create_info;
            allocCreateInfo.pool = vd->rgba32fImagePool;
        } else {
            LOG << "ERROR: Input image format for creating shared resource not supported yet";
            exit(-1);
        }
        isShared = true;
    }
    vmaCreateImage(vd->allocator, &static_cast<const VkImageCreateInfo&>(imgCreateInfo), &allocCreateInfo, reinterpret_cast<VkImage*>(&img), &alloc, nullptr);
    vmaGetAllocationMemoryProperties(vd->allocator, alloc, &memPropFlags);

    if(share) {
        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(vd->allocator,alloc,&allocInfo);
#ifdef WIN32
        vk::MemoryGetWin32HandleInfoKHR memoryFdInfo {allocInfo.deviceMemory, ExternalMemoryHandleBit};
        vk::Result res = vd->device->getMemoryWin32HandleKHR(&memoryFdInfo, &memHandle);
        if(res != vk::Result::eSuccess) {
            LOG << "error occured when trying to get Win32 Memory handle! for Image" << vk::to_string(res);
        }
#else
        vk::MemoryGetFdInfoKHR memoryFdInfo {allocInfo.deviceMemory, ExternalMemoryHandleBit};
        vd->device->getMemoryFdKHR(&memoryFdInfo, &memHandle);
#endif
    }

    valid = true;
    isExternal = false;
}

void Image::createFromSharedHandle(int width, int height, vk::Flags<vk::ImageUsageFlagBits> usageFlags, vk::ImageTiling tiling, vk::Format imgFormat, Handle handle) {
    vk::ImageCreateInfo imgCreateInfo(
                vk::ImageCreateFlags(),
                vk::ImageType::e2D,
                imgFormat,
                vk::Extent3D(width,height,1),
                1,1,
                vk::SampleCountFlagBits::e1,
                tiling,
                usageFlags);

    vk::ExternalMemoryImageCreateInfo external_memory_image_create_info;
    external_memory_image_create_info.handleTypes = ExternalMemoryHandleBit; // VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    imgCreateInfo.pNext = &external_memory_image_create_info;

    img = vd->device->createImage(imgCreateInfo);
#ifdef WIN32
    vk::ImportMemoryWin32HandleInfoKHR importMemInfo;
    importMemInfo.handle = handle;
#else
    vk::ImportMemoryFdInfoKHR importMemInfo;
    importMemInfo.fd = handle;
#endif
    importMemInfo.handleType = ExternalMemoryHandleBit;

    vk::MemoryRequirements memoryRequirements = vd->device->getImageMemoryRequirements(img);
    uint32_t typeBits = memoryRequirements.memoryTypeBits;
    uint32_t typeIndex = uint32_t(~0);
    for (uint32_t i = 0; (1u << i) <= typeBits; i++) {
        if ((typeBits & (1u << i)) && ((vd->memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            typeIndex = i;
            break;
        }
    }

    vk::MemoryAllocateInfo memAllocInfo(memoryRequirements.size);
    memAllocInfo.pNext = &importMemInfo;
    memAllocInfo.memoryTypeIndex = typeIndex;
    memory = vd->device->allocateMemoryUnique(memAllocInfo);
    vd->device->bindImageMemory(img, memory.get(), 0);

    isExternal = true;
    valid = true;
    isShared = false;
}

void Image::destroy() {
    if(valid) {
        if(!isExternal) {
            vmaDestroyImage(vd->allocator,img,alloc);
        } else {
            vd->device->destroyImage(img);
        }
    }
    valid = false;
}

void *Image::map() {
    void *pData;
    vmaMapMemory(vd->allocator,alloc,&pData);
    return pData;
}

void Image::unmap() {
    vmaUnmapMemory(vd->allocator,alloc);
}

void Image::copyToBuffer(PBuffer buf, vk::Extent3D size, vk::Offset3D offset, int32_t transferQueue) {
    vk::BufferImageCopy imgcpy;
    imgcpy.bufferOffset = 0;
    imgcpy.bufferRowLength = size.width;
    imgcpy.bufferImageHeight = size.height;
    imgcpy.imageExtent = size;
    imgcpy.imageOffset = offset;
    imgcpy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    imgcpy.imageSubresource.layerCount = 1;
    if(transferQueue != -1) {
        vd->transferCommandBuffer[transferQueue]->copyImageToBuffer(img,vk::ImageLayout::eGeneral,buf->buf,1,&imgcpy);
    } else {
        vd->commandBuffer->copyImageToBuffer(img,vk::ImageLayout::eGeneral,buf->buf,1,&imgcpy);
    }
}

void Image::convertLayout(vk::PipelineStageFlags srcStage, vk::AccessFlags srcAccessMask, vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccessMask, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    vk::ImageMemoryBarrier barrier(srcAccessMask,dstAccessMask,
                                   oldLayout,newLayout,
                                   VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,
                                   this->img,vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor,0,1,0,1));
    vd->commandBuffer->pipelineBarrier(srcStage,dstStage,vk::DependencyFlags(),
                                   nullptr,nullptr,
                                   barrier);
}


FrameBuffer::FrameBuffer(PVkDevice vd): valid(false), vd(vd) {
    colorImg = PImage(new Image(vd));
    colorImg2 = PImage(new Image(vd));
}

FrameBuffer::~FrameBuffer() {
    this->destroy();
}

void FrameBuffer::create(vk::Format colorFormat, int width, int height, int noColAttachments, MemoryType memType, bool nocopy, bool sampler, bool depth, bool share) {
    this->colorFormat = colorFormat;
    this->wd = width;
    this->ht = height;
    this->depth = depth;
    this->internal = internal;
    this->noColAttachments = noColAttachments;

    if((noColAttachments < 1) || (noColAttachments > 2)) {
        throw std::runtime_error("number of ColorAttachment shoud be 1 or 2.");
    }

    vk::FormatProperties formatProperties = vd->physicalDevice.getFormatProperties(colorFormat);
    if (formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eColorAttachment) {
        tiling = vk::ImageTiling::eOptimal;
    } else if (formatProperties.linearTilingFeatures & vk::FormatFeatureFlagBits::eColorAttachment) {
        tiling = vk::ImageTiling::eLinear;
    } else {
        throw std::runtime_error("ColorAttachment is not supported for given color format.");
    }


    // TODO use sampled only when used as texture.
    vk::Flags<vk::ImageUsageFlagBits> flags = vk::ImageUsageFlagBits::eColorAttachment;
    if(!nocopy) {
        flags |= vk::ImageUsageFlagBits::eTransferSrc;
    }
    if(sampler) {
        flags |= vk::ImageUsageFlagBits::eSampled;
    }

    colorImg->create(width,height,flags,memType,tiling,colorFormat,share);

    if(noColAttachments == 2) {
        colorImg2->create(width,height,flags,memType,tiling,colorFormat,share);
    }

    if(depth) {
        // TODO
        std::cerr << "depth attachment to be defined!!!" << std::endl;
        exit(0);
    }

    vk::ComponentMapping componentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);
    vk::ImageSubresourceRange subResourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

    colorView = vd->device->createImageView(vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), colorImg->img, vk::ImageViewType::e2D, colorFormat, componentMapping, subResourceRange, nullptr));
    if(noColAttachments == 2) {
        colorView2 = vd->device->createImageView(vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), colorImg2->img, vk::ImageViewType::e2D, colorFormat, componentMapping, subResourceRange, nullptr));
    }
    valid = true;
}

void FrameBuffer::bind(vk::RenderPass &pass) {
    if(depth) {
//        vk::ImageView attachments[2];
//        attachments[0] = colorView;
//        attachments[1] = depthView;
//        fbo = vd->device->createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), pass, 2, attachments, wd, ht, 1));
        std::cerr << "error! depth not yet supported" << std::endl;
        exit(-2);
    } else {
        vk::ImageView attachments[2];
        attachments[0] = colorView;
        if(noColAttachments == 2) {
            attachments[1] = colorView2;
        }
        fbo = vd->device->createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), pass, noColAttachments, attachments, wd, ht, 1));
    }
}

void FrameBuffer::readData(char *data, size_t size) {
    // Create the image
    PImage dstImage(new Image(vd));
    dstImage->create(wd,ht,vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eTransferSrc,MemoryType::ReadOnly,vk::ImageTiling::eLinear,this->colorFormat);

    // Do the actual blit from the offscreen image to our host visible destination image
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));

    // Transition destination image to transfer destination layout
    vk::ImageMemoryBarrier barrier(vk::AccessFlags(),vk::AccessFlagBits::eTransferWrite,
                                   vk::ImageLayout::eUndefined,vk::ImageLayout::eTransferDstOptimal,
                                   VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,
                                   dstImage->img,vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor,0,1,0,1));
    vd->commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags(),
                                       nullptr,nullptr,
                                       barrier);


    // transition colorAttachment.image to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    vk::ImageMemoryBarrier transbarrier(vk::AccessFlags(),vk::AccessFlagBits::eTransferRead,
                                   vk::ImageLayout::eUndefined,vk::ImageLayout::eTransferSrcOptimal,
                                   VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,
                                   colorImg->img,vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor,0,1,0,1));
    vd->commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags(),
                                       nullptr,nullptr,
                                       transbarrier);

    vk::ImageCopy imgCpy(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor,0,0,1),
                         vk::Offset3D(),
                         vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor,0,0,1),
                         vk::Offset3D(),vk::Extent3D(wd,ht,1));
    vd->commandBuffer->copyImage(colorImg->img, vk::ImageLayout::eTransferSrcOptimal,
                                 dstImage->img,vk::ImageLayout::eTransferDstOptimal,
                                 imgCpy);


    // Transition destination image to general layout, which is the required layout for mapping the image memory later on
    vk::ImageMemoryBarrier barrier2(vk::AccessFlagBits::eTransferWrite,vk::AccessFlagBits::eMemoryRead,
                                    vk::ImageLayout::eTransferDstOptimal,vk::ImageLayout::eGeneral,
                                    VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,
                                    dstImage->img,vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor,0,1,0,1));
    vd->commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags(),
                                       nullptr,nullptr,
                                       barrier2);

    vd->commandBuffer->end();
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo,drawFence.get(),false);
    vd->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);

    void* imagedata = dstImage->map();
    memcpy(data,imagedata,size);
    dstImage->unmap();
    dstImage.reset();
}

void FrameBuffer::destroy() {
    if(valid) {
        vd->device->destroyImageView(colorView);
        colorImg.reset();

        if(noColAttachments == 2) {
            vd->device->destroyImageView(colorView2);
            colorImg2.reset();
        }
    }
    valid = false;
}

Buffer::Buffer(PVkDevice vd) : vd(vd), type(MemoryType::Internal), valid(false), mapped(false), size(0) {
}

Buffer::~Buffer() {
    if(valid) {
        this->destroy();
    }
}

void Buffer::create(vk::DeviceSize size, vk::BufferUsageFlags usage, MemoryType type, bool share) {
    if(valid) {
        if(size <= this->size && usage == this->flags && type == this->type) {
            return;
        }
        this->destroy();
    }
    this->isShared = false;
    this->isExternal = false;
    this->flags = usage;
    this->type = type;

    vk::BufferCreateInfo bufferCreateInfo;
    VmaAllocationCreateInfo allocCreateInfo = {};

    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    bufferCreateInfo.usage = usage;
    allocCreateInfo.flags = getVmaFlags(vd, type);

    bufferCreateInfo.size = size;
    this->size = size;

    vk::ExternalMemoryBufferCreateInfo external_memory_buffer_create_info;
    if(share) {
        if(usage == (vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst)) {
            external_memory_buffer_create_info.handleTypes = ExternalMemoryHandleBit;
            bufferCreateInfo.pNext = &external_memory_buffer_create_info;
            allocCreateInfo.pool = vd->bufferPool;
        } else {
            LOG << "ERROR: Input image format for creating shared resource not supported yet";
            exit(-1);
        }
        isShared = true;
    }

    vmaCreateBuffer(vd->allocator, &static_cast<const VkBufferCreateInfo&>(bufferCreateInfo), &allocCreateInfo, reinterpret_cast<VkBuffer*>(&buf), &alloc, nullptr);
    vmaGetAllocationMemoryProperties(vd->allocator, alloc, &memPropFlags);

    if(share) {
        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(vd->allocator,alloc,&allocInfo);
#ifdef WIN32
        vk::MemoryGetWin32HandleInfoKHR memoryFdInfo {allocInfo.deviceMemory, ExternalMemoryHandleBit};
        vk::Result res = vd->device->getMemoryWin32HandleKHR(&memoryFdInfo, &memHandle);
        if(res != vk::Result::eSuccess) {
            LOG << "error occured when trying to get Win32 Memory handle for buffer!" << vk::to_string(res);
        }

#else
        vk::MemoryGetFdInfoKHR memoryFdInfo {allocInfo.deviceMemory, ExternalMemoryHandleBit};
        vd->device->getMemoryFdKHR(&memoryFdInfo, &memHandle);
#endif
    }

    valid = true;
}

void Buffer::createFromSharedHandle(vk::DeviceSize size, vk::BufferUsageFlags usage, MemoryType type, Handle handle) {
    vk::BufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.usage = usage;
    bufferCreateInfo.size = size;
    this->size = size;
    this->type = type;

    vk::ExternalMemoryBufferCreateInfo external_memory_buffer_create_info;
    external_memory_buffer_create_info.handleTypes = ExternalMemoryHandleBit;
    bufferCreateInfo.pNext = &external_memory_buffer_create_info;


    buf = vd->device->createBuffer(bufferCreateInfo);
#ifdef WIN32
    vk::ImportMemoryWin32HandleInfoKHR importMemInfo;
    importMemInfo.handle = handle;
#else
    vk::ImportMemoryFdInfoKHR importMemInfo;
    importMemInfo.fd = handle;
#endif
    importMemInfo.handleType = ExternalMemoryHandleBit;

    vk::MemoryRequirements memoryRequirements = vd->device->getBufferMemoryRequirements(buf);
    uint32_t typeBits = memoryRequirements.memoryTypeBits;
    uint32_t typeIndex = uint32_t(~0);
    for (uint32_t i = 0; (1u << i) <= typeBits; i++) {
        if ((typeBits & (1u << i)) && ((vd->memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            typeIndex = i;
            break;
        }
    }

    vk::MemoryAllocateInfo memAllocInfo(memoryRequirements.size);
    memAllocInfo.pNext = &importMemInfo;
    memAllocInfo.memoryTypeIndex = typeIndex;
    memory = vd->device->allocateMemoryUnique(memAllocInfo);
    vd->device->bindBufferMemory(buf, memory.get(), 0);

    isExternal = true;
    valid = true;
    isShared = false;
}

void Buffer::loadData(char *data, size_t size, size_t offset) {
    if((size + offset) > this->size) {
        if(offset > 0) {
            std::cerr << "---------> size not sufficient for copy... old data will be lost!!" << std::endl;
            exit(0);
        }
        this->destroy();
        this->create(size + offset,flags,type);
    }
    if(!mapped) {
        this->map();
    }
    memcpy(((char*)ptr)+offset,data,size);
}

void Buffer::readData(char *data, size_t size, size_t offset) {
    if(!valid) {
        return;
    }
    if(!this->mapped) {
        this->map();
    }
    memcpy(data,((char*)ptr)+offset,size);
}

void Buffer::clearBuffer(uint32_t fillVal) {
    vd->commandBuffer->fillBuffer(buf,0,size,fillVal);
}

void Buffer::barrier(vk::PipelineStageFlagBits srcStage, vk::PipelineStageFlagBits dstStage, vk::Flags<vk::AccessFlagBits> srcAccessMask, vk::Flags<vk::AccessFlagBits> dstAccessMask) {
    vk::BufferMemoryBarrier bufferBarrier;
    bufferBarrier.buffer = buf;
    bufferBarrier.size = VK_WHOLE_SIZE;
    bufferBarrier.srcAccessMask = srcAccessMask; //vk::AccessFlagBits::eTransferWrite;
    bufferBarrier.dstAccessMask = dstAccessMask; //vk::AccessFlagBits::eShaderRead;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vd->commandBuffer->pipelineBarrier(srcStage, dstStage, {}, nullptr, bufferBarrier, nullptr);
}

void Buffer::clearBufferWithBarrier(vk::PipelineStageFlagBits dstStage,uint32_t fillVal) {
    this->clearBuffer(fillVal);
    this->barrier(vk::PipelineStageFlagBits::eTransfer, dstStage, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
}

void Buffer::copyFrom(size_t size, size_t srcOffset, size_t dstOffset, PBuffer buf, int32_t transferQueue) {
    if(size + dstOffset > this->size) {
        std::cerr << "Size not sufficient to copy!!!";
        std::exit(0);
    }
    vk::BufferCopy copyRegion{ srcOffset, dstOffset, size };
    if(transferQueue == -1) {
        vd->commandBuffer->copyBuffer(buf->buf, this->buf, copyRegion);
    } else {
        vd->transferCommandBuffer[transferQueue]->copyBuffer(buf->buf, this->buf, copyRegion);
    }
}

void Buffer::destroy() {
    if(!valid)
        return;

    if(mapped) {
        this->unmap();
    }
    if(!isExternal) {
        vmaDestroyBuffer(vd->allocator,buf,alloc);
    } else {
        vd->device->destroyBuffer(buf);
    }
    valid = false;
}

void Buffer::map() {
    vmaMapMemory(vd->allocator,alloc,&(ptr));
    mapped = true;
}

void Buffer::unmap() {
    vmaUnmapMemory(vd->allocator,alloc);
    mapped = false;
}

ImageBuffer::ImageBuffer(PVkDevice vd) : Buffer(vd) {
}

ImageBuffer::~ImageBuffer() {
    this->destroy();
}

void ImageBuffer::create(vk::DeviceSize size, vk::Format format, vk::BufferUsageFlags usage, MemoryType type) {
    if(valid) {
        this->destroy();
    }
    Buffer::create(size,usage | vk::BufferUsageFlagBits::eStorageTexelBuffer,type);
    view = vd->device->createBufferView(vk::BufferViewCreateInfo(vk::BufferViewCreateFlagBits(),buf,format,0,VK_WHOLE_SIZE));
}

void ImageBuffer::destroy() {
    if(!valid)
        return;
    vd->device->destroyBufferView(view);
    Buffer::destroy();
}

} // namespace
