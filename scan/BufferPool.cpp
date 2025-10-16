// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "BufferPool.hpp"

#include <core/VkEngine.hpp>

#include <iostream>
#ifdef WIN32
#include <intrin.h>
#endif

using namespace vkcore;

CommonBufferPool::CommonBufferPool(PVkDevice vd): vd(vd), valid(false) {
    this->initialize();
}

CommonBufferPool::~CommonBufferPool() {
    this->destroy();
}

void CommonBufferPool::initialize() {
    dummyFbo.reset(new FrameBuffer(vd));
    dummyFbo->create(vk::Format::eR8Sint,INDEX_RESOLUTION,INDEX_RESOLUTION,1,MemoryType::Internal,false);

    size_t uint4needed = (size_t) std::ceil(double(MAX_RESULT_SIZE) / sizeof(uint32_t));
    resBuffer.reset(new Buffer(vd));
    resBuffer->create(uint4needed * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc,MemoryType::Internal);

    rctBuffer.reset(new Buffer(vd));
    rctBuffer->create(MAX_QUERIES * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc,MemoryType::Internal);

    edgeBuffer.reset(new Buffer(vd));
    edgeBuffer->create(2 * INDEX_RESOLUTION * INDEX_RESOLUTION * sizeof(uint32_t), vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc,MemoryType::Internal);
    valid = true;
}

void CommonBufferPool::destroy() {
    if(!valid) {
        return;
    }
    dummyFbo->destroy();
    dummyFbo.reset();

    resBuffer->destroy();
    resBuffer.reset();

    rctBuffer->destroy();
    rctBuffer.reset();

    vd.reset();
    valid = false;
}


IndexBuffers::IndexBuffers(PVkDevice vd, uint32_t npoints, bool singleColumn, size_t scanBufSize): vd(vd), valid(false) {
    this->npoints = npoints;
    this->indexSize = INDEX_RESOLUTION * INDEX_RESOLUTION;
    this->countBufSize = size_t(std::ceil(double(indexSize + 1) / scanBufSize) * scanBufSize);
    this->singleColumn = singleColumn;
    this->initialize();
}

IndexBuffers::~IndexBuffers() {
    this->destroy();
}

void IndexBuffers::initialize() {
    cstartBuffer.reset(new Buffer(vd));
    cstartBuffer->create(countBufSize * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc,MemoryType::Internal);
    cendBuffer.reset(new Buffer(vd));
    cendBuffer->create(countBufSize * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc,MemoryType::Internal);

    uint32_t ncols = singleColumn? 2 : 4;
    indexBuffer.reset(new Buffer(vd));
    indexBuffer->create(ncols * npoints * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc,MemoryType::Internal);
    valid = true;
}

void IndexBuffers::destroy() {
    if(!valid) {
        return;
    }
    cstartBuffer->destroy();
    cstartBuffer.reset();
    cendBuffer->destroy();
    cendBuffer.reset();
    vd.reset();
    valid = false;
}
