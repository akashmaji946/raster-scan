// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VkInclude.hpp>

#include <core/VulkanDevice.hpp>
#include <core/VkData.hpp>

#include <common/constants.h>

#define INDEX_RESOLUTION 1024

// requires 1 GB space for storing results
#define MAX_RESULT_SIZE (1 << 28)
#define MAX_QUERIES (1 << 24)

class CommonBufferPool {
public:
    CommonBufferPool(vkcore::PVkDevice vd);
    ~CommonBufferPool();

    void destroy();

protected:
    void initialize();

protected:
    vkcore::PVkDevice vd;
    bool valid;

public:
    // dummy FBO
    vkcore::PFrameBuffer dummyFbo;

    // storing intermediate edges
    vkcore::PBuffer edgeBuffer;
    // storing results
    vkcore::PBuffer resBuffer, rctBuffer;
};

typedef std::shared_ptr<CommonBufferPool> PBufferCache;

class IndexBuffers {
public:
    IndexBuffers(vkcore::PVkDevice vd, uint32_t npoints, bool singleColumn, size_t scanBufSize);
    ~IndexBuffers();

    void destroy();

protected:
    void initialize();

protected:
    vkcore::PVkDevice vd;
    bool valid;

public:
    vkcore::PBuffer cstartBuffer, cendBuffer, indexBuffer;
    size_t countBufSize;
    uint32_t npoints, minVal[2], maxVal[2], binRange[2];
    uint32_t indexSize, maxBinCt;
    bool singleColumn;
};

typedef std::shared_ptr<IndexBuffers> PRasterIndex;
