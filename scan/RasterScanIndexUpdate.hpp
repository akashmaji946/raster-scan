// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VkEngine.hpp>
#include <core/vkutils.h>
#include <core/GraphicsPipelineProperties.hpp>

#include <common/utils.h>

#include "BufferPool.hpp"

// Page size: number of data items per page
#define PAGE_DATA_SIZE 16

// Page structure layout in GPU buffer:
// struct Page {
//     uvec4 data[PAGE_DATA_SIZE];  // 3 columns (x, y, z) + rowId stored as uvec4
//     uint count;                   // Number of items currently in this page
//     uint nextPagePtr;             // Pointer to the next page (or NULL_PAGE_PTR)
// };
#define NULL_PAGE_PTR 0xFFFFFFFF

// Size of a page in uint32_t units
// PAGE_DATA_SIZE * 4 (uvec4) + 1 (count) + 1 (nextPagePtr)
#define PAGE_SIZE_UINTS (PAGE_DATA_SIZE * 4 + 2)

// Maximum number of pages that can be allocated
// Note: With 1-item-per-page approach, this limits max data points
// Memory usage: MAX_PAGES * PAGE_SIZE_UINTS * 4 bytes
// 8M pages * 66 * 4 = ~2.1 GB
#define MAX_PAGES (1 << 23)  // 8M pages

class PageAllocator {
public:
    PageAllocator(vkcore::PVkDevice vd, uint32_t maxPages);
    ~PageAllocator();

    void initialize();
    void destroy();

public:
    vkcore::PVkDevice vd;
    
    // GPU buffer storing all pages
    vkcore::PBuffer pageBuffer;
    
    // GPU buffer for atomic page allocation counter
    vkcore::PBuffer allocCounterBuffer;
    
    uint32_t maxPages;
    size_t pageBufferSize;
    bool valid;
};

typedef std::shared_ptr<PageAllocator> PPageAllocator;

class LinkedListIndex {
public:
    LinkedListIndex(vkcore::PVkDevice vd, uint32_t npoints, PPageAllocator pageAlloc);
    ~LinkedListIndex();

    void initialize();
    void destroy();

public:
    vkcore::PVkDevice vd;
    PPageAllocator pageAlloc;
    
    // Index texture buffer: stores head pointer for each cell
    // Size: INDEX_RESOLUTION * INDEX_RESOLUTION
    // Value: page pointer (index into pageBuffer) or NULL_PAGE_PTR
    vkcore::PBuffer headPtrBuffer;
    
    uint32_t npoints;
    uint32_t minVal[2], maxVal[2], binRange[2];
    uint32_t indexSize;
    bool valid;
};

typedef std::shared_ptr<LinkedListIndex> PLinkedListIndex;

class RasterScanIndexUpdate
{
public:
    RasterScanIndexUpdate(vkcore::PVkDevice vd, PBufferCache bufs, int32_t ncols);
    ~RasterScanIndexUpdate();

public:
    void initialize();

public:
    // Build initial index from data points
    PLinkedListIndex buildIndex(vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal);
    
    // Insert new data points into existing index
    void insertPoints(PLinkedListIndex index, vkcore::PBuffer pointsBuffer, uint32_t npoints);
    
    // Delete data points from index (mark as deleted)
    void deletePoints(PLinkedListIndex index, vkcore::PBuffer pointsBuffer, uint32_t npoints);
    
    // Run range queries on the linked list index
    void runRangeQueries(PLinkedListIndex index, vkcore::PBuffer qranges, uint32_t nqueries);

protected:
    void initShaders();
    void initBuffers();
    void setupInsertPipeline();
    void setupQueryTexturePipeline();
    void setupQueryPagePipeline();

protected:
    void runInsertPipeline(vkcore::PBuffer pointsBuffer, PLinkedListIndex index, uint32_t npoints);
    void runQueryTexturePipeline(PLinkedListIndex index, vkcore::PBuffer qranges, uint32_t nqueries);
    void runQueryPagePipeline(PLinkedListIndex index, vkcore::PBuffer qranges, uint32_t nqueries);

protected:
    int32_t ncols;

public:
    vkcore::PVkDevice vd;
    PBufferCache bufs;
    PPageAllocator pageAlloc;
    vkcore::PBuffer maxBuffer;

    // Pipelines
    vkcore::GraphicsPipelineProperties insertPipelineProps;
    vkcore::GraphicsPipelineProperties queryTexturePipelineProps;
    vkcore::GraphicsPipelineProperties queryPagePipelineProps;
    
    vk::UniquePipeline insertPipeline;
    vk::UniquePipeline queryTexturePipeline;
    vk::UniquePipeline queryPagePipeline;

    // Shaders
    vk::UniqueShaderModule dummyFragShader;
    vk::UniqueShaderModule insertVertShader;
    vk::UniqueShaderModule queryTexVertShader, queryTexGeomShader, queryTexFragShader;
    vk::UniqueShaderModule queryPageVertShader, queryPageGeomShader, queryPageFragShader;
};

typedef std::shared_ptr<RasterScanIndexUpdate> PRasterScanIndexUpdate;
