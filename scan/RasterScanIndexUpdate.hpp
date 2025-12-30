// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VkEngine.hpp>
#include <core/vkutils.h>
#include <core/GraphicsPipelineProperties.hpp>

#include <common/utils.h>

#include "BufferPool.hpp"

// Page size: number of data items per page
#define PAGE_DATA_SIZE 1

// Page structure layout in GPU buffer:
// struct Page {
//     uvec4 data[PAGE_DATA_SIZE];  // 3 columns (x, y, z) + rowId stored as uvec4
//     uint count;                   // Number of items currently in this page
//     uint nextPagePtr;             // Pointer to the next page (or NULL_PAGE_PTR)
// };
#define NULL_PAGE_PTR 0xFFFFFFFF

// Size of a page in uint32_t units
// PAGE_DATA_SIZE * 4 (uvec4) + 1 (count) + 1 (nextPagePtr)
// Note: std430 alignment requires struct size to be multiple of 16 bytes (uvec4 alignment)
// 6 uints = 24 bytes -> padded to 32 bytes = 8 uints
#define PAGE_SIZE_UINTS 8

// Maximum number of pages that can be allocated
// Note: With 1-item-per-page approach, this limits max data points
// Memory usage: MAX_PAGES * PAGE_SIZE_UINTS * 4 bytes
// 
// GPU Memory Budget (GTX 1650 = 4GB, ~3.4GB usable):
//   - Page buffer: MAX_PAGES * 6 * 4 bytes
//   - Head buffer: 1M * 4 = 4MB
//   - Points buffer: nPoints * 3 * 4 bytes
//   - Result/staging buffers: ~100MB
//
// For COMPARISON MODE (both RasterScan2D and RasterScanIndexUpdate):
//   RasterScan2D uses ~1GB, so we have ~2.4GB left
//   With 32M pages: 768MB page buffer + 384MB points = ~1.2GB ✓

// #define MAX_PAGES (1 << 25)  // 32M pages = 768MB page buffer (safe for comparison mode)

#define MAX_PAGES (1 << 25) // 128M pages = 1.536GB page buffer 

// Bitmap size for tracking free pages
// 128M pages / 32 bits per uint = 4M uints = 16MB bitmap
#define BITMAP_SIZE_UINTS (MAX_PAGES / 32)
#define BITMAP_SIZE_BYTES (BITMAP_SIZE_UINTS * sizeof(uint32_t))

// Free page management class using bitmap
class FreeBitmap {
public:
    FreeBitmap(vkcore::PVkDevice vd, uint32_t maxPages);
    ~FreeBitmap();

    void initialize();
    void initializeWithAllocatedCount(uint32_t allocatedPages);  // Set first N pages as allocated, rest as free
    void destroy();

public:
    vkcore::PVkDevice vd;
    
    // GPU buffer storing the bitmap (1 = free, 0 = allocated)
    vkcore::PBuffer bitmapBuffer;
    
    // GPU buffer for tracking number of free pages
    vkcore::PBuffer freeCountBuffer;
    
    // GPU buffer for next allocation hint (for faster allocation)
    vkcore::PBuffer nextFreeHintBuffer;
    
    uint32_t maxPages;
    uint32_t bitmapSizeUints;
    bool valid;
};

typedef std::shared_ptr<FreeBitmap> PFreeBitmap;

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
    // rowIdOffset: offset to add to gl_VertexIndex to generate unique rowIds
    void insertPoints(PLinkedListIndex index, vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t rowIdOffset = 0);
    
    // Delete data points from index (mark as deleted)
    void deletePoints(PLinkedListIndex index, vkcore::PBuffer pointsBuffer, uint32_t npoints);
    
    // Delete data points by coordinate range (mark as deleted)
    // Range format: [x1, x2, y1, y2, z1, z2]
    void deleteRange(PLinkedListIndex index, uint32_t* range);
    
    // Run range queries on the linked list index
    void runRangeQueries(PLinkedListIndex index, vkcore::PBuffer qranges, uint32_t nqueries);
    
    // ==================== Bitmap-based Free Space Management ====================
    
    // Initialize bitmap with initial allocation count (call after buildIndex)
    void initializeBitmapWithAllocation(uint32_t allocatedPages);
    
    // Mark deleted pages as free in the bitmap
    void markDeletedPagesAsFree(PLinkedListIndex index);
    
    // Compact pages: reclaim deleted pages and rebuild linked lists
    // Call this when allocation limit is reached but there are free pages
    void compactPages(PLinkedListIndex index);
    
    // Insert points using bitmap-based allocation (reuses freed pages)
    void insertPointsWithBitmap(PLinkedListIndex index, vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t rowIdOffset = 0);
    
    // Insert points using bitmap-based allocation V2 (stores pageId in count field)
    void insertPointsWithBitmapV2(PLinkedListIndex index, vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t rowIdOffset = 0);
    
    // Delete points by data coordinates (x, y, z) - marks as invalid and updates bitmap
    void deletePointsByData(PLinkedListIndex index, vkcore::PBuffer dataBuffer, uint32_t ndeletes);
    
    // Verify index by traversing linked list and outputting valid points
    void verifyIndex(PLinkedListIndex index, vkcore::PBuffer resultBuffer, vkcore::PBuffer countBuffer);

    // Get statistics about page allocation
    struct AllocationStats {
        uint32_t totalPages;        // MAX_PAGES (128M)
        uint32_t allocatedPages;    // Pages that have been allocated (allocCounter)
        uint32_t validPages;        // Allocated pages with valid bit set (in use)
        uint32_t invalidPages;      // Allocated pages with valid bit cleared (deleted)
        uint32_t freePages;         // Pages marked free in bitmap (available for reuse)
        uint32_t unallocatedPages;  // Pages never allocated (totalPages - allocatedPages)
    };
    AllocationStats getAllocationStats(PLinkedListIndex index);
    
    // Count valid/invalid pages by scanning page buffer (expensive - for debugging)
    void countValidInvalidPages(uint32_t allocCounter, uint32_t& validCount, uint32_t& invalidCount);
    
    // Traverse all linked lists and count reachable pages (expensive - for debugging)
    // Returns: validCount = pages with valid bit set, invalidCount = pages with valid bit cleared
    void traverseLinkedListsAndCount(PLinkedListIndex index, uint32_t& validCount, uint32_t& invalidCount);

protected:
    void initShaders();
    void initBuffers();
    void setupInsertPipeline();
    void setupDeletePipeline();
    void setupDeleteRangePipeline();
    void setupQueryTexturePipeline();
    void setupQueryPagePipeline();
    void setupInsertBitmapPipeline();
    void setupInsertBitmapV2Pipeline();
    void setupDeleteByDataPipeline();
    void setupMarkFreePipeline();
    void setupCompactPipeline();
    void setupVerifyIndexPipeline();

protected:
    void runInsertPipeline(vkcore::PBuffer pointsBuffer, PLinkedListIndex index, uint32_t npoints, uint32_t rowIdOffset = 0);
    void runDeletePipeline(vkcore::PBuffer rowIdBuffer, PLinkedListIndex index, uint32_t ndeletes);
    void runDeleteRangePipeline(PLinkedListIndex index, uint32_t* range);
    void runQueryTexturePipeline(PLinkedListIndex index, vkcore::PBuffer qranges, uint32_t nqueries);
    void runQueryPagePipeline(PLinkedListIndex index, vkcore::PBuffer qranges, uint32_t nqueries);
    void runInsertBitmapPipeline(vkcore::PBuffer pointsBuffer, PLinkedListIndex index, uint32_t npoints, uint32_t rowIdOffset = 0);
    void runInsertBitmapV2Pipeline(vkcore::PBuffer pointsBuffer, PLinkedListIndex index, uint32_t npoints, uint32_t rowIdOffset = 0);
    void runDeleteByDataPipeline(vkcore::PBuffer dataBuffer, PLinkedListIndex index, uint32_t ndeletes);
    void runMarkFreePipeline(PLinkedListIndex index);
    void runCompactPipeline(PLinkedListIndex index);
    void runVerifyIndexPipeline(PLinkedListIndex index, vkcore::PBuffer resultBuffer, vkcore::PBuffer countBuffer);

protected:
    int32_t ncols;

public:
    vkcore::PVkDevice vd;
    PBufferCache bufs;
    PPageAllocator pageAlloc;
    vkcore::PBuffer maxBuffer;

    // Pipelines
    vkcore::GraphicsPipelineProperties insertPipelineProps;
    vkcore::GraphicsPipelineProperties deletePipelineProps;
    vkcore::GraphicsPipelineProperties deleteRangePipelineProps;
    vkcore::GraphicsPipelineProperties queryTexturePipelineProps;
    vkcore::GraphicsPipelineProperties queryPagePipelineProps;
    vkcore::GraphicsPipelineProperties insertBitmapPipelineProps;
    vkcore::GraphicsPipelineProperties insertBitmapV2PipelineProps;
    vkcore::GraphicsPipelineProperties deleteByDataPipelineProps;
    vkcore::GraphicsPipelineProperties markFreePipelineProps;
    vkcore::GraphicsPipelineProperties compactPipelineProps;
    
    // Compute pipeline for verification
    vk::UniqueDescriptorSetLayout verifyIndexDescSetLayout;
    vk::UniquePipelineLayout verifyIndexPipelineLayout;
    vk::UniquePipeline verifyIndexPipeline;
    vk::UniqueDescriptorPool verifyIndexDescPool;
    vk::UniqueDescriptorSet verifyIndexDescSet;

    vk::UniquePipeline insertPipeline;
    vk::UniquePipeline deletePipeline;
    vk::UniquePipeline deleteRangePipeline;
    vk::UniquePipeline queryTexturePipeline;
    vk::UniquePipeline queryPagePipeline;
    vk::UniquePipeline insertBitmapPipeline;
    vk::UniquePipeline insertBitmapV2Pipeline;
    vk::UniquePipeline deleteByDataPipeline;
    vk::UniquePipeline markFreePipeline;
    vk::UniquePipeline compactPipeline;
    
    // Shaders
    vk::UniqueShaderModule dummyFragShader;
    vk::UniqueShaderModule insertVertShader;
    vk::UniqueShaderModule deleteVertShader;
    vk::UniqueShaderModule deleteRangeVertShader;
    vk::UniqueShaderModule queryTexVertShader, queryTexGeomShader, queryTexFragShader;
    vk::UniqueShaderModule queryPageVertShader, queryPageGeomShader, queryPageFragShader;
    vk::UniqueShaderModule insertBitmapVertShader;
    vk::UniqueShaderModule insertBitmapV2VertShader;
    vk::UniqueShaderModule deleteByDataVertShader;
    vk::UniqueShaderModule markFreeVertShader;
    vk::UniqueShaderModule compactVertShader;
    vk::UniqueShaderModule verifyIndexCompShader;
    
    // Free bitmap for page reclamation
    PFreeBitmap freeBitmap;
};

typedef std::shared_ptr<RasterScanIndexUpdate> PRasterScanIndexUpdate;
