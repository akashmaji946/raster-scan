#pragma once

#include <core/VkEngine.hpp>
#include <core/vkutils.h>
#include <operators/SinglePassScan.hpp>
#include <memory>
#include <vector>

// Default scale factor for dataBuffer allocation (1.2 = 20% extra space)
#ifndef BRUTE_SCALE_FACTOR
#define BRUTE_SCALE_FACTOR 2
#endif

// BruteForceEntry structure (GPU layout) - simple (x, y, z, rowId)
// Size: 16 bytes (4 uints)
struct BruteForceEntryUpdatable {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t rowId;  // MSB is valid bit (0x80000000), lower 31 bits are rowId
};

// Extended BruteForceIndex with update support
// Allocates extra space in dataBuffer for inserts/updates/deletes
class BruteForceIndexUpdatable {
public:
    BruteForceIndexUpdatable(vkcore::PVkDevice vd, int32_t ncols, double scaleFactor = BRUTE_SCALE_FACTOR);
    ~BruteForceIndexUpdatable();

    // Initialize pipelines
    void initialize();
    
    // Build index - allocates capacity = npoints * scaleFactor
    void buildIndex(vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal);

    // Run brute-force range queries
    void runRangeQueries(vkcore::PBuffer queryBuffer, uint32_t nqueries, vkcore::PBuffer resultBuffer);
    
    // Update operations (GPU-accelerated)
    // deleteByRowId: takes buffer of row indices to delete (clears valid bit)
    void deleteByRowId(vkcore::PBuffer rowIdBuffer, uint32_t count);
    // insertPoints: takes buffer of (x,y,z) points, buffer of slot indices, and buffer of original rowIds
    void insertPoints(vkcore::PBuffer pointsBuffer, vkcore::PBuffer slotBuffer, vkcore::PBuffer rowIdBuffer, uint32_t count);
    
    // Legacy CPU-based operations (slow, for reference only)
    void deletePoints(vkcore::PBuffer pointsBuffer, uint32_t count);  // Mark points as invalid
    
    // CPU verification helper
    uint32_t cpuRangeQuery(const std::vector<BruteForceEntryUpdatable>& entries,
                           uint32_t x1, uint32_t x2, uint32_t y1, uint32_t y2, uint32_t z1, uint32_t z2);

public:
    vkcore::PVkDevice vd;
    int32_t ncols;
    uint32_t npoints;      // Current logical count
    uint32_t capacity;     // Allocated capacity
    uint32_t activeCount;  // Number of valid (non-deleted) entries
    double scaleFactor;
    uint32_t minVal[3];
    uint32_t maxVal[3];

    // Data Buffer: Stores BruteForceEntryUpdatable (x, y, z, rowId) for all entries
    vkcore::PBuffer dataBuffer;
    
    // Compute Pipelines
    vk::UniquePipeline queryPipeline;
    vk::UniquePipeline buildPipeline;
    vk::UniqueShaderModule queryShader;
    vk::UniqueShaderModule buildShader;
    
    // Descriptor sets for query
    vk::UniqueDescriptorSetLayout queryDescSetLayout;
    vk::UniquePipelineLayout queryPipelineLayout;
    vk::UniqueDescriptorPool queryDescPool;
    vk::UniqueDescriptorSet queryDescSet;
    
    // Descriptor sets for build
    vk::UniqueDescriptorSetLayout buildDescSetLayout;
    vk::UniquePipelineLayout buildPipelineLayout;
    
    // Delete pipeline (GPU compute)
    vk::UniquePipeline deletePipeline;
    vk::UniqueShaderModule deleteShader;
    vk::UniqueDescriptorSetLayout deleteDescSetLayout;
    vk::UniquePipelineLayout deletePipelineLayout;
    vk::UniqueDescriptorPool deleteDescPool;
    vk::UniqueDescriptorSet deleteDescSet;
    
    // Insert pipeline (GPU compute)
    vk::UniquePipeline insertPipeline;
    vk::UniqueShaderModule insertShader;
    vk::UniqueDescriptorSetLayout insertDescSetLayout;
    vk::UniquePipelineLayout insertPipelineLayout;
    vk::UniqueDescriptorPool insertDescPool;
    vk::UniqueDescriptorSet insertDescSet;
    
    // Cached fence for query execution
    vk::UniqueFence queryFence;
    vk::UniqueFence updateFence;
    // Track last bound buffers to avoid redundant descriptor updates
    vk::Buffer lastQueryBuffer = nullptr;
    vk::Buffer lastResultBuffer = nullptr;
    
    // Track free slots for inserts
    std::vector<uint32_t> freeSlots;
    uint32_t nextFreeSlot;  // Next slot after all used slots

private:
    void setupPipelines();

public:
    float getSizeMB() {
        if(!dataBuffer) return 0.0f;
        return (float)(capacity * sizeof(BruteForceEntryUpdatable)) / (1024.0f * 1024.0f);
    }
};

typedef std::shared_ptr<BruteForceIndexUpdatable> PBruteForceIndexUpdatable;
