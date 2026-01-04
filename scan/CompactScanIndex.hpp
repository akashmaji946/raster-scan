#pragma once

#include <core/VkEngine.hpp>
#include <core/vkutils.h>
#include <core/GraphicsPipelineProperties.hpp>
#include <operators/SinglePassScan.hpp>
#include "BufferPool.hpp"
#include <memory>

// INITIAL_SCALE_FACTOR as requested
#define COMPACT_INITIAL_SCALE_FACTOR 2
#define COMPACT_GROW_SCALE_FACTOR 1.2

// CompactEntry structure (GPU layout)
// Size: 32 bytes (8 uints) to maintain 16-byte alignment
struct CompactEntry {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t rowId;
    uint32_t pageId; // MSB is valid bit (0x80000000)
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

class CompactScanIndex {
public:
    CompactScanIndex(vkcore::PVkDevice vd, int32_t ncols, vkcore::SinglePassScan* scan = nullptr);
    ~CompactScanIndex();

    // Initialize buffers and pipelines
    void initialize();
    
    // Build index from points
    void buildIndex(vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal);

    // Run queries
    // queryBuffer contains sets of 6 uints: [x1, x2, y1, y2, z1, z2]
    // outputBuffer stores results (bitmask or count)
    void runRangeQueries(vkcore::PBuffer queryBuffer, uint32_t nqueries, vkcore::PBuffer resultBuffer);

    // Delete points (mark invalid)
    // dataBuffer contains points to delete: [x, y, z]
    void deletePoints(vkcore::PBuffer dataBuffer, uint32_t ndeletes);

    // Insert new points
    // pointsBuffer contains new points [x, y, z] (raw)
    void insertPoints(vkcore::PBuffer pointsBuffer, uint32_t npoints);

    // Get statistics
    uint32_t getMaxBinCount();

public:
    vkcore::PVkDevice vd;
    vkcore::SinglePassScan* scan; // For GPU prefix sum
    int32_t ncols;
    uint32_t npoints;
    uint32_t minVal[3];
    uint32_t maxVal[3];
    uint32_t binRange; // Average entries per bin (Allocation hint)
    uint32_t binWidth[3]; // Coordinate width of each bin
    size_t countBufSize; // Size of count buffer (aligned for prefix sum)

    // Buffers
    // T: Start Address Buffer (1024*1024 uints)
    vkcore::PBuffer startAddrBuffer;
    
    // C: Count Buffer (1024*1024 uints)
    vkcore::PBuffer countBuffer;
    
    // Data Buffer: Stores CompactEntry
    vkcore::PBuffer dataBuffer;
    
    // Overflow Flag Buffer (1 uint)
    vkcore::PBuffer overflowBuffer;
    
    // Stats Buffer (2 uints: min, max)
    vkcore::PBuffer statsBuffer;

    // Capacity Buffer (1024*1024 uints)
    vkcore::PBuffer capacityBuffer;
    
    // Global Free Offset (for growing bins)
    uint64_t globalFreeOffset;
    uint64_t totalAllocatedCapacity;

    // Pipelines
    vkcore::GraphicsPipelineProperties buildPipelineProps;
    vkcore::GraphicsPipelineProperties queryPipelineProps;
    vkcore::GraphicsPipelineProperties deletePipelineProps;

    vk::UniquePipeline buildPipeline;
    vk::UniquePipeline countPipeline;
    vk::UniquePipeline statsPipeline;
    vk::UniquePipeline queryPipeline;
    vk::UniquePipeline deletePipeline;

    vk::UniqueShaderModule buildShader;
    vk::UniqueShaderModule countShader;
    vk::UniqueShaderModule statsShader;
    vk::UniqueShaderModule queryShader;
    vk::UniqueShaderModule deleteShader;
    
    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueDescriptorPool descPool;
    vk::UniqueDescriptorSet descSet;

private:
    void setupPipelines();
    void allocateBuffers(uint32_t npoints);
    void computeAndPrintStats(const std::string& phase); // Helper for GPU stats

public:
    float getSizeMB() {
        if(!dataBuffer) return 0.0f;
        return (float)(totalAllocatedCapacity * sizeof(CompactEntry)) / (1024.0f * 1024.0f);
    }

};

typedef std::shared_ptr<CompactScanIndex> PCompactScanIndex;
