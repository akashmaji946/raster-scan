#pragma once

#include <core/VkEngine.hpp>
#include <core/vkutils.h>
#include <core/GraphicsPipelineProperties.hpp>
#include <operators/SinglePassScan.hpp>
#include "BufferPool.hpp"
#include <memory>

// INITIAL_SCALE_FACTOR - allocate 2x space per bin for inserts
#define COMPACT_INITIAL_SCALE_FACTOR 8
#define COMPACT_GROW_SCALE_FACTOR 2

// CompactEntry structure (GPU layout) - same as RasterScan2D's uvec4
// Size: 16 bytes (4 uints)
struct CompactEntry {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t rowId;  // MSB is valid bit (0x80000000), lower 31 bits are rowId
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

    // Capacity Buffer (1024*1024 uints) - max capacity per bin
    vkcore::PBuffer capacityBuffer;
    
    // Extent Buffer (1024*1024 uints) - highest index written per bin
    // On build: extent[bin] = original_count
    // On insert: extent[bin] = max(extent[bin], new_offset + 1)
    // On delete: unchanged (entries not shifted)
    // Query iterates extent[bin] entries and skips invalid ones
    vkcore::PBuffer extentBuffer;
    
    // Global Free Offset (for growing bins)
    uint64_t globalFreeOffset;
    uint64_t totalAllocatedCapacity;

    // Graphics Pipelines (for fast build like RasterScan2D)
    vkcore::GraphicsPipelineProperties bcPipelineProps; // Build Count
    vkcore::GraphicsPipelineProperties bPipelineProps;  // Build Insert
    vkcore::GraphicsPipelineProperties queryGfxPipelineProps; // Query Pass 1 (range)
    vkcore::GraphicsPipelineProperties edgePipelineProps;     // Query Pass 2 (edge)
    vk::UniquePipeline bcPipeline; // Build Count
    vk::UniquePipeline bPipeline;  // Build Insert
    vk::UniquePipeline queryGfxPipeline; // Query Pass 1 (range)
    vk::UniquePipeline edgePipeline;     // Query Pass 2 (edge)
    vk::UniqueShaderModule bcVertexShader;
    vk::UniqueShaderModule bVertexShader;
    vk::UniqueShaderModule fragmentShader; // Dummy fragment shader
    vk::UniqueShaderModule queryGfxVertexShader;
    vk::UniqueShaderModule queryGfxGeomShader;
    vk::UniqueShaderModule queryGfxFragShader;
    vk::UniqueShaderModule edgeVertexShader;
    vk::UniqueShaderModule edgeGeomShader;
    vk::UniqueShaderModule edgeFragShader;
    
    // Buffers for two-pass query (like RasterScan2D)
    vkcore::PBuffer maxBuffer;   // [0]=numRanges, [1]=maxCount for indirect draw
    vkcore::PBuffer edgeBuffer;  // Stores [st, en) pairs from pass 1
    
    // Compute Pipelines (for delete, stats, insert)
    vk::UniquePipeline scalePipeline;
    vk::UniquePipeline statsPipeline;
    vk::UniquePipeline queryPipeline; // Compute query (fallback)
    vk::UniquePipeline deletePipeline;
    vk::UniquePipeline insertPipeline;
    vk::UniqueShaderModule scaleShader;
    vk::UniqueShaderModule statsShader;
    vk::UniqueShaderModule queryShader;
    vk::UniqueShaderModule deleteShader;
    vk::UniqueShaderModule insertShader;
    
    // Compute pipeline layout
    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueDescriptorPool descPool;
    vk::UniqueDescriptorSet descSet;
    
    // Dummy FBO for graphics pipeline
    vkcore::PFrameBuffer dummyFbo;

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
