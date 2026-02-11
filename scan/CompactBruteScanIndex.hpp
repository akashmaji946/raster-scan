#pragma once

#include <core/VkEngine.hpp>
#include <core/vkutils.h>
#include <core/GraphicsPipelineProperties.hpp>
#include <operators/SinglePassScan.hpp>
#include "BufferPool.hpp"
#include <memory>

// CompactBruteScan: Builds like CompactScan (binned by x,y), queries like BruteForceScan (all bins)
// Uses auxiliary buffer for inserts, marks deletes as invalid in original buffer
// For updates: delete from original, insert to auxiliary with same rowID

// Scale factor for main buffer (extra space per bin) - supports 1.2, 1.5, 2.0
#define COMPACTBRUTE_INITIAL_SCALE_FACTOR (1.2)

// Auxiliary buffer size as fraction of main buffer (10% = 0.1, 20% = 0.2)
// For Mode 81 testing: 0.00025 = 100 slots for 400M points
#define COMPACTBRUTE_AUX_FRACTION 0.00025

// CompactBruteEntry structure (GPU layout) - same as CompactEntry
// Size: 16 bytes (4 uints)
struct CompactBruteEntry {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t rowId;  // MSB is valid bit (0x80000000), lower 31 bits are rowId
};

class CompactBruteScanIndex {
public:
    CompactBruteScanIndex(vkcore::PVkDevice vd, int32_t ncols, vkcore::SinglePassScan* scan = nullptr);
    ~CompactBruteScanIndex();

    // Initialize buffers and pipelines
    void initialize();
    
    // Build index from points (like CompactScan - bin by x,y coords)
    void buildIndex(vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal);

    // Run brute-force queries (search ALL entries in both main and aux buffers)
    // queryBuffer contains sets of 6 uints: [x1, x2, y1, y2, z1, z2]
    // resultBuffer stores results (bitmask)
    void runRangeQueries(vkcore::PBuffer queryBuffer, uint32_t nqueries, vkcore::PBuffer resultBuffer);

    // Delete points (mark invalid in main buffer) - by data values (x,y,z)
    // dataBuffer contains points to delete: [x, y, z] triples
    void deletePoints(vkcore::PBuffer dataBuffer, uint32_t ndeletes);

    // Insert new points into auxiliary buffer
    // pointsBuffer contains new points [x, y, z] (row-major, 3 uints per point)
    // Uses new rowIDs starting from nextRowId
    void insertPoints(vkcore::PBuffer pointsBuffer, uint32_t npoints);
    
    // Update points: delete from main buffer, insert to aux buffer with SAME rowID
    // dataBuffer contains [x, y, z] triples of points to update
    // This finds matching entries, gets their rowIDs, marks them invalid, 
    // and inserts the same points to aux buffer with the same rowIDs
    void updatePoints(vkcore::PBuffer dataBuffer, uint32_t nupdates);

    // Push aux buffer data to main buffer bins and clear aux buffer
    // Remaps all valid aux entries to their appropriate bins in main buffer
    // Resets aux buffer count to 0
    void pushAuxToMain();

    // Get statistics
    uint32_t getMaxBinCount();
    
    // Get current counts
    uint32_t getMainValidCount();
    uint32_t getAuxValidCount();

public:
    vkcore::PVkDevice vd;
    vkcore::SinglePassScan* scan; // For GPU prefix sum
    int32_t ncols;
    uint32_t npoints;
    uint32_t minVal[3];
    uint32_t maxVal[3];
    uint32_t binRange; // Average entries per bin
    uint32_t binWidth[3]; // Coordinate width of each bin
    size_t countBufSize; // Size of count buffer (aligned for prefix sum)

    // Main buffer structures (like CompactScan)
    vkcore::PBuffer startAddrBuffer;   // Start address per bin
    vkcore::PBuffer countBuffer;       // Count per bin (for build)
    vkcore::PBuffer extentBuffer;      // Actual count per bin (for query iteration)
    vkcore::PBuffer capacityBuffer;    // Original capacity per bin
    vkcore::PBuffer mainDataBuffer;    // Main data buffer: stores CompactBruteEntry
    
    // Auxiliary buffer for inserts (simple append-only)
    vkcore::PBuffer auxDataBuffer;     // Auxiliary data buffer for inserts
    vkcore::PBuffer auxCountBuffer;    // Single uint: current count in aux buffer (GPU)
    uint32_t auxCapacity;              // Max entries in aux buffer
    uint32_t cachedAuxCount;           // CPU-side cache of aux count (avoid GPU readback)
    uint32_t nextRowId;                // Next available rowID for new inserts
    
    // Stats Buffer (2 uints: min, max)
    vkcore::PBuffer statsBuffer;

    // Global allocation tracking
    uint64_t mainAllocatedCapacity;
    uint64_t globalFreeOffset;

    // Graphics Pipelines (for build like CompactScan)
    vkcore::GraphicsPipelineProperties bcPipelineProps; // Build Count
    vkcore::GraphicsPipelineProperties bPipelineProps;  // Build Insert
    vk::UniquePipeline bcPipeline; // Build Count
    vk::UniquePipeline bPipeline;  // Build Insert
    vk::UniqueShaderModule bcVertexShader;
    vk::UniqueShaderModule bVertexShader;
    vk::UniqueShaderModule fragmentShader; // Dummy fragment shader
    
    // Compute Pipelines
    vk::UniquePipeline queryPipeline;      // Brute-force query (scan all entries)
    vk::UniquePipeline deletePipeline;     // Delete from main buffer
    vk::UniquePipeline insertPipeline;     // Insert to aux buffer
    vk::UniquePipeline updatePipeline;     // Update: delete + insert with same rowID
    vk::UniquePipeline pushAuxPipeline;    // Push aux buffer to main buffer bins
    vk::UniquePipeline scaleCountsPipeline; // Scale bin counts before prefix sum
    vk::UniqueShaderModule queryShader;
    vk::UniqueShaderModule deleteShader;
    vk::UniqueShaderModule insertShader;
    vk::UniqueShaderModule updateShader;
    vk::UniqueShaderModule pushAuxShader;
    vk::UniqueShaderModule scaleCountsShader;
    
    // Compute pipeline layout
    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueDescriptorPool descPool;
    vk::UniqueDescriptorSet descSet;
    
    // Dummy FBO for graphics pipeline
    vkcore::PFrameBuffer dummyFbo;
    
    // Cached fence for queries
    vk::UniqueFence queryFence;

private:
    void setupPipelines();
    void allocateBuffers(uint32_t npoints);

public:
    float getSizeMB() {
        float mainSize = mainDataBuffer ? (float)(mainAllocatedCapacity * sizeof(CompactBruteEntry)) / (1024.0f * 1024.0f) : 0.0f;
        float auxSize = auxDataBuffer ? (float)(auxCapacity * sizeof(CompactBruteEntry)) / (1024.0f * 1024.0f) : 0.0f;
        return mainSize + auxSize;
    }
};

typedef std::shared_ptr<CompactBruteScanIndex> PCompactBruteScanIndex;
