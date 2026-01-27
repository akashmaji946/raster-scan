#pragma once

#include <core/VkEngine.hpp>
#include <core/vkutils.h>
#include <operators/SinglePassScan.hpp>
#include <memory>

// BruteForceEntry structure (GPU layout) - simple (x, y, z, rowId)
// Size: 16 bytes (4 uints)
struct BruteForceEntry {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t rowId;  // MSB is valid bit (0x80000000), lower 31 bits are rowId
};

class BruteForceIndex {
public:
    BruteForceIndex(vkcore::PVkDevice vd, int32_t ncols);
    ~BruteForceIndex();

    // Initialize pipelines
    void initialize();
    
    // Build index - just copy data to GPU buffer (no binning)
    void buildIndex(vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal);

    // Run brute-force range queries
    // queryBuffer contains sets of 6 uints: [x1, x2, y1, y2, z1, z2]
    // resultBuffer stores results (bitmask)
    void runRangeQueries(vkcore::PBuffer queryBuffer, uint32_t nqueries, vkcore::PBuffer resultBuffer);

public:
    vkcore::PVkDevice vd;
    int32_t ncols;
    uint32_t npoints;
    uint32_t minVal[3];
    uint32_t maxVal[3];

    // Data Buffer: Stores BruteForceEntry (x, y, z, rowId) for all N points
    vkcore::PBuffer dataBuffer;
    
    // Compute Pipeline for brute-force query
    vk::UniquePipeline queryPipeline;
    vk::UniquePipeline buildPipeline;
    vk::UniqueShaderModule queryShader;
    vk::UniqueShaderModule buildShader;
    
    // Descriptor set layout and pool
    vk::UniqueDescriptorSetLayout queryDescSetLayout;
    vk::UniquePipelineLayout queryPipelineLayout;
    vk::UniqueDescriptorPool queryDescPool;
    vk::UniqueDescriptorSet queryDescSet;
    
    vk::UniqueDescriptorSetLayout buildDescSetLayout;
    vk::UniquePipelineLayout buildPipelineLayout;
    vk::UniqueDescriptorPool buildDescPool;
    vk::UniqueDescriptorSet buildDescSet;

private:
    void setupPipelines();

public:
    float getSizeMB() {
        if(!dataBuffer) return 0.0f;
        return (float)(npoints * sizeof(BruteForceEntry)) / (1024.0f * 1024.0f);
    }
};

typedef std::shared_ptr<BruteForceIndex> PBruteForceIndex;
