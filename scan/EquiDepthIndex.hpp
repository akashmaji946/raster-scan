#pragma once

#include <core/VkEngine.hpp>
#include <core/vkutils.h>
#include <core/GraphicsPipelineProperties.hpp>
#include <operators/SinglePassScan.hpp>
#include "BufferPool.hpp"
#include <memory>

// Coarse histogram resolution (>= 4 * INDEX_RESOLUTION for good quantile approximation)
#define EQUIDEPTH_COARSE_BINS 4096

// EquiDepthEntry structure (GPU layout) - same as CompactEntry
struct EquiDepthEntry {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t rowId;  // MSB is valid bit (0x80000000), lower 31 bits are rowId
};

class EquiDepthIndex {
public:
    EquiDepthIndex(vkcore::PVkDevice vd, int32_t ncols, vkcore::SinglePassScan* scan = nullptr);
    ~EquiDepthIndex();

    // Initialize buffers and pipelines
    void initialize();
    
    // Build index from points using equi-depth binning
    void buildIndex(vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal);

    // Run range queries
    void runRangeQueries(vkcore::PBuffer queryBuffer, uint32_t nqueries, vkcore::PBuffer resultBuffer);

    // Get statistics
    uint32_t getMaxBinCount();

public:
    vkcore::PVkDevice vd;
    vkcore::SinglePassScan* scan;
    int32_t ncols;
    uint32_t npoints;
    uint32_t minVal[3];
    uint32_t maxVal[3];
    size_t countBufSize;

    // Quantile boundary buffers (INDEX_RESOLUTION + 1 values each)
    vkcore::PBuffer quantileXBuffer;
    vkcore::PBuffer quantileYBuffer;

    // Coarse histogram buffers (EQUIDEPTH_COARSE_BINS values each)
    vkcore::PBuffer histXBuffer;
    vkcore::PBuffer histYBuffer;

    // Bin management buffers
    vkcore::PBuffer startAddrBuffer;  // Prefix sum of counts
    vkcore::PBuffer countBuffer;      // Current count per bin
    vkcore::PBuffer extentBuffer;     // Extent per bin (for queries)

    // Data buffer: stores EquiDepthEntry
    vkcore::PBuffer dataBuffer;
    uint64_t totalAllocatedCapacity;

    // Query result buffers
    vkcore::PBuffer maxBuffer;   // VkDrawIndirectCommand for pass 2
    vkcore::PBuffer edgeBuffer;  // [st, en) pairs from pass 1

    // Compute Pipelines
    vk::UniquePipeline histogramPipeline;
    vk::UniquePipeline quantilesPipeline;
    vk::UniqueShaderModule histogramShader;
    vk::UniqueShaderModule quantilesShader;

    // Graphics Pipelines
    vkcore::GraphicsPipelineProperties countPipelineProps;
    vkcore::GraphicsPipelineProperties buildPipelineProps;
    vkcore::GraphicsPipelineProperties queryPipelineProps;
    vkcore::GraphicsPipelineProperties edgePipelineProps;
    
    vk::UniquePipeline countPipeline;
    vk::UniquePipeline buildPipeline;
    vk::UniquePipeline queryPipeline;
    vk::UniquePipeline edgePipeline;
    
    vk::UniqueShaderModule countVertexShader;
    vk::UniqueShaderModule buildVertexShader;
    vk::UniqueShaderModule queryVertexShader;
    vk::UniqueShaderModule queryGeomShader;
    vk::UniqueShaderModule queryFragShader;
    vk::UniqueShaderModule edgeVertexShader;
    vk::UniqueShaderModule edgeGeomShader;
    vk::UniqueShaderModule edgeFragShader;
    vk::UniqueShaderModule dummyFragShader;

    // Compute pipeline layout
    vk::UniqueDescriptorSetLayout histDescSetLayout;
    vk::UniquePipelineLayout histPipelineLayout;
    vk::UniqueDescriptorPool histDescPool;
    vk::UniqueDescriptorSet histDescSet;

    vk::UniqueDescriptorSetLayout quantDescSetLayout;
    vk::UniquePipelineLayout quantPipelineLayout;
    vk::UniqueDescriptorPool quantDescPool;
    vk::UniqueDescriptorSet quantDescSet;

    // Dummy FBO for graphics pipeline
    vkcore::PFrameBuffer dummyFbo;

    // Cached state
    bool queryDescriptorsInitialized = false;
    vkcore::PBuffer lastResultBuffer = nullptr;
    vk::UniqueFence queryFence;

private:
    void setupPipelines();
    void allocateBuffers(uint32_t npoints);
    void buildHistograms(vkcore::PBuffer pointsBuffer, uint32_t npoints, bool bindDescriptors = true);
    void computeQuantiles();
    void updateHistogramDescriptors(vkcore::PBuffer pointsBuffer, uint32_t npoints);
    void updateQuantileDescriptors();

public:
    float getSizeMB() {
        if (!dataBuffer) return 0.0f;
        return (float)(totalAllocatedCapacity * sizeof(EquiDepthEntry)) / (1024.0f * 1024.0f);
    }
};

typedef std::shared_ptr<EquiDepthIndex> PEquiDepthIndex;
