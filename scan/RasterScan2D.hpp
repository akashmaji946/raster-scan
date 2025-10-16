// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VkEngine.hpp>
#include <core/vkutils.h>
#include <core/GraphicsPipelineProperties.hpp>

#include <operators/SinglePassScan.hpp>
#include <operators/ReduceMax.hpp>

#include <common/utils.h>

#include "BufferPool.hpp"

class RasterScan2D
{
public:
    RasterScan2D(vkcore::PVkDevice vd, PBufferCache bufs, vkcore::SinglePassScan *scan, vkcore::ReduceMax *reduce, int32_t ncols);
    ~RasterScan2D();

public:
    void initalize();

public:
    PRasterIndex buildIndex(vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal);
    void runRangeQueries(PRasterIndex index, vkcore::PBuffer qranges, uint32_t nqueries);

protected:
    void initShaders();
    void initBuffers();
    void setupBuildCountPipeline();
    void setupBuildPipeline();
    void setupRQTPipeline();
    void setupRQEPipeline();

protected:
    void buildHistogram(vkcore::PBuffer pointsBuffer, PRasterIndex index);
    void build(vkcore::PBuffer pointsBuffer, PRasterIndex index);
    void runRQTPipeline(PRasterIndex index, vkcore::PBuffer qranges, uint32_t nqueries);
    void runRQEPipeline(PRasterIndex index, vkcore::PBuffer qranges, uint32_t nqueries);

protected:
    int32_t ncols;

public:
    vkcore::PVkDevice vd;
    PBufferCache bufs;
    vkcore::SinglePassScan *scan;
    vkcore::ReduceMax *reduce;
    vkcore::PBuffer maxBuffer;

    // the different pipelines
    vkcore::GraphicsPipelineProperties bcPipelineProps, bPipelineProps;
    vkcore::GraphicsPipelineProperties rqtPipelineProps, rqePipelineProps;
    vk::UniquePipeline bcPipeline, bPipeline, rqtPipeline, rqePipeline;

    vk::UniqueShaderModule fragmentShader;
    vk::UniqueShaderModule bcVertexShader, bVertexShader;
    vk::UniqueShaderModule rqVertexShader, rqGeomShader, rqFragShader;
    vk::UniqueShaderModule eVertexShader, eGeomShader, eFragShader;
};

