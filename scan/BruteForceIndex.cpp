#include "BruteForceIndex.hpp"
#include <common/utils.h>
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace vkcore;

BruteForceIndex::BruteForceIndex(PVkDevice vd, int32_t ncols)
    : vd(vd), ncols(ncols), npoints(0) {
    memset(minVal, 0, sizeof(minVal));
    memset(maxVal, 0, sizeof(maxVal));
}

BruteForceIndex::~BruteForceIndex() {
    if (dataBuffer) dataBuffer->destroy();
}

void BruteForceIndex::initialize() {
    setupPipelines();
}

void BruteForceIndex::setupPipelines() {
    // Build shader - converts column-major input to row-major BruteForceEntry
    // Uses buffer device addresses to handle >4GB buffers
    {
        std::vector<uint32_t> code;
        if (!readShader(SHADER_FOLDER + "/bruteforce_build.comp.spv", code)) {
            throw std::runtime_error("Failed to load bruteforce_build.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        buildShader = vd->device->createShaderModuleUnique(createInfo);
        
        // No descriptor bindings needed - using buffer device addresses for all buffers
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, {});
        buildDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
        
        // Push constants: npoints (4) + ncols (4) + pointsBufferAddr (8) + dataBufferAddr (8) = 24 bytes
        vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0, 24);
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, *buildDescSetLayout, pushRange);
        buildPipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
        
        // Create compute pipeline (no descriptors needed)
        vk::ComputePipelineCreateInfo pipelineInfo({}, {{}, vk::ShaderStageFlagBits::eCompute, *buildShader, "main"}, *buildPipelineLayout);
        buildPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Query shader - brute force scan all N points
    // Uses buffer device address for data buffer to handle >4GB
    {
        std::vector<uint32_t> code;
        if (!readShader(SHADER_FOLDER + "/bruteforce_query.comp.spv", code)) {
            throw std::runtime_error("Failed to load bruteforce_query.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        queryShader = vd->device->createShaderModuleUnique(createInfo);
        
        // Descriptor set layout: queryBuffer, resultBuffer (data uses buffer device address)
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},  // query
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},  // result
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
        queryDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
        
        // Push constants: npoints (4 bytes) + padding (4 bytes) + dataBufferAddr (8 bytes) = 16 bytes
        // GLSL aligns uvec2 to 8 bytes, so total is 16 bytes
        vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0, 16);
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, *queryDescSetLayout, pushRange);
        queryPipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
        
        // Descriptor pool
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eStorageBuffer, 2}
        };
        vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes);
        queryDescPool = vd->device->createDescriptorPoolUnique(poolInfo);
        
        // Allocate descriptor set
        vk::DescriptorSetAllocateInfo allocInfo(*queryDescPool, *queryDescSetLayout);
        queryDescSet = std::move(vd->device->allocateDescriptorSetsUnique(allocInfo)[0]);
        
        // Create compute pipeline
        vk::ComputePipelineCreateInfo pipelineInfo({}, {{}, vk::ShaderStageFlagBits::eCompute, *queryShader, "main"}, *queryPipelineLayout);
        queryPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
}

void BruteForceIndex::buildIndex(PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal) {
    this->npoints = npoints;
    memcpy(this->minVal, minVal, 3 * sizeof(uint32_t));
    memcpy(this->maxVal, maxVal, 3 * sizeof(uint32_t));
    
    std::cerr << "[BruteForceIndex] Building index with " << npoints << " points...\n";
    
    // Allocate data buffer: N entries of (x, y, z, rowId)
    // Use eShaderDeviceAddress for buffer device address support (handles >4GB)
    size_t dataSize = (size_t)npoints * sizeof(BruteForceEntry);
    dataBuffer = std::make_shared<Buffer>(vd);
    dataBuffer->create(dataSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        MemoryType::Internal);
    
    std::cerr << "[BruteForceIndex] Data buffer size: " << (dataSize / (1024.0 * 1024.0)) << " MB\n";
    
    // Run build compute shader using buffer device addresses
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *buildPipeline);
    
    // Push constants: npoints (4) + ncols (4) + pointsBufferAddr (8) + dataBufferAddr (8) = 24 bytes
    uint64_t pointsAddr = pointsBuffer->getDeviceAddress();
    uint64_t dataAddr = dataBuffer->getDeviceAddress();
    struct {
        uint32_t npoints;
        uint32_t ncols;
        uint32_t pointsAddrLo;
        uint32_t pointsAddrHi;
        uint32_t dataAddrLo;
        uint32_t dataAddrHi;
    } pushData = {
        npoints,
        (uint32_t)ncols,
        static_cast<uint32_t>(pointsAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(pointsAddr >> 32),
        static_cast<uint32_t>(dataAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(dataAddr >> 32)
    };
    vd->commandBuffer->pushConstants(*buildPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pushData), &pushData);
    
    // Dispatch: 256 threads per workgroup
    uint32_t numWorkgroups = (npoints + 255) / 256;
    vd->commandBuffer->dispatch(numWorkgroups, 1, 1);
    
    vd->commandBuffer->end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
    
    std::cerr << "[BruteForceIndex] Build complete.\n";
}

void BruteForceIndex::runRangeQueries(PBuffer queryBuffer, uint32_t nqueries, PBuffer resultBuffer) {
    // Update query descriptor set (query and result buffers only, data uses buffer device address)
    vk::DescriptorBufferInfo queryInfo(queryBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo resultInfo(resultBuffer->buf, 0, VK_WHOLE_SIZE);
    
    std::vector<vk::WriteDescriptorSet> writes = {
        {*queryDescSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &queryInfo},
        {*queryDescSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &resultInfo},
    };
    vd->device->updateDescriptorSets(writes, {});
    
    // Clear result buffer and run query
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    uint32_t resultSizeUints = (npoints + 31) / 32;
    vd->commandBuffer->fillBuffer(resultBuffer->buf, 0, resultSizeUints * sizeof(uint32_t), 0);
    
    // Barrier after fill
    vk::BufferMemoryBarrier barrier(
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        resultBuffer->buf, 0, VK_WHOLE_SIZE
    );
    vd->commandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, barrier, {}
    );
    
    // Run query compute shader
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *queryPipeline);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *queryPipelineLayout, 0, *queryDescSet, {});
    
    // Push constants: npoints (4 bytes) + padding (4 bytes) + dataBufferAddr (8 bytes) = 16 bytes
    // GLSL aligns uvec2 to 8 bytes
    uint64_t dataAddr = dataBuffer->getDeviceAddress();
    struct {
        uint32_t npoints;
        uint32_t padding;  // Align uvec2 to 8 bytes
        uint32_t dataAddrLo;
        uint32_t dataAddrHi;
    } pushData = {
        npoints,
        0,  // padding
        static_cast<uint32_t>(dataAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(dataAddr >> 32)
    };
    vd->commandBuffer->pushConstants(*queryPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pushData), &pushData);
    
    // Dispatch: 256 threads per workgroup
    uint32_t numWorkgroups = (npoints + 255) / 256;
    vd->commandBuffer->dispatch(numWorkgroups, 1, 1);
    
    vd->commandBuffer->end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
}
