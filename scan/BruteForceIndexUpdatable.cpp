#include "BruteForceIndexUpdatable.hpp"
#include <common/utils.h>
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace vkcore;

BruteForceIndexUpdatable::BruteForceIndexUpdatable(PVkDevice vd, int32_t ncols, double scaleFactor)
    : vd(vd), ncols(ncols), npoints(0), capacity(0), scaleFactor(scaleFactor), activeCount(0), nextFreeSlot(0) {
    memset(minVal, 0, sizeof(minVal));
    memset(maxVal, 0, sizeof(maxVal));
}

BruteForceIndexUpdatable::~BruteForceIndexUpdatable() {
    if (dataBuffer) dataBuffer->destroy();
}

void BruteForceIndexUpdatable::initialize() {
    setupPipelines();
}

void BruteForceIndexUpdatable::setupPipelines() {
    // Build shader - converts column-major input to row-major BruteForceEntry
    {
        std::vector<uint32_t> code;
        if (!readShader(SHADER_FOLDER + "/bruteforce_build.comp.spv", code)) {
            throw std::runtime_error("Failed to load bruteforce_build.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        buildShader = vd->device->createShaderModuleUnique(createInfo);
        
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, {});
        buildDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
        
        // Push constants: npoints (4) + ncols (4) + pointsBufferAddr (8) + dataBufferAddr (8) = 24 bytes
        vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0, 24);
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, *buildDescSetLayout, pushRange);
        buildPipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
        
        vk::ComputePipelineCreateInfo pipelineInfo({}, {{}, vk::ShaderStageFlagBits::eCompute, *buildShader, "main"}, *buildPipelineLayout);
        buildPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Query shader - brute force scan all N points
    {
        std::vector<uint32_t> code;
        if (!readShader(SHADER_FOLDER + "/bruteforce_query.comp.spv", code)) {
            throw std::runtime_error("Failed to load bruteforce_query.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        queryShader = vd->device->createShaderModuleUnique(createInfo);
        
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},  // query
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},  // result
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
        queryDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
        
        vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0, 16);
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, *queryDescSetLayout, pushRange);
        queryPipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
        
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eStorageBuffer, 2}
        };
        vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes);
        queryDescPool = vd->device->createDescriptorPoolUnique(poolInfo);
        
        vk::DescriptorSetAllocateInfo allocInfo(*queryDescPool, *queryDescSetLayout);
        queryDescSet = std::move(vd->device->allocateDescriptorSetsUnique(allocInfo)[0]);
        
        vk::ComputePipelineCreateInfo pipelineInfo({}, {{}, vk::ShaderStageFlagBits::eCompute, *queryShader, "main"}, *queryPipelineLayout);
        queryPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Delete shader - clears valid bit for entries at specified row indices
    {
        std::vector<uint32_t> code;
        if (!readShader(SHADER_FOLDER + "/bruteforce_delete.comp.spv", code)) {
            throw std::runtime_error("Failed to load bruteforce_delete.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        deleteShader = vd->device->createShaderModuleUnique(createInfo);
        
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},  // rowIds
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
        deleteDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
        
        // Push constants: count (4) + padding (4) + dataBufferAddr (8) = 16 bytes
        vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0, 16);
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, *deleteDescSetLayout, pushRange);
        deletePipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
        
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eStorageBuffer, 1}
        };
        vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes);
        deleteDescPool = vd->device->createDescriptorPoolUnique(poolInfo);
        
        vk::DescriptorSetAllocateInfo allocInfo(*deleteDescPool, *deleteDescSetLayout);
        deleteDescSet = std::move(vd->device->allocateDescriptorSetsUnique(allocInfo)[0]);
        
        vk::ComputePipelineCreateInfo pipelineInfo({}, {{}, vk::ShaderStageFlagBits::eCompute, *deleteShader, "main"}, *deletePipelineLayout);
        deletePipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Insert shader - inserts new entries at specified slot positions
    {
        std::vector<uint32_t> code;
        if (!readShader(SHADER_FOLDER + "/bruteforce_insert.comp.spv", code)) {
            throw std::runtime_error("Failed to load bruteforce_insert.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        insertShader = vd->device->createShaderModuleUnique(createInfo);
        
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},  // slots
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},  // points
            {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},  // rowIds
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
        insertDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
        
        // Push constants: count (4) + padding (4) + dataBufferAddr (8) = 16 bytes
        vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0, 16);
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, *insertDescSetLayout, pushRange);
        insertPipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
        
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eStorageBuffer, 3}
        };
        vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes);
        insertDescPool = vd->device->createDescriptorPoolUnique(poolInfo);
        
        vk::DescriptorSetAllocateInfo allocInfo(*insertDescPool, *insertDescSetLayout);
        insertDescSet = std::move(vd->device->allocateDescriptorSetsUnique(allocInfo)[0]);
        
        vk::ComputePipelineCreateInfo pipelineInfo({}, {{}, vk::ShaderStageFlagBits::eCompute, *insertShader, "main"}, *insertPipelineLayout);
        insertPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
}

void BruteForceIndexUpdatable::buildIndex(PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal) {
    this->npoints = npoints;
    this->activeCount = npoints;
    memcpy(this->minVal, minVal, 3 * sizeof(uint32_t));
    memcpy(this->maxVal, maxVal, 3 * sizeof(uint32_t));
    
    // Allocate with scale factor for future inserts
    this->capacity = static_cast<uint32_t>(npoints * scaleFactor);
    this->nextFreeSlot = npoints;  // Next available slot after initial data
    this->freeSlots.clear();
    
    std::cerr << "[BruteForceUpdatable] Building index with " << npoints << " points...\n";
    std::cerr << "[BruteForceUpdatable] Capacity: " << capacity << " (scale factor: " << scaleFactor << ")\n";
    
    size_t dataSize = (size_t)capacity * sizeof(BruteForceEntryUpdatable);
    dataBuffer = std::make_shared<Buffer>(vd);
    dataBuffer->create(dataSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        MemoryType::Internal);
    
    std::cerr << "[BruteForceUpdatable] Data buffer size: " << (dataSize / (1024.0 * 1024.0)) << " MB\n";
    
    // Run build compute shader
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *buildPipeline);
    
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
    
    std::cerr << "[BruteForceUpdatable] Build complete.\n";
}

void BruteForceIndexUpdatable::runRangeQueries(PBuffer queryBuffer, uint32_t nqueries, PBuffer resultBuffer) {
    if (!queryFence) {
        queryFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    }
    
    if (queryBuffer->buf != lastQueryBuffer || resultBuffer->buf != lastResultBuffer) {
        vk::DescriptorBufferInfo queryInfo(queryBuffer->buf, 0, VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo resultInfo(resultBuffer->buf, 0, VK_WHOLE_SIZE);
        
        std::vector<vk::WriteDescriptorSet> writes = {
            {*queryDescSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &queryInfo},
            {*queryDescSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &resultInfo},
        };
        vd->device->updateDescriptorSets(writes, {});
        lastQueryBuffer = queryBuffer->buf;
        lastResultBuffer = resultBuffer->buf;
    }
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    // Use capacity for result buffer size to handle all possible entries
    uint32_t resultSizeUints = (capacity + 31) / 32;
    vd->commandBuffer->fillBuffer(resultBuffer->buf, 0, resultSizeUints * sizeof(uint32_t), 0);
    
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
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *queryPipeline);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *queryPipelineLayout, 0, *queryDescSet, {});
    
    // Query all entries up to nextFreeSlot (includes both valid and deleted)
    // The shader checks the valid bit
    uint64_t dataAddr = dataBuffer->getDeviceAddress();
    struct {
        uint32_t npoints;
        uint32_t padding;
        uint32_t dataAddrLo;
        uint32_t dataAddrHi;
    } pushData = {
        nextFreeSlot,  // Scan all allocated slots
        0,
        static_cast<uint32_t>(dataAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(dataAddr >> 32)
    };
    vd->commandBuffer->pushConstants(*queryPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pushData), &pushData);
    
    uint32_t numWorkgroups = (nextFreeSlot + 255) / 256;
    vd->commandBuffer->dispatch(numWorkgroups, 1, 1);
    
    vd->commandBuffer->end();
    
    vd->device->resetFences(*queryFence);
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vd->submit(submitInfo, *queryFence, false);
    vd->waitForFences(*queryFence, VK_TRUE, UINT64_MAX);
}

void BruteForceIndexUpdatable::deleteByRowId(PBuffer rowIdBuffer, uint32_t count) {
    // GPU-accelerated delete: directly clear valid bit at specified row indices
    if (!updateFence) {
        updateFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    }
    
    // Update descriptor set
    vk::DescriptorBufferInfo rowIdInfo(rowIdBuffer->buf, 0, VK_WHOLE_SIZE);
    std::vector<vk::WriteDescriptorSet> writes = {
        {*deleteDescSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &rowIdInfo},
    };
    vd->device->updateDescriptorSets(writes, nullptr);
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *deletePipeline);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *deletePipelineLayout, 0, *deleteDescSet, nullptr);
    
    uint64_t dataAddr = dataBuffer->getDeviceAddress();
    struct {
        uint32_t count;
        uint32_t padding;
        uint32_t dataAddrLo;
        uint32_t dataAddrHi;
    } pushData = {
        count,
        0,
        static_cast<uint32_t>(dataAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(dataAddr >> 32)
    };
    vd->commandBuffer->pushConstants(*deletePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pushData), &pushData);
    
    uint32_t numWorkgroups = (count + 255) / 256;
    vd->commandBuffer->dispatch(numWorkgroups, 1, 1);
    
    vd->commandBuffer->end();
    
    vd->device->resetFences(*updateFence);
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vd->submit(submitInfo, *updateFence, false);
    vd->waitForFences(*updateFence, VK_TRUE, UINT64_MAX);
    
    // Update CPU-side tracking
    activeCount -= count;
    // Note: freeSlots tracking would need the actual rowIds read back if we want to reuse slots
}

void BruteForceIndexUpdatable::insertPoints(PBuffer pointsBuffer, PBuffer slotBuffer, PBuffer rowIdBuffer, uint32_t count) {
    // GPU-accelerated insert: write entries at specified slot positions with original rowIds
    if (!updateFence) {
        updateFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    }
    
    // Update descriptor set
    vk::DescriptorBufferInfo slotInfo(slotBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo pointsInfo(pointsBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo rowIdInfo(rowIdBuffer->buf, 0, VK_WHOLE_SIZE);
    std::vector<vk::WriteDescriptorSet> writes = {
        {*insertDescSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &slotInfo},
        {*insertDescSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &pointsInfo},
        {*insertDescSet, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &rowIdInfo},
    };
    vd->device->updateDescriptorSets(writes, nullptr);
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *insertPipeline);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *insertPipelineLayout, 0, *insertDescSet, nullptr);
    
    uint64_t dataAddr = dataBuffer->getDeviceAddress();
    struct {
        uint32_t count;
        uint32_t padding;
        uint32_t dataAddrLo;
        uint32_t dataAddrHi;
    } pushData = {
        count,
        0,
        static_cast<uint32_t>(dataAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(dataAddr >> 32)
    };
    vd->commandBuffer->pushConstants(*insertPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pushData), &pushData);
    
    uint32_t numWorkgroups = (count + 255) / 256;
    vd->commandBuffer->dispatch(numWorkgroups, 1, 1);
    
    vd->commandBuffer->end();
    
    vd->device->resetFences(*updateFence);
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vd->submit(submitInfo, *updateFence, false);
    vd->waitForFences(*updateFence, VK_TRUE, UINT64_MAX);
    
    // Update CPU-side tracking
    activeCount += count;
    nextFreeSlot += count;  // Advance next free slot for append mode
}

void BruteForceIndexUpdatable::deletePoints(PBuffer pointsBuffer, uint32_t count) {
    // Read points to delete from GPU (row-major format: x, y, z per point)
    std::vector<uint32_t> deleteData(count * 3);
    
    // Create staging buffer for reading
    PBuffer staging(new Buffer(vd));
    staging->create(std::max(count * 3 * sizeof(uint32_t), nextFreeSlot * sizeof(BruteForceEntryUpdatable)),
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::ReadWrite);
    
    // Read delete points using helper
    readUsingStagingBuf((char*)deleteData.data(), count * 3 * sizeof(uint32_t), pointsBuffer, staging, vd);
    
    // Read current data buffer
    std::vector<BruteForceEntryUpdatable> entries(nextFreeSlot);
    readUsingStagingBuf((char*)entries.data(), nextFreeSlot * sizeof(BruteForceEntryUpdatable), dataBuffer, staging, vd);
    
    // Mark matching entries as invalid (clear valid bit)
    uint32_t deletedCount = 0;
    for (uint32_t d = 0; d < count; d++) {
        uint32_t dx = deleteData[d * 3 + 0];
        uint32_t dy = deleteData[d * 3 + 1];
        uint32_t dz = deleteData[d * 3 + 2];
        
        for (uint32_t i = 0; i < nextFreeSlot; i++) {
            if ((entries[i].rowId & 0x80000000) &&  // Valid
                entries[i].x == dx && entries[i].y == dy && entries[i].z == dz) {
                entries[i].rowId &= 0x7FFFFFFF;  // Clear valid bit
                freeSlots.push_back(i);
                deletedCount++;
                activeCount--;
                break;
            }
        }
    }
    
    // Write back modified entries
    loadUsingStagingBuf((char*)entries.data(), nextFreeSlot * sizeof(BruteForceEntryUpdatable), dataBuffer, staging, vd, 0);
    
    staging->destroy();
}

uint32_t BruteForceIndexUpdatable::cpuRangeQuery(const std::vector<BruteForceEntryUpdatable>& entries,
                                                  uint32_t x1, uint32_t x2, uint32_t y1, uint32_t y2, uint32_t z1, uint32_t z2) {
    uint32_t count = 0;
    for (const auto& e : entries) {
        if ((e.rowId & 0x80000000) &&  // Valid
            e.x >= x1 && e.x <= x2 &&
            e.y >= y1 && e.y <= y2 &&
            e.z >= z1 && e.z <= z2) {
            count++;
        }
    }
    return count;
}
