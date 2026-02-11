#include "CompactBruteScanIndex.hpp"
#include <common/utils.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>

using namespace vkcore;

// Helper function for graphics pipeline rendering
inline vk::RenderingInfo setupRenderingCB(PVkDevice vd, PFrameBuffer fbo, vk::RenderingAttachmentInfo &colorInfo) {
    colorInfo.imageView = fbo->colorView;
    colorInfo.imageLayout = vk::ImageLayout::eGeneral;
    colorInfo.loadOp = vk::AttachmentLoadOp::eDontCare;
    colorInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
    
    int width = INDEX_RESOLUTION;
    int height = INDEX_RESOLUTION;
    vk::Rect2D renderArea(vk::Offset2D(0, 0), vk::Extent2D(width, height));
    vk::RenderingInfo renderingInfo;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorInfo;
    renderingInfo.renderArea = renderArea;
    renderingInfo.layerCount = 1;
    
    vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
    vk::Rect2D scissor(vk::Offset2D(0, 0), vk::Extent2D(width, height));
    vd->commandBuffer->setViewport(0, 1, &viewport);
    vd->commandBuffer->setScissor(0, 1, &scissor);
    
    return renderingInfo;
}

CompactBruteScanIndex::CompactBruteScanIndex(PVkDevice vd, int32_t ncols, SinglePassScan* scan) 
    : vd(vd), ncols(ncols), scan(scan), npoints(0), binRange(0), auxCapacity(0), cachedAuxCount(0), nextRowId(0),
      mainAllocatedCapacity(0), globalFreeOffset(0) {
    memset(minVal, 0, sizeof(minVal));
    memset(maxVal, 0, sizeof(maxVal));
    memset(binWidth, 0, sizeof(binWidth));
}

CompactBruteScanIndex::~CompactBruteScanIndex() {
    // Wait for all GPU operations to complete before destroying buffers
    if (vd && vd->device) {
        vd->device->waitIdle();
    }
    
    if (startAddrBuffer) startAddrBuffer->destroy();
    if (countBuffer) countBuffer->destroy();
    if (extentBuffer) extentBuffer->destroy();
    if (capacityBuffer) capacityBuffer->destroy();
    if (mainDataBuffer) mainDataBuffer->destroy();
    if (auxDataBuffer) auxDataBuffer->destroy();
    if (auxCountBuffer) auxCountBuffer->destroy();
    if (statsBuffer) statsBuffer->destroy();
}

void CompactBruteScanIndex::initialize() {
    setupPipelines();
}

void CompactBruteScanIndex::allocateBuffers(uint32_t npoints) {
    this->npoints = npoints;
    std::cout << "[CompactBruteScan] INDEX_RESOLUTION: " << INDEX_RESOLUTION << "\n";
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    binRange = (npoints + totalBins - 1) / totalBins;
    
    // Calculate count buffer size for prefix sum compatibility
    size_t scanBufSize = scan->getBufSizeDivisor();
    countBufSize = size_t(std::ceil(double(totalBins + 1) / scanBufSize) * scanBufSize);
    
    // Allocate main buffer structures
    startAddrBuffer = std::make_shared<Buffer>(vd);
    startAddrBuffer->create(countBufSize * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);

    countBuffer = std::make_shared<Buffer>(vd);
    countBuffer->create(countBufSize * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
        MemoryType::Internal);

    extentBuffer = std::make_shared<Buffer>(vd);
    extentBuffer->create(totalBins * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
        MemoryType::Internal);
    
    capacityBuffer = std::make_shared<Buffer>(vd);
    capacityBuffer->create(totalBins * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
        MemoryType::Internal);

    // Main data buffer with scale factor
    mainAllocatedCapacity = (uint64_t)npoints * COMPACTBRUTE_INITIAL_SCALE_FACTOR;
    globalFreeOffset = npoints;
    
    mainDataBuffer = std::make_shared<Buffer>(vd);
    mainDataBuffer->create(mainAllocatedCapacity * sizeof(CompactBruteEntry), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | 
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress, 
        MemoryType::Internal);
    
    // Auxiliary buffer for inserts (fraction of main buffer)
    auxCapacity = (uint32_t)(npoints * COMPACTBRUTE_AUX_FRACTION);
    if (auxCapacity < 1000) auxCapacity = 1000; // Minimum size
    
    auxDataBuffer = std::make_shared<Buffer>(vd);
    auxDataBuffer->create(auxCapacity * sizeof(CompactBruteEntry), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | 
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress, 
        MemoryType::Internal);
    
    // Aux count buffer (single uint)
    auxCountBuffer = std::make_shared<Buffer>(vd);
    auxCountBuffer->create(sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    // Stats buffer
    statsBuffer = std::make_shared<Buffer>(vd);
    statsBuffer->create(2 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    // Initialize nextRowId to npoints (new inserts get IDs starting from here)
    nextRowId = npoints;
    
    std::cout << "[CompactBruteScan] Main buffer: " << mainAllocatedCapacity << " entries (" 
              << (mainAllocatedCapacity * sizeof(CompactBruteEntry) / (1024*1024.0)) << " MB)\n";
    std::cout << "[CompactBruteScan] Aux buffer: " << auxCapacity << " entries (" 
              << (auxCapacity * sizeof(CompactBruteEntry) / (1024*1024.0)) << " MB)\n";
}

void CompactBruteScanIndex::setupPipelines() {
    // ========== COMPUTE PIPELINES ==========
    // Descriptor Set Layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.push_back(vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute));
    bindings.push_back(vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute));
    bindings.push_back(vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute));
    bindings.push_back(vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute));
    bindings.push_back(vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute));
    
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, (uint32_t)bindings.size(), bindings.data());
    descSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
    
    vk::PushConstantRange pushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, 20 * sizeof(uint32_t));
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &descSetLayout.get(), 1, &pushConstantRange);
    pipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
    
    std::vector<vk::DescriptorPoolSize> poolSizes;
    poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 10));
    vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, (uint32_t)poolSizes.size(), poolSizes.data());
    descPool = vd->device->createDescriptorPoolUnique(poolInfo);
    
    vk::DescriptorSetAllocateInfo allocInfo(descPool.get(), 1, &descSetLayout.get());
    descSet = std::move(vd->device->allocateDescriptorSetsUnique(allocInfo)[0]);
    
    // Load dummy fragment shader
    {
        std::vector<uint32_t> fshader;
        validate(readShader(SHADER_FOLDER + "/dummy.frag.spv", fshader), "dummy fragment shader");
        vk::ShaderModuleCreateInfo createInfo({}, fshader.size() * sizeof(uint32_t), fshader.data());
        fragmentShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // ========== GRAPHICS PIPELINE: Build Count ==========
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/compactbrute_count_gfx.vert.spv", vshader), "compactbrute count vertex shader");
        vk::ShaderModuleCreateInfo createInfo({}, vshader.size() * sizeof(uint32_t), vshader.data());
        bcVertexShader = vd->device->createShaderModuleUnique(createInfo);
        
        bcPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, bcVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, fragmentShader.get(), "main")
        };
        bcPipelineProps.setShaderStageFlag();
        
        bcPipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, sizeof(uint32_t)),
            vk::VertexInputBindingDescription(1, sizeof(uint32_t))
        };
        bcPipelineProps.setInputBindingFlag();
        
        bcPipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
            vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32Uint, 0)
        };
        bcPipelineProps.setInputAttrFlag();
        
        bcPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        bcPipelineProps.setInputAssemblyFlag();
        
        bcPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}
        };
        bcPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1}
        };
        bcPipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, 5 * sizeof(uint32_t))
        };
        bcPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);
        
        vk::PipelineRenderingCreateInfo rpCreateInfo;
        rpCreateInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpCreateInfo.pColorAttachmentFormats = &colorFormat;
        
        vk::UniqueRenderPass dummyRenderPass;
        bcPipeline = bcPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
    }
    
    // ========== GRAPHICS PIPELINE: Build Insert ==========
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/compactbrute_build_gfx.vert.spv", vshader), "compactbrute build vertex shader");
        vk::ShaderModuleCreateInfo createInfo({}, vshader.size() * sizeof(uint32_t), vshader.data());
        bVertexShader = vd->device->createShaderModuleUnique(createInfo);
        
        bPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, bVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, fragmentShader.get(), "main")
        };
        bPipelineProps.setShaderStageFlag();
        
        bPipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, sizeof(uint32_t)),
            vk::VertexInputBindingDescription(1, sizeof(uint32_t)),
            vk::VertexInputBindingDescription(2, sizeof(uint32_t))
        };
        bPipelineProps.setInputBindingFlag();
        
        bPipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
            vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32Uint, 0),
            vk::VertexInputAttributeDescription(2, 2, vk::Format::eR32Uint, 0)
        };
        bPipelineProps.setInputAttrFlag();
        
        bPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        bPipelineProps.setInputAssemblyFlag();
        
        bPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}
        };
        bPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 2}
        };
        bPipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, 8 * sizeof(uint32_t))
        };
        bPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);
        
        vk::PipelineRenderingCreateInfo rpCreateInfo;
        rpCreateInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpCreateInfo.pColorAttachmentFormats = &colorFormat;
        
        vk::UniqueRenderPass dummyRenderPass;
        bPipeline = bPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
    }
    
    // ========== COMPUTE PIPELINES ==========
    // Query Pipeline (brute-force scan all entries)
    {
        std::vector<uint32_t> code;
        if(!vkcore::readShader(SHADER_FOLDER + "/compactbrute_query.comp.spv", code)) {
            throw std::runtime_error("Failed to load compactbrute_query.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        queryShader = vd->device->createShaderModuleUnique(createInfo);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, queryShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout.get());
        queryPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Delete Pipeline
    {
        std::vector<uint32_t> code;
        if(!vkcore::readShader(SHADER_FOLDER + "/compactbrute_delete.comp.spv", code)) {
            throw std::runtime_error("Failed to load compactbrute_delete.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        deleteShader = vd->device->createShaderModuleUnique(createInfo);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, deleteShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout.get());
        deletePipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Insert Pipeline
    {
        std::vector<uint32_t> code;
        if(!vkcore::readShader(SHADER_FOLDER + "/compactbrute_insert.comp.spv", code)) {
            throw std::runtime_error("Failed to load compactbrute_insert.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        insertShader = vd->device->createShaderModuleUnique(createInfo);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, insertShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout.get());
        insertPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Update Pipeline
    {
        std::vector<uint32_t> code;
        if(!vkcore::readShader(SHADER_FOLDER + "/compactbrute_update.comp.spv", code)) {
            throw std::runtime_error("Failed to load compactbrute_update.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        updateShader = vd->device->createShaderModuleUnique(createInfo);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, updateShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout.get());
        updatePipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Create dummy FBO for graphics pipelines
    dummyFbo = std::make_shared<FrameBuffer>(vd);
    dummyFbo->create(vk::Format::eR8Sint, INDEX_RESOLUTION, INDEX_RESOLUTION, 1, MemoryType::Internal);
}

void CompactBruteScanIndex::buildIndex(vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal) {
    std::cout << "[CompactBruteScan] Building index with " << npoints << " points...\n";
    
    allocateBuffers(npoints);
    initialize();
    
    // Store min/max values
    for(int i = 0; i < 3; i++) {
        this->minVal[i] = minVal[i];
        this->maxVal[i] = maxVal[i];
    }
    
    // Calculate bin widths
    uint32_t binRange0 = uint32_t(ceil(double(maxVal[0] - minVal[0]) / INDEX_RESOLUTION));
    uint32_t binRange1 = uint32_t(ceil(double(maxVal[1] - minVal[1]) / INDEX_RESOLUTION));
    uint32_t binRange2 = uint32_t(ceil(double(maxVal[2] - minVal[2]) / INDEX_RESOLUTION));
    if(binRange0 == 0) binRange0 = 1;
    if(binRange1 == 0) binRange1 = 1;
    if(binRange2 == 0) binRange2 = 1;
    binWidth[0] = binRange0;
    binWidth[1] = binRange1;
    binWidth[2] = binRange2;
    
    std::array<uint32_t, 5> gfxPC = {minVal[0], minVal[1], binRange0, binRange1, INDEX_RESOLUTION};
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    // Clear extentBuffer and auxCountBuffer
    extentBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader);
    auxCountBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eComputeShader);
    
    // ========== PASS 1: Count Points per Bin ==========
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRenderingCB(vd, dummyFbo, colorInfo);
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, bcPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        
        vk::DescriptorBufferInfo extentDescriptor{extentBuffer->buf, 0, VK_WHOLE_SIZE};
        std::vector<vk::WriteDescriptorSet> descriptorSets = {
            vk::WriteDescriptorSet{bcPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &extentDescriptor}
        };
        vd->device->updateDescriptorSets(descriptorSets, nullptr);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bcPipelineProps.pipelineLayout.get(), 0, bcPipelineProps.descriptorSet.get(), nullptr);
        
        vd->commandBuffer->pushConstants<uint32_t>(bcPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, gfxPC);
        
        vk::DeviceSize offset = 0;
        vd->commandBuffer->bindVertexBuffers(0, pointsBuffer->buf, offset);
        offset += npoints * sizeof(uint32_t);
        vd->commandBuffer->bindVertexBuffers(1, pointsBuffer->buf, offset);
        
        vd->commandBuffer->draw(npoints, 1, 0, 0);
        vd->commandBuffer->endRendering();
    }
    
    // Barrier and copy to capacityBuffer
    extentBuffer->barrier(vk::PipelineStageFlagBits::eVertexShader, vk::PipelineStageFlagBits::eTransfer,
                         vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead);
    capacityBuffer->copyFrom(totalBins * sizeof(uint32_t), 0, 0, extentBuffer);
    
    // Copy to startAddrBuffer for prefix sum
    startAddrBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eTransfer);
    startAddrBuffer->copyFrom(totalBins * sizeof(uint32_t), 0, 0, extentBuffer);
    
    capacityBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
    startAddrBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                             vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    
    // ========== PASS 2: GPU Prefix Sum ==========
    if(scan) {
        scan->prefixSum(startAddrBuffer->buf, countBufSize);
    } else {
        std::cerr << "[CompactBruteScan] ERROR: No SinglePassScan provided!\n";
        vd->commandBuffer->end();
        return;
    }
    
    startAddrBuffer->barrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eVertexShader,
                             vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
    
    // Clear countBuffer for build pass
    countBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader, 0);
    
    // ========== PASS 3: Insert Points into Bins ==========
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRenderingCB(vd, dummyFbo, colorInfo);
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, bPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        
        vk::DescriptorBufferInfo countDesc{countBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo startDesc{startAddrBuffer->buf, 0, VK_WHOLE_SIZE};
        std::vector<vk::WriteDescriptorSet> descriptorSets = {
            vk::WriteDescriptorSet{bPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDesc},
            vk::WriteDescriptorSet{bPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &startDesc}
        };
        vd->device->updateDescriptorSets(descriptorSets, nullptr);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bPipelineProps.pipelineLayout.get(), 0, bPipelineProps.descriptorSet.get(), nullptr);
        
        uint64_t dataAddr = mainDataBuffer->getDeviceAddress();
        std::array<uint32_t, 8> gfxPCWithAddr = {gfxPC[0], gfxPC[1], gfxPC[2], gfxPC[3], gfxPC[4], 0, 
                                                  static_cast<uint32_t>(dataAddr & 0xFFFFFFFF), 
                                                  static_cast<uint32_t>(dataAddr >> 32)};
        vd->commandBuffer->pushConstants<uint32_t>(bPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, gfxPCWithAddr);
        
        vk::DeviceSize offset = 0;
        vd->commandBuffer->bindVertexBuffers(0, pointsBuffer->buf, offset);
        offset += npoints * sizeof(uint32_t);
        vd->commandBuffer->bindVertexBuffers(1, pointsBuffer->buf, offset);
        offset += npoints * sizeof(uint32_t);
        vd->commandBuffer->bindVertexBuffers(2, pointsBuffer->buf, offset);
        
        vd->commandBuffer->draw(npoints, 1, 0, 0);
        vd->commandBuffer->endRendering();
    }
    
    vd->commandBuffer->end();
    
    // Submit and wait
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
    
    std::cout << "[CompactBruteScan] Build complete.\n";
}

void CompactBruteScanIndex::runRangeQueries(vkcore::PBuffer queryBuffer, uint32_t nqueries, vkcore::PBuffer resultBuffer) {
    // Create cached fence on first use
    if (!queryFence) {
        queryFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    }
    
    // Use cached aux count (no GPU readback needed)
    uint32_t currentAuxCount = cachedAuxCount;
    
    // Update descriptors
    vk::DescriptorBufferInfo queryInfo(queryBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo resultInfo(resultBuffer->buf, 0, VK_WHOLE_SIZE);
    std::vector<vk::WriteDescriptorSet> writes = {
        {descSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &queryInfo},
        {descSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &resultInfo},
    };
    vd->device->updateDescriptorSets(writes, {});
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    // Clear result buffer (use VK_WHOLE_SIZE to fill entire buffer)
    vd->commandBuffer->fillBuffer(resultBuffer->buf, 0, VK_WHOLE_SIZE, 0);
    
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
    
    // Run brute-force query
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, queryPipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descSet.get(), {});
    
    // Push constants: mainCount, auxCount, mainBufferAddr, auxBufferAddr
    uint64_t mainAddr = mainDataBuffer->getDeviceAddress();
    uint64_t auxAddr = auxDataBuffer->getDeviceAddress();
    struct {
        uint32_t mainCount;
        uint32_t auxCount;
        uint32_t mainAddrLo, mainAddrHi;
        uint32_t auxAddrLo, auxAddrHi;
    } pushData = {
        npoints,
        currentAuxCount,
        static_cast<uint32_t>(mainAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(mainAddr >> 32),
        static_cast<uint32_t>(auxAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(auxAddr >> 32)
    };
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(pushData), &pushData);
    
    // Dispatch: scan all entries in both buffers
    uint32_t totalEntries = npoints + currentAuxCount;
    uint32_t numWorkgroups = (totalEntries + 255) / 256;
    vd->commandBuffer->dispatch(numWorkgroups, 1, 1);
    
    vd->commandBuffer->end();
    
    vd->device->resetFences(*queryFence);
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vd->submit(submitInfo, *queryFence, false);
    vd->waitForFences(*queryFence, VK_TRUE, UINT64_MAX);
}

void CompactBruteScanIndex::deletePoints(vkcore::PBuffer dataBuffer, uint32_t ndeletes) {
    // Use cached aux count (no GPU readback needed)
    uint32_t currentAuxCount = cachedAuxCount;
    
    // Update descriptors
    vk::DescriptorBufferInfo startInfo(startAddrBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo extentInfo(extentBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo deleteInfo(dataBuffer->buf, 0, VK_WHOLE_SIZE);
    
    std::vector<vk::WriteDescriptorSet> writes = {
        {descSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &startInfo},
        {descSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &extentInfo},
        {descSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &deleteInfo},
    };
    vd->device->updateDescriptorSets(writes, {});
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, deletePipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descSet.get(), {});
    
    // Push constants - includes aux buffer for searching both buffers
    uint64_t mainAddr = mainDataBuffer->getDeviceAddress();
    uint64_t auxAddr = auxDataBuffer->getDeviceAddress();
    uint32_t pc[14] = {
        minVal[0], minVal[1], minVal[2],
        INDEX_RESOLUTION,
        binWidth[0], binWidth[1], binWidth[2],
        ndeletes,
        static_cast<uint32_t>(mainAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(mainAddr >> 32),
        static_cast<uint32_t>(auxAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(auxAddr >> 32),
        currentAuxCount,
        0  // padding
    };
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), pc);
    
    // THREADS_PER_DELETE threads per delete request (must match shader define)
    const uint32_t THREADS_PER_DELETE = 1024;
    const uint32_t LOCAL_SIZE = 1024;
    uint32_t totalThreads = ndeletes * THREADS_PER_DELETE;
    uint32_t groups = (totalThreads + LOCAL_SIZE - 1) / LOCAL_SIZE;
    vd->commandBuffer->dispatch(groups, 1, 1);
    
    vd->commandBuffer->end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
}

void CompactBruteScanIndex::insertPoints(vkcore::PBuffer pointsBuffer, uint32_t ninserts) {
    // Update descriptors
    vk::DescriptorBufferInfo auxCountInfo(auxCountBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo insertInfo(pointsBuffer->buf, 0, VK_WHOLE_SIZE);
    
    std::vector<vk::WriteDescriptorSet> writes = {
        {descSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &auxCountInfo},
        {descSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &insertInfo},
    };
    vd->device->updateDescriptorSets(writes, {});
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, insertPipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descSet.get(), {});
    
    // Push constants
    uint64_t auxAddr = auxDataBuffer->getDeviceAddress();
    struct {
        uint32_t ninserts;
        uint32_t nextRowId;
        uint32_t auxCapacity;
        uint32_t pad;
        uint32_t auxAddrLo, auxAddrHi;
    } pushData = {
        ninserts,
        nextRowId,
        auxCapacity,
        0,
        static_cast<uint32_t>(auxAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(auxAddr >> 32)
    };
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(pushData), &pushData);
    
    uint32_t groups = (ninserts + 255) / 256;
    vd->commandBuffer->dispatch(groups, 1, 1);
    
    vd->commandBuffer->end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
    
    // Update nextRowId for future inserts
    nextRowId += ninserts;
    
    // Update cached aux count (avoid GPU readback in queries)
    cachedAuxCount += ninserts;
}

void CompactBruteScanIndex::updatePoints(vkcore::PBuffer dataBuffer, uint32_t nupdates) {
    // Update descriptors
    vk::DescriptorBufferInfo startInfo(startAddrBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo extentInfo(extentBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo updateInfo(dataBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo auxCountInfo(auxCountBuffer->buf, 0, VK_WHOLE_SIZE);
    
    std::vector<vk::WriteDescriptorSet> writes = {
        {descSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &startInfo},
        {descSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &extentInfo},
        {descSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &updateInfo},
        {descSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &auxCountInfo},
    };
    vd->device->updateDescriptorSets(writes, {});
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, updatePipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descSet.get(), {});
    
    // Push constants
    uint64_t mainAddr = mainDataBuffer->getDeviceAddress();
    uint64_t auxAddr = auxDataBuffer->getDeviceAddress();
    uint32_t pc[16] = {
        minVal[0], minVal[1], minVal[2],
        INDEX_RESOLUTION,
        binWidth[0], binWidth[1], binWidth[2],
        nupdates,
        auxCapacity, 0,  // auxCapacity, pad
        static_cast<uint32_t>(mainAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(mainAddr >> 32),
        static_cast<uint32_t>(auxAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(auxAddr >> 32),
        0, 0  // extra padding
    };
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, 14 * sizeof(uint32_t), pc);
    
    uint32_t groups = (nupdates + 255) / 256;
    vd->commandBuffer->dispatch(groups, 1, 1);
    
    vd->commandBuffer->end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
}

uint32_t CompactBruteScanIndex::getMainValidCount() {
    // Count valid entries in main buffer (for verification)
    PBuffer stagingBuf = std::make_shared<Buffer>(vd);
    size_t dataSize = mainAllocatedCapacity * sizeof(CompactBruteEntry);
    stagingBuf->create(dataSize, vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadOnly);
    
    std::vector<CompactBruteEntry> entries(mainAllocatedCapacity);
    readUsingStagingBuf((char*)entries.data(), dataSize, mainDataBuffer, stagingBuf, vd);
    stagingBuf->destroy();
    
    uint32_t validCount = 0;
    for (uint64_t i = 0; i < mainAllocatedCapacity; i++) {
        if (entries[i].rowId & 0x80000000u) validCount++;
    }
    return validCount;
}

uint32_t CompactBruteScanIndex::getAuxValidCount() {
    // Read aux count
    uint32_t auxCount = 0;
    PBuffer stagingBuf = std::make_shared<Buffer>(vd);
    stagingBuf->create(sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadOnly);
    readUsingStagingBuf((char*)&auxCount, sizeof(uint32_t), auxCountBuffer, stagingBuf, vd);
    stagingBuf->destroy();
    return auxCount;
}
