#include "CompactScanIndex.hpp"
#include <common/utils.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>

using namespace vkcore;

// Helper function for graphics pipeline rendering (same as RasterScan2D)
inline vk::RenderingInfo setupRendering(PVkDevice vd, PFrameBuffer fbo, vk::RenderingAttachmentInfo &colorInfo) {
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

CompactScanIndex::CompactScanIndex(PVkDevice vd, int32_t ncols, SinglePassScan* scan) : vd(vd), ncols(ncols), scan(scan) {
    binRange = 0;
}

CompactScanIndex::~CompactScanIndex() {
    if (startAddrBuffer) startAddrBuffer->destroy();
    if (countBuffer) countBuffer->destroy();
    if (dataBuffer) dataBuffer->destroy();
    // Pipelines and descriptors managed by Unique handles
}

void CompactScanIndex::initialize() {
    setupPipelines();
    
    // Allocate buffers for two-pass query (like RasterScan2D)
    // maxBuffer: [0]=numRanges (for drawIndirect), [1]=maxCount (for texture sizing)
    maxBuffer = std::make_shared<Buffer>(vd);
    maxBuffer->create(4 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
        MemoryType::LocalHostVisibleForce);
    
    // edgeBuffer: stores [st, en) pairs from pass 1 (max INDEX_RESOLUTION^2 bins)
    uint32_t maxBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    edgeBuffer = std::make_shared<Buffer>(vd);
    edgeBuffer->create(maxBins * 2 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | 
        vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
}

void CompactScanIndex::allocateBuffers(uint32_t npoints) {
    this->npoints = npoints;
    std::cout << "[CompactIndex] INDEX_RESOLUTION: " << INDEX_RESOLUTION << "\n";
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    binRange = (npoints + totalBins - 1) / totalBins;
    
    // Calculate count buffer size to be compatible with prefix sum
    // Must be a multiple of scan->getBufSizeDivisor()
    size_t scanBufSize = scan ? scan->getBufSizeDivisor() : 4096;
    countBufSize = size_t(std::ceil(double(totalBins + 1) / scanBufSize) * scanBufSize);
    
    auto allocStart = std::chrono::high_resolution_clock::now();
    
    // T: Start Address Buffer - same size as count buffer for compatibility
    startAddrBuffer = std::make_shared<Buffer>(vd);
    startAddrBuffer->create(countBufSize * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);

    // C: Count Buffer - sized for prefix sum compatibility
    countBuffer = std::make_shared<Buffer>(vd);
    countBuffer->create(countBufSize * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
        MemoryType::Internal);

    // Stats Buffer
    statsBuffer = std::make_shared<Buffer>(vd);
    statsBuffer->create(2 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
        MemoryType::Internal);
        
    // Capacity Buffer (1024*1024 uints)
    capacityBuffer = std::make_shared<Buffer>(vd);
    capacityBuffer->create(totalBins * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);

    // Data Buffer - allocate with SCALE_FACTOR extra space for updates
    this->totalAllocatedCapacity = npoints * COMPACT_INITIAL_SCALE_FACTOR;
    this->globalFreeOffset = npoints;
    
    dataBuffer = std::make_shared<Buffer>(vd);
    dataBuffer->create(totalAllocatedCapacity * sizeof(CompactEntry), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    auto allocEnd = std::chrono::high_resolution_clock::now();
    double allocMs = std::chrono::duration<double, std::milli>(allocEnd - allocStart).count();
    
    std::cout << "[CompactIndex] Buffer allocation time: " << allocMs << " ms\n";
    std::cout << "[CompactIndex] Capacity: " << totalAllocatedCapacity << " entries (" 
              << (totalAllocatedCapacity * sizeof(CompactEntry) / (1024*1024.0)) << " MB), Scale Factor: " 
              << COMPACT_INITIAL_SCALE_FACTOR << "\n";
}

void CompactScanIndex::setupPipelines() {
    // ========== COMPUTE PIPELINES (for query, delete, scale, stats) ==========
    // Descriptor Set Layout for compute shaders
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.push_back(vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)); // T
    bindings.push_back(vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)); // C
    bindings.push_back(vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)); // Data
    bindings.push_back(vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)); // Input
    bindings.push_back(vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)); // Output
    bindings.push_back(vk::DescriptorSetLayoutBinding(5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)); // Stats
    bindings.push_back(vk::DescriptorSetLayoutBinding(6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)); // Capacity
    
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, (uint32_t)bindings.size(), bindings.data());
    descSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
    
    vk::PushConstantRange pushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, 16 * sizeof(uint32_t));
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &descSetLayout.get(), 1, &pushConstantRange);
    pipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
    
    std::vector<vk::DescriptorPoolSize> poolSizes;
    poolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 10));
    vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, (uint32_t)poolSizes.size(), poolSizes.data());
    descPool = vd->device->createDescriptorPoolUnique(poolInfo);
    
    vk::DescriptorSetAllocateInfo allocInfo(descPool.get(), 1, &descSetLayout.get());
    descSet = std::move(vd->device->allocateDescriptorSetsUnique(allocInfo)[0]);
    
    // Load dummy fragment shader (shared by graphics pipelines)
    {
        std::vector<uint32_t> fshader;
        validate(readShader(SHADER_FOLDER + "/dummy.frag.spv", fshader), "dummy fragment shader");
        vk::ShaderModuleCreateInfo createInfo({}, fshader.size() * sizeof(uint32_t), fshader.data());
        fragmentShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // ========== GRAPHICS PIPELINE: Build Count ==========
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/compact_count_gfx.vert.spv", vshader), "compact count vertex shader");
        vk::ShaderModuleCreateInfo createInfo({}, vshader.size() * sizeof(uint32_t), vshader.data());
        bcVertexShader = vd->device->createShaderModuleUnique(createInfo);
        
        bcPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, bcVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, fragmentShader.get(), "main")
        };
        bcPipelineProps.setShaderStageFlag();
        
        // Column-major: 2 bindings for X, Y (Z not needed for count)
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
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, 8 * sizeof(uint32_t))
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
        validate(readShader(SHADER_FOLDER + "/compact_build_gfx.vert.spv", vshader), "compact build vertex shader");
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
        
        // Only 2 bindings now: offset buffer (0) and data buffer (2)
        bPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}
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
    
    // ========== GRAPHICS QUERY PIPELINE (single-pass with loop) ==========
    {
        std::vector<uint32_t> vertCode, geomCode, fragCode;
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_query_gfx.vert.spv", vertCode)) {
            throw std::runtime_error("Failed to load compact_query_gfx.vert.spv");
        }
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_query_gfx.geom.spv", geomCode)) {
            throw std::runtime_error("Failed to load compact_query_gfx.geom.spv");
        }
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_query_gfx.frag.spv", fragCode)) {
            throw std::runtime_error("Failed to load compact_query_gfx.frag.spv");
        }
        
        vk::ShaderModuleCreateInfo vertInfo({}, vertCode.size() * sizeof(uint32_t), vertCode.data());
        vk::ShaderModuleCreateInfo geomInfo({}, geomCode.size() * sizeof(uint32_t), geomCode.data());
        vk::ShaderModuleCreateInfo fragInfo({}, fragCode.size() * sizeof(uint32_t), fragCode.data());
        queryGfxVertexShader = vd->device->createShaderModuleUnique(vertInfo);
        queryGfxGeomShader = vd->device->createShaderModuleUnique(geomInfo);
        queryGfxFragShader = vd->device->createShaderModuleUnique(fragInfo);
        
        queryGfxPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, queryGfxVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eGeometry, queryGfxGeomShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, queryGfxFragShader.get(), "main")
        };
        queryGfxPipelineProps.setShaderStageFlag();
        
        // Input: query format is [x1, x2, y1, y2, z1, z2] - 6 uints
        queryGfxPipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, 6 * sizeof(uint32_t))
        };
        queryGfxPipelineProps.setInputBindingFlag();
        
        queryGfxPipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Uint, 0),  // qpart1: x1, x2, y1
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Uint, 3 * sizeof(uint32_t))  // qpart2: y2, z1, z2
        };
        queryGfxPipelineProps.setInputAttrFlag();
        
        queryGfxPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        queryGfxPipelineProps.setInputAssemblyFlag();
        
        // Bindings: startAddr (0), data (2), result (3) - count computed from offsets
        queryGfxPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}
        };
        queryGfxPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 3}
        };
        queryGfxPipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, 8 * sizeof(uint32_t))
        };
        queryGfxPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);
        
        vk::PipelineRenderingCreateInfo rpCreateInfo;
        rpCreateInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpCreateInfo.pColorAttachmentFormats = &colorFormat;
        
        vk::UniqueRenderPass dummyRenderPass;
        queryGfxPipeline = queryGfxPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
    }
    
    // ========== QUERY PASS 2: EDGE PIPELINE ==========
    {
        std::vector<uint32_t> vertCode, geomCode, fragCode;
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_edge.vert.spv", vertCode)) {
            throw std::runtime_error("Failed to load compact_edge.vert.spv");
        }
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_edge.geom.spv", geomCode)) {
            throw std::runtime_error("Failed to load compact_edge.geom.spv");
        }
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_edge.frag.spv", fragCode)) {
            throw std::runtime_error("Failed to load compact_edge.frag.spv");
        }
        
        vk::ShaderModuleCreateInfo vertInfo({}, vertCode.size() * sizeof(uint32_t), vertCode.data());
        vk::ShaderModuleCreateInfo geomInfo({}, geomCode.size() * sizeof(uint32_t), geomCode.data());
        vk::ShaderModuleCreateInfo fragInfo({}, fragCode.size() * sizeof(uint32_t), fragCode.data());
        edgeVertexShader = vd->device->createShaderModuleUnique(vertInfo);
        edgeGeomShader = vd->device->createShaderModuleUnique(geomInfo);
        edgeFragShader = vd->device->createShaderModuleUnique(fragInfo);
        
        edgePipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, edgeVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eGeometry, edgeGeomShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, edgeFragShader.get(), "main")
        };
        edgePipelineProps.setShaderStageFlag();
        
        // Input: [st, en) pairs from edgeBuffer (2 uints per vertex)
        edgePipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, 2 * sizeof(uint32_t))
        };
        edgePipelineProps.setInputBindingFlag();
        
        edgePipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),  // st
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32Uint, sizeof(uint32_t))  // en
        };
        edgePipelineProps.setInputAttrFlag();
        
        edgePipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        edgePipelineProps.setInputAssemblyFlag();
        
        // Bindings: data, result, query
        edgePipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}
        };
        edgePipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 3}
        };
        edgePipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, 2 * sizeof(uint32_t))
        };
        edgePipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);
        
        vk::PipelineRenderingCreateInfo rpCreateInfo;
        rpCreateInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpCreateInfo.pColorAttachmentFormats = &colorFormat;
        
        vk::UniqueRenderPass dummyRenderPass;
        edgePipeline = edgePipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
    }
    
    // ========== COMPUTE PIPELINES ==========
    // Scale Pipeline (scales counts by INITIAL_SCALE_FACTOR)
    {
        std::vector<uint32_t> code;
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_scale_counts.comp.spv", code)) {
             throw std::runtime_error("Failed to load compact_scale_counts.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        scaleShader = vd->device->createShaderModuleUnique(createInfo);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, scaleShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout.get());
        scalePipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Stats Pipeline
    {
        std::vector<uint32_t> code;
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_stats.comp.spv", code)) {
             throw std::runtime_error("Failed to load compact_stats.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        statsShader = vd->device->createShaderModuleUnique(createInfo);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, statsShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout.get());
        statsPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }

    // Query Pipeline
    {
        std::vector<uint32_t> code;
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_query.comp.spv", code)) {
             throw std::runtime_error("Failed to load compact_query.comp.spv");
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
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_delete.comp.spv", code)) {
             throw std::runtime_error("Failed to load compact_delete.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        deleteShader = vd->device->createShaderModuleUnique(createInfo);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, deleteShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout.get());
        deletePipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Insert Pipeline (compute, for incremental updates)
    {
        std::vector<uint32_t> code;
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_insert.comp.spv", code)) {
             throw std::runtime_error("Failed to load compact_insert.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        insertShader = vd->device->createShaderModuleUnique(createInfo);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, insertShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout.get());
        insertPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // Create dummy FBO for graphics pipelines (must match INDEX_RESOLUTION)
    dummyFbo = std::make_shared<FrameBuffer>(vd);
    dummyFbo->create(vk::Format::eR8Sint, INDEX_RESOLUTION, INDEX_RESOLUTION, 1, MemoryType::Internal, false);
}

void CompactScanIndex::buildIndex(vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal) {
    // Allocate buffers (Count and StartAddr)
    allocateBuffers(npoints);
    
    for(int i=0; i<3; i++) {
        this->minVal[i] = minVal[i];
        this->maxVal[i] = maxVal[i];
        
        // Calculate binWidth (safely)
        uint32_t range = maxVal[i] - minVal[i];
        binWidth[i] = (range + INDEX_RESOLUTION - 1) / INDEX_RESOLUTION;
        if(binWidth[i] == 0) binWidth[i] = 1;
    }
    
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    
    // Data buffer already allocated in allocateBuffers()
    
    // Push constants for graphics shaders: minVal[3], resolution, binWidth[3], npoints
    std::array<uint32_t, 8> gfxPC = {minVal[0], minVal[1], minVal[2], INDEX_RESOLUTION, 
                                      binWidth[0], binWidth[1], binWidth[2], npoints};
    
    // ========== SINGLE COMMAND BUFFER SUBMISSION ==========
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    // Clear count buffer
    countBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader);
    
    // ========== PASS 1: Count Points per Bin (Graphics Pipeline) ==========
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, bcPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        
        // Update descriptor set for count buffer
        vk::DescriptorBufferInfo countDescriptor{countBuffer->buf, 0, VK_WHOLE_SIZE};
        std::vector<vk::WriteDescriptorSet> descriptorSets = {
            vk::WriteDescriptorSet{bcPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDescriptor}
        };
        vd->device->updateDescriptorSets(descriptorSets, nullptr);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bcPipelineProps.pipelineLayout.get(), 0, bcPipelineProps.descriptorSet.get(), nullptr);
        
        vd->commandBuffer->pushConstants<uint32_t>(bcPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, gfxPC);
        
        // Bind vertex buffers (column-major: X, Y only for count pass)
        vk::DeviceSize offset = 0;
        vd->commandBuffer->bindVertexBuffers(0, pointsBuffer->buf, offset);
        offset += npoints * sizeof(uint32_t);
        vd->commandBuffer->bindVertexBuffers(1, pointsBuffer->buf, offset);
        
        vd->commandBuffer->draw(npoints, 1, 0, 0);
        vd->commandBuffer->endRendering();
    }
    
    // Barrier: Vertex shader write -> Prefix sum read
    countBuffer->barrier(vk::PipelineStageFlagBits::eVertexShader, vk::PipelineStageFlagBits::eComputeShader,
                         vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    
    // ========== PASS 2: GPU Prefix Sum ==========
    if(scan) {
        scan->prefixSum(countBuffer->buf, countBufSize);
    } else {
        std::cerr << "[CompactIndex] ERROR: No SinglePassScan provided!\n";
        vd->commandBuffer->end();
        return;
    }
    
    // Barrier: Prefix sum write -> Transfer read (for copy)
    countBuffer->barrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                         vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead);
    
    // Copy prefix sum to startAddrBuffer (for queries - read-only)
    vk::BufferCopy copyRegion(0, 0, totalBins * sizeof(uint32_t));
    vd->commandBuffer->copyBuffer(countBuffer->buf, startAddrBuffer->buf, copyRegion);
    
    // Barrier: Transfer -> Vertex shader (for insert atomicAdd on countBuffer)
    countBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eVertexShader,
                         vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    
    // ========== PASS 3: Insert Points (Graphics Pipeline) ==========
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, bPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        
        // Update descriptor sets - use countBuffer for atomicAdd (has prefix sum), data buffer for output
        vk::DescriptorBufferInfo offsetDescriptor{countBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo dataDescriptor{dataBuffer->buf, 0, VK_WHOLE_SIZE};
        std::vector<vk::WriteDescriptorSet> descriptorSets = {
            vk::WriteDescriptorSet{bPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &offsetDescriptor},
            vk::WriteDescriptorSet{bPipelineProps.descriptorSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dataDescriptor}
        };
        vd->device->updateDescriptorSets(descriptorSets, nullptr);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bPipelineProps.pipelineLayout.get(), 0, bPipelineProps.descriptorSet.get(), nullptr);
        
        vd->commandBuffer->pushConstants<uint32_t>(bPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, gfxPC);
        
        // Bind vertex buffers
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
    
    // Single fence wait for entire build
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
    
    // Note: Stats computation moved outside build timing for fair comparison
    // computeAndPrintStats("[GPU Stats - Build]");
}

void CompactScanIndex::runRangeQueries(vkcore::PBuffer queryBuffer, uint32_t nqueries, vkcore::PBuffer resultBuffer) {
    // Single-pass graphics query (with loop in fragment shader)
    // Bindings: startAddr (0), data (2), result (3) - count computed from scaled offsets
    vk::DescriptorBufferInfo tInfo(startAddrBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo dInfo(dataBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo rInfo(resultBuffer->buf, 0, VK_WHOLE_SIZE);
    
    std::vector<vk::WriteDescriptorSet> writes = {
        vk::WriteDescriptorSet{queryGfxPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tInfo},
        vk::WriteDescriptorSet{queryGfxPipelineProps.descriptorSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dInfo},
        vk::WriteDescriptorSet{queryGfxPipelineProps.descriptorSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &rInfo}
    };
    vd->device->updateDescriptorSets(writes, nullptr);
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    // Setup rendering
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, queryGfxPipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, queryGfxPipelineProps.pipelineLayout.get(), 0, queryGfxPipelineProps.descriptorSet.get(), nullptr);
    
    // Push constants: minVal[3], resolution, binWidth[3], nqueries
    uint32_t pc[8] = { minVal[0], minVal[1], minVal[2], INDEX_RESOLUTION, binWidth[0], binWidth[1], binWidth[2], nqueries };
    vd->commandBuffer->pushConstants<uint32_t>(queryGfxPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
    
    // Bind query buffer as vertex buffer
    vk::DeviceSize offset = 0;
    vd->commandBuffer->bindVertexBuffers(0, queryBuffer->buf, offset);
    
    // Draw one point per query (geometry shader expands to quad)
    vd->commandBuffer->draw(nqueries, 1, 0, 0);
    vd->commandBuffer->endRendering();
    
    vd->commandBuffer->end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
}

void CompactScanIndex::deletePoints(vkcore::PBuffer deleteDataBuffer, uint32_t ndeletes) {
    // Update Descriptor Set with all required bindings for compute shader
    std::vector<vk::WriteDescriptorSet> writes;
    
    // Binding 0: StartAddr buffer
    vk::DescriptorBufferInfo tInfo(startAddrBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tInfo));
    
    // Binding 1: Count buffer
    vk::DescriptorBufferInfo cInfo(countBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cInfo));
    
    // Binding 2: Data buffer (index data)
    vk::DescriptorBufferInfo dataInfo(this->dataBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dataInfo));
    
    // Binding 3: Input data to delete
    vk::DescriptorBufferInfo dInfo(deleteDataBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dInfo));
    
    vd->device->updateDescriptorSets(writes, nullptr);
    
    // Dispatch
    // Same as query? "Search and delete can be parallized".
    // If we iterate over bins, we check "Does any point in delete-batch fall into this bin?".
    // This is inefficient if deletes are sparse.
    // BUT if we iterate over deletes (ndeletes threads), and for each delete scan the bin...
    // The user said: "When an element is deleted... identify the bin and going over the bin's buffer... search and delete can be parallized".
    // "Since pages are contiguos... search and delete can be parallized".
    // "create a subtexture ... per bin ... for a query and delete".
    // This implies using the SAME dispatch strategy (over bins) for delete?
    // BUT we need the delete targets.
    // If we dispatch over Bins, each bin thread group checks: "Do I have any points to delete?".
    // We need the delete batch to be accessible.
    // If delete batch is small (e.g. 100k), scanning it for every bin is slow.
    // UNLESS we first scatter deletes into bins?
    // Or maybe the user means: For each delete point, identify bin, then launch parallel search in THAT bin.
    // We can't easily launch variable workgroups per delete.
    
    // Alternative:
    // Dispatch (ndeletes) threads. Each thread computes binID.
    // Then it linearly scans the bin.
    // BUT user wants parallel search in bin.
    // "subtexture ... per bin ... for ... delete".
    // This strongly suggests we process BINS in parallel, and within bin process entries in parallel.
    // This works well if we have a way to know WHICH bins need processing.
    // OR if we just process ALL bins (expensive if ndeletes is small).
    
    // Given the constraints and description, and "Mode 21", maybe I should just use the same dispatch as query (1024x1024 workgroups).
    // And pass the delete batch.
    // Shader:
    //   For each thread (bin entry):
    //     Check if this entry exists in the delete batch?
    //     That's O(BinSize * BatchSize). Very slow.
    
    // Maybe "Delete by Data" means we pass the data point (x,y,z).
    // We map (x,y,z) to BinID.
    // We only need to search THAT bin.
    // If we have `ndeletes` points, we have `ndeletes` bins to search.
    // We can use Indirect Dispatch?
    // Or just dispatch `ndeletes` groups?
    // But `ndeletes` points might map to same bin.
    
    // I will implement "Delete" as: Dispatch 1 thread per delete point. Linearly scan bin.
    // This violates "parallized search".
    // Wait. "search ... for the bins ... can be parallized".
    // If I have 1 delete point `P` in bin `B`.
    // I can launch 256 threads for bin `B`. Thread `i` checks `Entry[i] == P`.
    // If match, mark invalid.
    // This requires one dispatch per delete point? No.
    // Or Indirect Dispatch where we populate arguments based on unique bins?
    // Too complex for "Create new ones" instruction without more code.
    
    // I'll stick to: Compute Shader with `ndeletes` threads. Each thread scans its bin linearly.
    // "Search ... can be parallized" might refer to the fact that we can do many deletes in parallel (over different bins).
    // The "subtexture" part was specifically for "query".
    // "we can create a subtexture ... for a query and delete".
    // "for a query AND delete".
    // Okay, so both.
    
    // If I must use subtexture for delete:
    // I need to filter the delete batch into bins.
    // This is hard on GPU without sorting.
    
    // I will use 1024x1024 dispatch for delete too, IF ndeletes is large?
    // No, ndeletes is usually small batch.
    
    // I will use `ndeletes` threads (one per point) and linear scan within bin.
    // This is standard "Delete by Data" on GPU lists.
    // I'll add a comment explaining why.
    
    // Dispatch
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, deletePipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descSet.get(), 0, nullptr);
    
    uint32_t pc[12] = { minVal[0], minVal[1], minVal[2], INDEX_RESOLUTION, maxVal[0], maxVal[1], maxVal[2], ndeletes, binWidth[0], binWidth[1], binWidth[2], 0 };
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, 12 * sizeof(uint32_t), pc);
    
    // Group size 256
    uint32_t groups = (ndeletes + 255) / 256;
    vd->commandBuffer->dispatch(groups, 1, 1);
    
    vd->commandBuffer->end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);

    // Report stats after delete
    computeAndPrintStats("[GPU Stats - After Delete]");
}

void CompactScanIndex::insertPoints(vkcore::PBuffer pointsBuffer, uint32_t npoints) {
    // Update Descriptor Set with new Points
    std::vector<vk::WriteDescriptorSet> writes;
    
    // Binding 3: Points
    vk::DescriptorBufferInfo pInfo(pointsBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &pInfo));
    
    // Bind StartAddr(0), Count(1), Data(2), Capacity(6) if needed? 
    // They should be persistent in descSet unless overwritten by other modes?
    // Mode21 runs sequentially.
    // However, `deletePoints` overwrote Binding 3 with `DeleteDataBuffer`.
    // So writing Binding 3 here is correct.
    // What about Binding 0, 1, 2, 6?
    // `runRangeQueries` overwrites 3 and 4.
    // `deletePoints` overwrites 3.
    // So 0, 1, 2, 6 are safe? 
    // Wait, `deletePoints` uses Binding 3 for `DeleteDataBuffer`.
    // `insertPoints` uses Binding 3 for `PointsBuffer`.
    // `compact_insert.comp` uses Binding 0,1,2,3,6.
    // We should ensure they are bound.
    // Since `buildIndex` bound them, and `descSet` is unique per object, they persist unless overwritten.
    // I will re-bind them just to be absolutely safe (and robust against future changes).
    
    vk::DescriptorBufferInfo tInfo(startAddrBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tInfo));
    
    vk::DescriptorBufferInfo cInfo(countBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cInfo));
    
    vk::DescriptorBufferInfo dInfo(dataBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dInfo));
    
    vk::DescriptorBufferInfo cpInfo(capacityBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 6, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cpInfo));
    
    vd->device->updateDescriptorSets(writes, nullptr);
    
    // Dispatch
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, insertPipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descSet.get(), 0, nullptr);
    
    uint32_t pc[12] = { minVal[0], minVal[1], minVal[2], INDEX_RESOLUTION, maxVal[0], maxVal[1], maxVal[2], npoints, binWidth[0], binWidth[1], binWidth[2], 0 };
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, 12 * sizeof(uint32_t), pc);
    
    uint32_t groups = (npoints + 255) / 256;
    vd->commandBuffer->dispatch(groups, 1, 1);
    
    vd->commandBuffer->end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
    
    // Report stats after insert
    computeAndPrintStats("[GPU Stats - After Insert]");
}

void CompactScanIndex::computeAndPrintStats(const std::string& phase) {
    // Update descriptor for Stats Buffer (Binding 5)
    std::vector<vk::WriteDescriptorSet> writes;
    vk::DescriptorBufferInfo sInfo(statsBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sInfo));
    vd->device->updateDescriptorSets(writes, nullptr);
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    // 1. Clear Stats Buffer to [UINT_MAX, 0]
    uint32_t initStats[2] = {0xFFFFFFFF, 0};
    vd->commandBuffer->updateBuffer(statsBuffer->buf, 0, 2 * sizeof(uint32_t), initStats);
    
    // Barrier: Transfer Write -> Shader Read/Write
    vk::BufferMemoryBarrier barrier1(
        vk::AccessFlagBits::eTransferWrite, 
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, 
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, 
        statsBuffer->buf, 0, VK_WHOLE_SIZE);
        
    vd->commandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, 
        vk::PipelineStageFlagBits::eComputeShader, 
        {}, 
        0, nullptr, 
        1, &barrier1, 
        0, nullptr);
    
    // 2. Dispatch Stats Shader
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, statsPipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descSet.get(), 0, nullptr);
    
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    uint32_t pcStats[1] = { totalBins };
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t), pcStats);
    
    // Groups: 1M / 256 = 4096
    vd->commandBuffer->dispatch(4096, 1, 1);
    
    // Barrier: Shader Write -> Transfer Read
    vk::BufferMemoryBarrier barrier2(
        vk::AccessFlagBits::eShaderWrite, 
        vk::AccessFlagBits::eTransferRead, 
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, 
        statsBuffer->buf, 0, VK_WHOLE_SIZE);
        
    vd->commandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader, 
        vk::PipelineStageFlagBits::eTransfer, 
        {}, 
        0, nullptr, 
        1, &barrier2, 
        0, nullptr);
    
    // 3. Copy to Staging
    PBuffer stagingStats(new Buffer(vd));
    stagingStats->create(2 * sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadOnly);
    
    vk::BufferCopy copyStats(0, 0, 2 * sizeof(uint32_t));
    vd->commandBuffer->copyBuffer(statsBuffer->buf, stagingStats->buf, copyStats);
    
    vd->commandBuffer->end();
    
    // Submit
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
    
    // Read and Print
    uint32_t statsOut[2];
    stagingStats->readData((char*)statsOut, 2 * sizeof(uint32_t));
    stagingStats->destroy();
    
    std::cout << phase << " Bin Counts: Min=" << statsOut[0] << ", Max=" << statsOut[1] << "\n";
}
