#include "CompactScanIndex.hpp"
#include <common/utils.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>

#ifndef VERBOSE_COMPACT
#define VERBOSE_COMPACT 1
#endif

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
    size_t scanBufSize = scan->getBufSizeDivisor();
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

    // Data Buffer - allocate with SCALE_FACTOR extra space for updates
    this->totalAllocatedCapacity = npoints * COMPACT_INITIAL_SCALE_FACTOR;
    this->globalFreeOffset = npoints;
    
    dataBuffer = std::make_shared<Buffer>(vd);
    dataBuffer->create(totalAllocatedCapacity * sizeof(CompactEntry), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress, 
        MemoryType::Internal);
        
    // Stats Buffer
    statsBuffer = std::make_shared<Buffer>(vd);
    statsBuffer->create(2 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
        
    // Capacity Buffer - max capacity per bin
    capacityBuffer = std::make_shared<Buffer>(vd);
    capacityBuffer->create(totalBins * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
        MemoryType::Internal);
    
    // Extent Buffer - highest index written per bin (used by query shader)
    // On build: extent[bin] = original_count
    // On insert: extent[bin] = max(extent[bin], new_offset + 1)
    // On delete: unchanged (entries not shifted)
    extentBuffer = std::make_shared<Buffer>(vd);
    extentBuffer->create(totalBins * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
        MemoryType::Internal);
    
    // Index Map Buffer - maps pointIndex → globalDataIndex for O(1) delete
    // Only allocated if useIndexedDelete is enabled (via -s flag)
    if (useIndexedDelete) {
        indexMapBuffer = std::make_shared<Buffer>(vd);
        indexMapBuffer->create(npoints * sizeof(uint32_t), 
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
            MemoryType::Internal);
    }
    
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
    bindings.push_back(vk::DescriptorSetLayoutBinding(7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)); // Extent
    bindings.push_back(vk::DescriptorSetLayoutBinding(8, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)); // IndexMap
    
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
        // Push constants: minVal[2], binRange[2], res (5 uints) - like RasterScan2D
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
        
        // Bindings: count buffer (0), startAddr buffer (1) - data buffer uses buffer device address
        bPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}
        };
        bPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 2}
        };
        // Push constants: minVal[2], binRange[2], res, pad, dataBufferAddr[2] (8 uints)
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

    // ========== GRAPHICS PIPELINE: Insert (incremental) ==========
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/compact_insert_gfx.vert.spv", vshader), "compact insert gfx vertex shader");
        vk::ShaderModuleCreateInfo createInfo({}, vshader.size() * sizeof(uint32_t), vshader.data());
        insVertexShader = vd->device->createShaderModuleUnique(createInfo);

        insPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, insVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, fragmentShader.get(), "main")
        };
        insPipelineProps.setShaderStageFlag();

        // Row-major vertex input: one binding with stride = 3 uints (x,y,z)
        insPipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, 3 * sizeof(uint32_t))
        };
        insPipelineProps.setInputBindingFlag();

        insPipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32Uint, sizeof(uint32_t)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32Uint, 2 * sizeof(uint32_t))
        };
        insPipelineProps.setInputAttrFlag();

        insPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        insPipelineProps.setInputAssemblyFlag();

        // Bindings: startAddr(0), extent(1), capacity(2) - data buffer uses buffer device address
        insPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}
        };
        insPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 3}
        };
        // Push constants: minVal[2], binWidth[2], res, scaleFactor, dataBufferAddr[2] (8 uints)
        insPipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, 8 * sizeof(uint32_t))
        };
        insPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

        vk::PipelineRenderingCreateInfo rpCreateInfo;
        rpCreateInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpCreateInfo.pColorAttachmentFormats = &colorFormat;

        vk::UniqueRenderPass dummyRenderPass;
        insPipeline = insPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
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
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_range.frag.spv", fragCode)) {
            throw std::runtime_error("Failed to load compact_range.frag.spv");
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
        
        // Bindings for compact_range.frag (Pass 1 - collect ranges):
        // 0: startAddr, 1: extent, 2: resct (maxBuffer), 3: result (edgeBuffer)
        queryGfxPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}
        };
        queryGfxPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 4}
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
        
        // Input: [st, en) pairs from edgeBuffer (uvec2 per vertex)
        edgePipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, 2 * sizeof(uint32_t))
        };
        edgePipelineProps.setInputBindingFlag();
        
        edgePipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Uint, 0)  // erange (uvec2)
        };
        edgePipelineProps.setInputAttrFlag();
        
        edgePipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        edgePipelineProps.setInputAssemblyFlag();
        
        // Bindings: result, query (data buffer now uses buffer device address)
        edgePipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}
        };
        edgePipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 2}
        };
        // Push constants: res, ncols, dataBufferAddr[2] (4 uints total)
        edgePipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, 4 * sizeof(uint32_t))
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
    
    // Delete Pipeline (linear scan)
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
    
    // Delete Indexed Pipeline (O(1) using index map)
    {
        std::vector<uint32_t> code;
        if(!vkcore::readShader(SHADER_FOLDER + "/compact_delete_indexed.comp.spv", code)) {
             throw std::runtime_error("Failed to load compact_delete_indexed.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        deleteIndexedShader = vd->device->createShaderModuleUnique(createInfo);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, deleteIndexedShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout.get());
        deleteIndexedPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
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
    dummyFbo->create(vk::Format::eR8Sint, INDEX_RESOLUTION, INDEX_RESOLUTION, 1, MemoryType::Internal);
}

void CompactScanIndex::buildIndex(vkcore::PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal) {

#if VERBOSE_COMPACT
    std::cout << "[CompactScan] Starting buildIndex..." << std::endl;
    auto buildStart = std::chrono::high_resolution_clock::now();
#endif

    #ifdef VERBOSE_COMPACT
        auto initTimeStart = std::chrono::high_resolution_clock::now();
    #endif

    // Allocate buffers (Count and StartAddr)
    allocateBuffers(npoints);
    initialize();

    #ifdef VERBOSE_COMPACT
        auto initTimeEnd = std::chrono::high_resolution_clock::now();
        double initTime = std::chrono::duration<double, std::milli>(initTimeEnd - initTimeStart).count();
        std::cerr << "[CompactScan] Index initialization time: " << initTime << " ms\n";
    #endif


    
    // Store min/max values
    for(int i=0; i<3; i++) {
        this->minVal[i] = minVal[i];
        this->maxVal[i] = maxVal[i];
    }
    
    // Calculate binRange like RasterScan2D
    uint32_t binRange0 = uint32_t(ceil(double(maxVal[0] - minVal[0]) / INDEX_RESOLUTION));
    uint32_t binRange1 = uint32_t(ceil(double(maxVal[1] - minVal[1]) / INDEX_RESOLUTION));
    uint32_t binRange2 = uint32_t(ceil(double(maxVal[2] - minVal[2]) / INDEX_RESOLUTION));
    if(binRange0 == 0) binRange0 = 1;
    if(binRange1 == 0) binRange1 = 1;
    if(binRange2 == 0) binRange2 = 1;
    binWidth[0] = binRange0;
    binWidth[1] = binRange1;
    binWidth[2] = binRange2;
    
    // Push constants: minVal[2], binRange[2], res (5 uints) - like RasterScan2D
    std::array<uint32_t, 5> gfxPC = {minVal[0], minVal[1], binRange0, binRange1, INDEX_RESOLUTION};
    
    // Total bins for capacity buffer operations
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    
    // Single command buffer submission
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    // Clear extentBuffer (will be used for count pass directly)
    // This eliminates one copy: count pass writes to extentBuffer, which IS the final extent
    extentBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader);
    
    // NOTE: Data buffer clear removed - not needed because:
    // 1. Query shader only reads entries up to extent[bin] (actual count)
    // 2. Insert shader writes valid data to all used positions
    // 3. Clearing large buffers (npoints * SCALE_FACTOR * 16 bytes) is very slow
    
    // ========== PASS 1: Count Points per Bin (Graphics Pipeline) ==========
#if VERBOSE_COMPACT
    std::cout << "[CompactScan] Step 1: Counting points per bin..." << std::endl;
    auto countStart = std::chrono::high_resolution_clock::now();
#endif
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, bcPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        
        // Update descriptor set - write directly to extentBuffer (eliminates copy)
        vk::DescriptorBufferInfo extentDescriptor{extentBuffer->buf, 0, VK_WHOLE_SIZE};
        std::vector<vk::WriteDescriptorSet> descriptorSets = {
            vk::WriteDescriptorSet{bcPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &extentDescriptor}
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
    
#if VERBOSE_COMPACT
    auto countEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> countDuration = countEnd - countStart;
    std::cout << "[CompactScan] Point counting time: " << countDuration.count() << " ms" << std::endl;
    
    std::cout << "[CompactScan] Step 1.5: Post-counting barriers and copies..." << std::endl;
    auto postCountStart = std::chrono::high_resolution_clock::now();
#endif

    // Barrier: Vertex shader write -> Transfer/Compute
    // extentBuffer now contains original counts (no copy needed!)
    extentBuffer->barrier(vk::PipelineStageFlagBits::eVertexShader, vk::PipelineStageFlagBits::eTransfer,
                         vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead);
    
    // ========== PASS 1.5: Copy extentBuffer to capacityBuffer ==========
    // capacityBuffer stores original counts for capacity tracking during updates
    capacityBuffer->copyFrom(totalBins * sizeof(uint32_t), 0, 0, extentBuffer);
    
#if COMPACT_INITIAL_SCALE_FACTOR > 1
    // When scaling: copy to countBuffer, scale in-place, then prefix sum
    countBuffer->copyFrom(totalBins * sizeof(uint32_t), 0, 0, extentBuffer);
    
    // Barrier: Transfer write -> Compute shader
    capacityBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
    countBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                         vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
#else
    // When SCALE_FACTOR=1: copy directly to startAddrBuffer for prefix sum (skip countBuffer entirely)
    // First clear startAddrBuffer to ensure padding bytes are zero (needed for prefix sum)
    startAddrBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eTransfer);
    // Note: only copy totalBins (extentBuffer size), not countBufSize (which is padded for prefix sum)
    startAddrBuffer->copyFrom(totalBins * sizeof(uint32_t), 0, 0, extentBuffer);
    
    // Barrier: Transfer write -> Compute shader
    capacityBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
    startAddrBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                             vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
#endif

#if VERBOSE_COMPACT
    auto postCountEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> postCountDuration = postCountEnd - postCountStart;
    std::cout << "[CompactScan] Post-counting barriers and copies time: " << postCountDuration.count() << " ms" << std::endl;
#endif
    
    // Scale counts by INITIAL_SCALE_FACTOR to reserve extra space per bin
    // Skip scale shader entirely when SCALE_FACTOR=1 (no scaling needed)
#if COMPACT_INITIAL_SCALE_FACTOR > 1
#if VERBOSE_COMPACT
    std::cout << "[CompactScan] Step 1.6: Scaling counts..." << std::endl;
    auto scaleStart = std::chrono::high_resolution_clock::now();
#endif
    {
        // Bind countBuffer to descSet binding 1 for scale shader
        vk::DescriptorBufferInfo countInfo(countBuffer->buf, 0, VK_WHOLE_SIZE);
        vk::WriteDescriptorSet write(descSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countInfo);
        vd->device->updateDescriptorSets({write}, nullptr);
        
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, scalePipeline.get());
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descSet.get(), 0, nullptr);
        
        // Push constants: totalBins, scaleFactor (as integer, e.g., 2 for 2x)
        uint32_t scaleFactorInt = (uint32_t)(COMPACT_INITIAL_SCALE_FACTOR + 0.5); // Round to nearest int
        if (scaleFactorInt < 1) scaleFactorInt = 1;
        uint32_t pcScale[2] = { totalBins, scaleFactorInt };
        vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, 2 * sizeof(uint32_t), pcScale);
        
        uint32_t groups = (totalBins + 255) / 256;
        vd->commandBuffer->dispatch(groups, 1, 1);
    }
    
#if VERBOSE_COMPACT
    auto scaleEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> scaleDuration = scaleEnd - scaleStart;
    std::cout << "[CompactScan] Scale counts time: " << scaleDuration.count() << " ms" << std::endl;
#endif
#endif // COMPACT_INITIAL_SCALE_FACTOR > 1

    // capacityBuffer now contains original counts (before scaling)
    // The insert shader will compute actual capacity as capacity[bin] * scaleFactor
    
#if VERBOSE_COMPACT
    std::cout << "[CompactScan] Step 1.7: Pre-prefix sum barriers..." << std::endl;
    auto prePrefixStart = std::chrono::high_resolution_clock::now();
#endif

    // Barrier: Previous stage -> Prefix sum read
#if COMPACT_INITIAL_SCALE_FACTOR > 1
    // Scale shader write -> Prefix sum read
    countBuffer->barrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                         vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
#else
    // Transfer (copy) -> Prefix sum read (no scale shader ran)
    countBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                         vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
#endif
    capacityBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);

#if VERBOSE_COMPACT
    auto prePrefixEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> prePrefixDuration = prePrefixEnd - prePrefixStart;
    std::cout << "[CompactScan] Pre-prefix sum barriers time: " << prePrefixDuration.count() << " ms" << std::endl;
#endif
    
    // ========== PASS 2: GPU Prefix Sum ==========
#if VERBOSE_COMPACT
    std::cout << "[CompactScan] Step 2: Computing prefix sum..." << std::endl;
    auto prefixStart = std::chrono::high_resolution_clock::now();
#endif
    if(scan) {
#if COMPACT_INITIAL_SCALE_FACTOR > 1
        // Prefix sum on scaled countBuffer, then copy to startAddrBuffer
        scan->prefixSum(countBuffer->buf, countBufSize);
        
        // Barrier: Prefix sum write -> Transfer
        countBuffer->barrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                             vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead);
        
        // Copy prefix sum to startAddrBuffer
        startAddrBuffer->copyFrom(countBufSize * sizeof(uint32_t), 0, 0, countBuffer);
#else
        // SCALE_FACTOR=1: prefix sum directly on startAddrBuffer (no extra copy!)
        scan->prefixSum(startAddrBuffer->buf, countBufSize);
#endif
    } else {
        std::cerr << "[CompactIndex] ERROR: No SinglePassScan provided!\n";
        vd->commandBuffer->end();
        return;
    }
    
#if VERBOSE_COMPACT
    auto prefixEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> prefixDuration = prefixEnd - prefixStart;
    std::cout << "[CompactScan] Prefix sum time: " << prefixDuration.count() << " ms" << std::endl;
    
    std::cout << "[CompactScan] Step 2.5: Post-prefix sum barriers and setup..." << std::endl;
    auto postPrefixStart = std::chrono::high_resolution_clock::now();
#endif

    // Barrier: startAddrBuffer ready for build pass
#if COMPACT_INITIAL_SCALE_FACTOR > 1
    startAddrBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eVertexShader,
                             vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
#else
    // SCALE_FACTOR=1: prefix sum ran directly on startAddrBuffer
    startAddrBuffer->barrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eVertexShader,
                             vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
#endif
    
    // Clear countBuffer to 0 before build pass (so atomicAdd returns local positions starting from 0)
    countBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader, 0);

#if VERBOSE_COMPACT
    auto postPrefixEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> postPrefixDuration = postPrefixEnd - postPrefixStart;
    std::cout << "[CompactScan] Post-prefix sum barriers and setup time: " << postPrefixDuration.count() << " ms" << std::endl;
#endif
    
    // ========== PASS 3: Insert Points (Graphics Pipeline) ==========
#if VERBOSE_COMPACT
    std::cout << "[CompactScan] Step 3: Inserting points into bins..." << std::endl;
    auto insertStart = std::chrono::high_resolution_clock::now();
#endif
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, bPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        
        // Update descriptor sets - binding 0: countBuffer, binding 1: startAddrBuffer (data buffer uses buffer device address)
        vk::DescriptorBufferInfo countDesc{countBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo startDesc{startAddrBuffer->buf, 0, VK_WHOLE_SIZE};
        std::vector<vk::WriteDescriptorSet> descriptorSets = {
            vk::WriteDescriptorSet{bPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDesc},
            vk::WriteDescriptorSet{bPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &startDesc}
        };
        vd->device->updateDescriptorSets(descriptorSets, nullptr);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bPipelineProps.pipelineLayout.get(), 0, bPipelineProps.descriptorSet.get(), nullptr);
        
        // Push constants: minVal[2], binRange[2], res, pad, dataBufferAddr[2] (8 uints)
        uint64_t dataAddr = dataBuffer->getDeviceAddress();
        std::array<uint32_t, 8> gfxPCWithAddr = {gfxPC[0], gfxPC[1], gfxPC[2], gfxPC[3], gfxPC[4], 0, 
                                                  static_cast<uint32_t>(dataAddr & 0xFFFFFFFF), 
                                                  static_cast<uint32_t>(dataAddr >> 32)};
        vd->commandBuffer->pushConstants<uint32_t>(bPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, gfxPCWithAddr);
        
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
    
#if VERBOSE_COMPACT
    auto insertEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> insertDuration = insertEnd - insertStart;
    std::cout << "[CompactScan] Point insertion time: " << insertDuration.count() << " ms" << std::endl;
    
    std::cout << "[CompactScan] Step 3.5: Post-insert barriers and cleanup..." << std::endl;
    auto postInsertStart = std::chrono::high_resolution_clock::now();
#endif

    // Barrier: Insert pass write -> Transfer (for copying capacity back to count)
    countBuffer->barrier(vk::PipelineStageFlagBits::eVertexShader, vk::PipelineStageFlagBits::eTransfer,
                         vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferWrite);
    capacityBuffer->barrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                            vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead);
    
    // Reset count buffer to original counts (from capacity buffer) for subsequent inserts
    // After build, count[bin] = startAddr[bin] + original_count, which is wrong for inserts
    // We need count[bin] = original_count so that atomicAdd returns the correct offset
    countBuffer->copyFrom(totalBins * sizeof(uint32_t), 0, 0, capacityBuffer);

#if VERBOSE_COMPACT
    auto postInsertEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> postInsertDuration = postInsertEnd - postInsertStart;
    std::cout << "[CompactScan] Post-insert barriers and cleanup time: " << postInsertDuration.count() << " ms" << std::endl;
#endif
    
    vd->commandBuffer->end();
    
#if VERBOSE_COMPACT
    std::cout << "[CompactScan] Step 4: Command buffer submission and GPU execution..." << std::endl;
    auto submitStart = std::chrono::high_resolution_clock::now();
#endif

    // Single fence wait for entire build
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);

#if VERBOSE_COMPACT
    auto submitEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> submitDuration = submitEnd - submitStart;
    std::cout << "[CompactScan] GPU execution and fence wait time: " << submitDuration.count() << " ms" << std::endl;
#endif
    
    // Build index map if enabled (for O(1) delete)
    // Read back data buffer and create pointIndex → globalDataIndex mapping
    if (useIndexedDelete && indexMapBuffer) {
        // Create staging buffer large enough for full data buffer readback
        // Note: Data buffer has totalAllocatedCapacity entries (npoints * SCALE_FACTOR)
        // but only npoints entries are valid after build
        vkcore::PBuffer stagingBuf = std::make_shared<Buffer>(vd);
        size_t dataSize = totalAllocatedCapacity * sizeof(CompactEntry);
        stagingBuf->create(dataSize, 
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
            MemoryType::ReadOnly);
        
        // Read full data buffer to CPU
        std::vector<CompactEntry> entries(totalAllocatedCapacity);
        readUsingStagingBuf((char*)entries.data(), dataSize, dataBuffer, stagingBuf, vd);
        
        // Build index map on CPU
        std::vector<uint32_t> indexMapData(npoints, 0xFFFFFFFF);  // Initialize to invalid
        uint32_t validCount = 0;
        
        for (uint64_t i = 0; i < totalAllocatedCapacity; i++) {
            uint32_t rowId = entries[i].rowId;
            if (rowId & 0x80000000u) {  // Valid entry
                uint32_t pointIndex = rowId & 0x7FFFFFFFu;  // Extract original point index
                if (pointIndex < npoints) {
                    indexMapData[pointIndex] = (uint32_t)i;  // Map pointIndex → globalDataIndex
                    validCount++;
                }
            }
        }
        
        // Upload index map to GPU
        loadUsingStagingBuf((char*)indexMapData.data(), npoints * sizeof(uint32_t), 
                           indexMapBuffer, stagingBuf, vd, 0);
        
        stagingBuf->destroy();
        
        std::cerr << "[CompactIndex] Index map built for O(1) delete (" << validCount << " entries mapped)\n";
    }
    
#if VERBOSE_COMPACT
    auto buildEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> buildDuration = buildEnd - buildStart;
    std::cout << "[CompactScan] Total buildIndex time (START-END): " << buildDuration.count() << " ms" << std::endl;
    std::cout << "[CompactScan] Total buildIndex time (PARTS): " << initTime + countDuration.count() + postInsertDuration.count() + prePrefixDuration.count() + prefixDuration.count() + submitDuration.count() << " ms" << std::endl;
    
#endif
}

void CompactScanIndex::runRangeQueries(vkcore::PBuffer queryBuffer, uint32_t nqueries, vkcore::PBuffer resultBuffer) {
#if VERBOSE_COMPACT
    std::cout << "[CompactScan] Starting runRangeQueries for " << nqueries << " queries..." << std::endl;
    auto queryStart = std::chrono::high_resolution_clock::now();
#endif

    // Two-pass graphics query like RasterScan2D
    // Pass 1 (Range): Collect [st, en) pairs for bins in query range
    // Pass 2 (Edge): One fragment per entry, check validity and range
    
    // Initialize query descriptors once (Pass 1 descriptors never change)
#if DISABLE_QUERY_CACHING
    if (true) {  // Caching disabled - always update descriptors
#else
    if (!queryDescriptorsInitialized) {
#endif
        // Pass 1 bindings: 0=startAddr, 1=extent, 2=resct(maxBuffer), 3=result(edgeBuffer)
        vk::DescriptorBufferInfo startInfo(startAddrBuffer->buf, 0, VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo extInfo(extentBuffer->buf, 0, VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo resctInfo(maxBuffer->buf, 0, VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo edgeInfo(edgeBuffer->buf, 0, VK_WHOLE_SIZE);
        
        std::vector<vk::WriteDescriptorSet> writes = {
            vk::WriteDescriptorSet{queryGfxPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &startInfo},
            vk::WriteDescriptorSet{queryGfxPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &extInfo},
            vk::WriteDescriptorSet{queryGfxPipelineProps.descriptorSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &resctInfo},
            vk::WriteDescriptorSet{queryGfxPipelineProps.descriptorSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &edgeInfo}
        };
        vd->device->updateDescriptorSets(writes, nullptr);
        
        // Data buffer now uses buffer device address - no descriptor binding needed
        
        // Create reusable fence
        queryFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
        
        queryDescriptorsInitialized = true;
    }
    
    // Only update Pass 2 bindings that change (result buffer and query buffer)
    // Binding 0 = result buffer, Binding 1 = query buffer (data buffer uses buffer device address)
#if DISABLE_QUERY_CACHING
    if (true) {  // Caching disabled
#else
    if (lastResultBuffer != resultBuffer) {
#endif
        vk::DescriptorBufferInfo resInfo(resultBuffer->buf, 0, VK_WHOLE_SIZE);
        vk::WriteDescriptorSet resWrite{edgePipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &resInfo};
        vd->device->updateDescriptorSets({resWrite}, nullptr);
        lastResultBuffer = resultBuffer;
    }
    
    // Always update query buffer binding (changes per query)
    vk::DescriptorBufferInfo qInfo(queryBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::WriteDescriptorSet qWrite{edgePipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &qInfo};
    vd->device->updateDescriptorSets({qWrite}, nullptr);
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    // Clear result buffer (included in timing like RasterScan2D)
    resultBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eFragmentShader, 0);
    
    // Clear maxBuffer: [0]=numRanges for drawIndirect
    maxBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eFragmentShader, 0);
    
    // ========== PASS 1: Range - collect [st, en) pairs ==========
#if VERBOSE_COMPACT
    std::cout << "[CompactScan] Query Pass 1: Range collection..." << std::endl;
    auto rangeStart = std::chrono::high_resolution_clock::now();
#endif
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
        
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, queryGfxPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, queryGfxPipelineProps.pipelineLayout.get(), 0, queryGfxPipelineProps.descriptorSet.get(), nullptr);
        
        // Push constants: minVal[3], resolution, binWidth[3], nqueries
        uint32_t pc[8] = { minVal[0], minVal[1], minVal[2], INDEX_RESOLUTION, binWidth[0], binWidth[1], binWidth[2], nqueries };
        vd->commandBuffer->pushConstants<uint32_t>(queryGfxPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        
        vk::DeviceSize offset = 0;
        vd->commandBuffer->bindVertexBuffers(0, queryBuffer->buf, offset);
        vd->commandBuffer->draw(nqueries, 1, 0, 0);
        vd->commandBuffer->endRendering();
    }
    
#if VERBOSE_COMPACT
    auto rangeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> rangeDuration = rangeEnd - rangeStart;
    std::cout << "[CompactScan] Range collection time: " << rangeDuration.count() << " ms" << std::endl;
#endif

    // Barrier: Pass 1 write -> Pass 2 read
    maxBuffer->barrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eDrawIndirect,
                       vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eIndirectCommandRead);
    edgeBuffer->barrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eVertexInput,
                        vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eVertexAttributeRead);
    
    // ========== PASS 2: Edge - one fragment per entry ==========
#if VERBOSE_COMPACT
    std::cout << "[CompactScan] Query Pass 2: Edge processing..." << std::endl;
    auto edgeStart = std::chrono::high_resolution_clock::now();
#endif
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
        
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, edgePipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, edgePipelineProps.pipelineLayout.get(), 0, edgePipelineProps.descriptorSet.get(), nullptr);
        
        // Push constants: res, ncols, dataBufferAddr[2] (4 uints total)
        uint64_t dataAddr = dataBuffer->getDeviceAddress();
        uint32_t pc2[4] = { INDEX_RESOLUTION, 3, static_cast<uint32_t>(dataAddr & 0xFFFFFFFF), static_cast<uint32_t>(dataAddr >> 32) };
        vd->commandBuffer->pushConstants<uint32_t>(edgePipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc2);
        
        vk::DeviceSize offset = 0;
        vd->commandBuffer->bindVertexBuffers(0, edgeBuffer->buf, offset);
        
        // Indirect draw: numRanges vertices from edgeBuffer
        vd->commandBuffer->drawIndirect(maxBuffer->buf, 0, 1, 4 * sizeof(uint32_t));
        vd->commandBuffer->endRendering();
    }
    
#if VERBOSE_COMPACT
    auto edgeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> edgeDuration = edgeEnd - edgeStart;
    std::cout << "[CompactScan] Edge processing time: " << edgeDuration.count() << " ms" << std::endl;
#endif

    vd->commandBuffer->end();
    
    // Reuse fence (reset before use)
    vd->device->resetFences({queryFence.get()});
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vd->submit(submitInfo, queryFence.get(), false);
    vd->waitForFences(queryFence.get(), VK_TRUE, UINT64_MAX);

#if VERBOSE_COMPACT
    auto queryEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> queryDuration = queryEnd - queryStart;
    std::cout << "[CompactScan] Total runRangeQueries time: " << queryDuration.count() << " ms" << std::endl;
#endif
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
    
    // Binding 2: Input data to delete (data buffer now uses buffer device address)
    vk::DescriptorBufferInfo dInfo(deleteDataBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dInfo));
    
    // Binding 3: Extent (not modified on delete - entries not shifted)
    vk::DescriptorBufferInfo extInfo(extentBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &extInfo));
    
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
    
    // Push constants: minVal[3], resolution, maxVal[3], ndeletes, binWidth[3], pad, dataBufferAddr[2] (14 uints)
    uint64_t dataAddr = dataBuffer->getDeviceAddress();
    uint32_t pc[14] = { minVal[0], minVal[1], minVal[2], INDEX_RESOLUTION, maxVal[0], maxVal[1], maxVal[2], ndeletes, binWidth[0], binWidth[1], binWidth[2], 0,
                        static_cast<uint32_t>(dataAddr & 0xFFFFFFFF), static_cast<uint32_t>(dataAddr >> 32) };
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, 14 * sizeof(uint32_t), pc);
    
    // One thread per delete request, linear scan within bin
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
    // computeAndPrintStats("[GPU Stats - After Delete]");
}

void CompactScanIndex::deletePointsIndexed(vkcore::PBuffer deleteIndicesBuffer, uint32_t ndeletes) {
    // O(1) delete using index map
    // deleteIndicesBuffer contains original point indices (0..npoints-1)
    
    if (!useIndexedDelete || !indexMapBuffer) {
        std::cerr << "[CompactIndex] ERROR: deletePointsIndexed called but index map not available!\n";
        std::cerr << "[CompactIndex] Use -s flag to enable indexed delete.\n";
        return;
    }
    
    // Update Descriptor Set
    // Note: dataBuffer is accessed via buffer device address in the shader to avoid >4GB descriptor limits.
    std::vector<vk::WriteDescriptorSet> writes;

    // Binding 3: Delete indices buffer
    vk::DescriptorBufferInfo dInfo(deleteIndicesBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dInfo));

    // Binding 8: Index map buffer
    vk::DescriptorBufferInfo mapInfo(indexMapBuffer->buf, 0, VK_WHOLE_SIZE);
    writes.push_back(vk::WriteDescriptorSet(descSet.get(), 8, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &mapInfo));

    vd->device->updateDescriptorSets(writes, nullptr);
    
    // Dispatch
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, deleteIndexedPipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descSet.get(), 0, nullptr);
    
    // Push constants: ndeletes, dataBufferAddr[2]
    uint64_t dataAddr = dataBuffer->getDeviceAddress();
    uint32_t pc[3] = {
        ndeletes,
        static_cast<uint32_t>(dataAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(dataAddr >> 32)
    };
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, 3 * sizeof(uint32_t), pc);
    
    // One thread per delete request - O(1) lookup
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
}

void CompactScanIndex::insertPoints(vkcore::PBuffer pointsBuffer, uint32_t npoints) {
    // Graphics-based insert avoids >4GB maxStorageBufferRange on dataBuffer.
    // It writes to dataBuffer via buffer device address in the vertex shader.
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);

    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, insPipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo startDesc{startAddrBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo extDesc{extentBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo capDesc{capacityBuffer->buf, 0, VK_WHOLE_SIZE};
    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{insPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &startDesc},
        vk::WriteDescriptorSet{insPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &extDesc},
        vk::WriteDescriptorSet{insPipelineProps.descriptorSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &capDesc}
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, insPipelineProps.pipelineLayout.get(), 0, insPipelineProps.descriptorSet.get(), nullptr);

    uint64_t dataAddr = dataBuffer->getDeviceAddress();
    uint32_t scaleFactorInt = (uint32_t)(COMPACT_INITIAL_SCALE_FACTOR + 0.5);
    if (scaleFactorInt < 1) scaleFactorInt = 1;
    std::array<uint32_t, 8> pc = {
        minVal[0], minVal[1],
        binWidth[0], binWidth[1],
        INDEX_RESOLUTION,
        scaleFactorInt,
        static_cast<uint32_t>(dataAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(dataAddr >> 32)
    };
    vd->commandBuffer->pushConstants<uint32_t>(insPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, pc);

    vd->commandBuffer->bindVertexBuffers(0, pointsBuffer->buf, vk::DeviceSize(0));
    vd->commandBuffer->draw(npoints, 1, 0, 0);
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
