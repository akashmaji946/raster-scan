#include "EquiDepthIndex.hpp"
#include "modes/ModeUtils.hpp"
#include <chrono>
#include <iostream>

using namespace vkcore;

// Helper to setup rendering (same as CompactScanIndex)
static vk::RenderingInfo setupRendering(PVkDevice vd, PFrameBuffer fbo, vk::RenderingAttachmentInfo& colorInfo) {
    colorInfo.imageView = fbo->colorView;
    colorInfo.imageLayout = vk::ImageLayout::eGeneral;
    colorInfo.loadOp = vk::AttachmentLoadOp::eDontCare;
    
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

EquiDepthIndex::EquiDepthIndex(PVkDevice vd, int32_t ncols, SinglePassScan* scan)
    : vd(vd), ncols(ncols), scan(scan), npoints(0), totalAllocatedCapacity(0) {
    for (int i = 0; i < 3; i++) {
        minVal[i] = 0;
        maxVal[i] = 0;
    }
}

EquiDepthIndex::~EquiDepthIndex() {
    if (quantileXBuffer) quantileXBuffer->destroy();
    if (quantileYBuffer) quantileYBuffer->destroy();
    if (histXBuffer) histXBuffer->destroy();
    if (histYBuffer) histYBuffer->destroy();
#if USE_MORTON_BINNING
    if (mortonHistBuffer) mortonHistBuffer->destroy();
    if (mortonQuantileBuffer) mortonQuantileBuffer->destroy();
#endif
    if (startAddrBuffer) startAddrBuffer->destroy();
    if (countBuffer) countBuffer->destroy();
    if (extentBuffer) extentBuffer->destroy();
    if (dataBuffer) dataBuffer->destroy();
    if (maxBuffer) maxBuffer->destroy();
    if (edgeBuffer) edgeBuffer->destroy();
    if (dummyFbo) dummyFbo->destroy();
}

void EquiDepthIndex::allocateBuffers(uint32_t npoints) {
    this->npoints = npoints;
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    
    // Count buffer size must be aligned for prefix sum
    uint32_t divisor = scan ? scan->getBufSizeDivisor() : 4096;
    
#if USE_MORTON_BINNING
    // For round-robin binning, we only need a small quantile buffer for descriptor binding
    // (the shader doesn't actually use it, but Vulkan requires valid bindings)
    mortonQuantileBuffer = std::make_shared<Buffer>(vd);
    mortonQuantileBuffer->create((totalBins + 1) * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    // Skip allocating histogram buffers - not needed for round-robin
#else
    // Quantile boundaries (INDEX_RESOLUTION + 1 values each)
    uint32_t quantileSize = (INDEX_RESOLUTION + 1) * sizeof(uint32_t);
    quantileXBuffer = std::make_shared<Buffer>(vd);
    quantileXBuffer->create(quantileSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    quantileYBuffer = std::make_shared<Buffer>(vd);
    quantileYBuffer->create(quantileSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    // Coarse histograms - padded for prefix sum
    uint32_t paddedHistSize = ((EQUIDEPTH_COARSE_BINS + divisor - 1) / divisor) * divisor;
    histXBuffer = std::make_shared<Buffer>(vd);
    histXBuffer->create(paddedHistSize * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    histYBuffer = std::make_shared<Buffer>(vd);
    histYBuffer->create(paddedHistSize * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
#endif
    
    countBufSize = ((totalBins + divisor - 1) / divisor) * divisor;
    
    startAddrBuffer = std::make_shared<Buffer>(vd);
    startAddrBuffer->create(countBufSize * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    countBuffer = std::make_shared<Buffer>(vd);
    countBuffer->create(countBufSize * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    extentBuffer = std::make_shared<Buffer>(vd);
    extentBuffer->create(totalBins * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    // Data buffer
    totalAllocatedCapacity = npoints;
    dataBuffer = std::make_shared<Buffer>(vd);
    dataBuffer->create(totalAllocatedCapacity * sizeof(EquiDepthEntry),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | 
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        MemoryType::Internal);
    
    // Query buffers
    maxBuffer = std::make_shared<Buffer>(vd);
    maxBuffer->create(4 * sizeof(uint32_t),  // VkDrawIndirectCommand
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    // Edge buffer: max totalBins * 2 for [st, en) pairs
    edgeBuffer = std::make_shared<Buffer>(vd);
    edgeBuffer->create(totalBins * 2 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
        MemoryType::Internal);
    
    // Dummy FBO
    dummyFbo = std::make_shared<FrameBuffer>(vd);
    dummyFbo->create(vk::Format::eR8Sint, INDEX_RESOLUTION, INDEX_RESOLUTION, 1, MemoryType::Internal);
}

void EquiDepthIndex::initialize() {
    setupPipelines();
}

void EquiDepthIndex::setupPipelines() {
#if USE_MORTON_BINNING
    // Round-robin binning: Skip all compute pipelines - not needed
#else
    // ========== COMPUTE PIPELINE: Histogram ==========
    {
        std::vector<uint32_t> code;
        if (!readShader(SHADER_FOLDER + "/equidepth_histogram.comp.spv", code)) {
            throw std::runtime_error("Failed to load equidepth_histogram.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        histogramShader = vd->device->createShaderModuleUnique(createInfo);
        
        // Descriptor set layout: 4 storage buffers (pointsX, pointsY, histX, histY)
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
        histDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
        
        vk::PushConstantRange pcRange(vk::ShaderStageFlagBits::eCompute, 0, 6 * sizeof(uint32_t));
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &histDescSetLayout.get(), 1, &pcRange);
        histPipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
        
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eStorageBuffer, 4}
        };
        vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes);
        histDescPool = vd->device->createDescriptorPoolUnique(poolInfo);
        
        vk::DescriptorSetAllocateInfo allocInfo(histDescPool.get(), 1, &histDescSetLayout.get());
        histDescSet = std::move(vd->device->allocateDescriptorSetsUnique(allocInfo)[0]);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, histogramShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, histPipelineLayout.get());
        histogramPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // ========== COMPUTE PIPELINE: Quantiles ==========
    {
        std::vector<uint32_t> code;
        if (!readShader(SHADER_FOLDER + "/equidepth_quantiles.comp.spv", code)) {
            throw std::runtime_error("Failed to load equidepth_quantiles.comp.spv");
        }
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        quantilesShader = vd->device->createShaderModuleUnique(createInfo);
        
        // Descriptor set layout: 4 storage buffers (cdfX, cdfY, quantileX, quantileY)
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
        quantDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
        
        vk::PushConstantRange pcRange(vk::ShaderStageFlagBits::eCompute, 0, 7 * sizeof(uint32_t));
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &quantDescSetLayout.get(), 1, &pcRange);
        quantPipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
        
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eStorageBuffer, 4}
        };
        vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes);
        quantDescPool = vd->device->createDescriptorPoolUnique(poolInfo);
        
        vk::DescriptorSetAllocateInfo allocInfo(quantDescPool.get(), 1, &quantDescSetLayout.get());
        quantDescSet = std::move(vd->device->allocateDescriptorSetsUnique(allocInfo)[0]);
        
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, quantilesShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, quantPipelineLayout.get());
        quantilesPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
#endif
    
    // ========== Load dummy fragment shader ==========
    {
        std::vector<uint32_t> fshader;
        validate(readShader(SHADER_FOLDER + "/dummy.frag.spv", fshader), "dummy fragment shader");
        vk::ShaderModuleCreateInfo createInfo({}, fshader.size() * sizeof(uint32_t), fshader.data());
        dummyFragShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // ========== GRAPHICS PIPELINE: Count ==========
    {
        std::vector<uint32_t> vshader;
#if USE_MORTON_BINNING
        validate(readShader(SHADER_FOLDER + "/equidepth_morton_count.vert.spv", vshader), "equidepth morton count vertex shader");
#else
        validate(readShader(SHADER_FOLDER + "/equidepth_count.vert.spv", vshader), "equidepth count vertex shader");
#endif
        vk::ShaderModuleCreateInfo createInfo({}, vshader.size() * sizeof(uint32_t), vshader.data());
        countVertexShader = vd->device->createShaderModuleUnique(createInfo);
        
        countPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, countVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, dummyFragShader.get(), "main")
        };
        countPipelineProps.setShaderStageFlag();
        
        countPipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, sizeof(uint32_t)),
            vk::VertexInputBindingDescription(1, sizeof(uint32_t))
        };
        countPipelineProps.setInputBindingFlag();
        
        countPipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
            vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32Uint, 0)
        };
        countPipelineProps.setInputAttrFlag();
        
        countPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        countPipelineProps.setInputAssemblyFlag();
        
#if USE_MORTON_BINNING
        // Morton count: binding 0 = mortonQuantiles, binding 1 = countBuffer
        countPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}
        };
        countPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 2}
        };
        // Push constants: resolution, npoints, minX, maxX, minY, maxY
        countPipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, 6 * sizeof(uint32_t))
        };
#else
        countPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}
        };
        countPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 3}
        };
        countPipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, 2 * sizeof(uint32_t))
        };
#endif
        countPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);
        
        vk::PipelineRenderingCreateInfo rpCreateInfo;
        rpCreateInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpCreateInfo.pColorAttachmentFormats = &colorFormat;
        
        vk::UniqueRenderPass dummyRenderPass;
        countPipeline = countPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
    }
    
    // ========== GRAPHICS PIPELINE: Build ==========
    {
        std::vector<uint32_t> vshader;
#if USE_MORTON_BINNING
        validate(readShader(SHADER_FOLDER + "/equidepth_morton_build.vert.spv", vshader), "equidepth morton build vertex shader");
#else
        validate(readShader(SHADER_FOLDER + "/equidepth_build.vert.spv", vshader), "equidepth build vertex shader");
#endif
        vk::ShaderModuleCreateInfo createInfo({}, vshader.size() * sizeof(uint32_t), vshader.data());
        buildVertexShader = vd->device->createShaderModuleUnique(createInfo);
        
        buildPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, buildVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, dummyFragShader.get(), "main")
        };
        buildPipelineProps.setShaderStageFlag();
        
        buildPipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, sizeof(uint32_t)),
            vk::VertexInputBindingDescription(1, sizeof(uint32_t)),
            vk::VertexInputBindingDescription(2, sizeof(uint32_t))
        };
        buildPipelineProps.setInputBindingFlag();
        
        buildPipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
            vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32Uint, 0),
            vk::VertexInputAttributeDescription(2, 2, vk::Format::eR32Uint, 0)
        };
        buildPipelineProps.setInputAttrFlag();
        
        buildPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        buildPipelineProps.setInputAssemblyFlag();
        
#if USE_MORTON_BINNING
        // Morton build: binding 0 = mortonQuantiles, binding 1 = startAddr, binding 2 = countBuffer
        buildPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}
        };
        buildPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 3}
        };
        // Push constants: resolution, npoints, minX, maxX, minY, maxY, dataBufferAddr[2]
        buildPipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, 8 * sizeof(uint32_t))
        };
#else
        buildPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}
        };
        buildPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 4}
        };
        buildPipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, 6 * sizeof(uint32_t))
        };
#endif
        buildPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);
        
        vk::PipelineRenderingCreateInfo rpCreateInfo;
        rpCreateInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpCreateInfo.pColorAttachmentFormats = &colorFormat;
        
        vk::UniqueRenderPass dummyRenderPass;
        buildPipeline = buildPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
    }
    
    // ========== GRAPHICS PIPELINE: Query Pass 1 (Range) ==========
    {
        std::vector<uint32_t> vertCode, geomCode, fragCode;
#if USE_MORTON_BINNING
        validate(readShader(SHADER_FOLDER + "/equidepth_morton_query.vert.spv", vertCode), "equidepth morton query vertex shader");
        validate(readShader(SHADER_FOLDER + "/equidepth_query.geom.spv", geomCode), "equidepth query geom shader");
        validate(readShader(SHADER_FOLDER + "/equidepth_morton_range.frag.spv", fragCode), "equidepth morton range frag shader");
#else
        validate(readShader(SHADER_FOLDER + "/equidepth_query.vert.spv", vertCode), "equidepth query vertex shader");
        validate(readShader(SHADER_FOLDER + "/equidepth_query.geom.spv", geomCode), "equidepth query geom shader");
        validate(readShader(SHADER_FOLDER + "/equidepth_range.frag.spv", fragCode), "equidepth range frag shader");
#endif
        
        vk::ShaderModuleCreateInfo vertInfo({}, vertCode.size() * sizeof(uint32_t), vertCode.data());
        vk::ShaderModuleCreateInfo geomInfo({}, geomCode.size() * sizeof(uint32_t), geomCode.data());
        vk::ShaderModuleCreateInfo fragInfo({}, fragCode.size() * sizeof(uint32_t), fragCode.data());
        
        queryVertexShader = vd->device->createShaderModuleUnique(vertInfo);
        queryGeomShader = vd->device->createShaderModuleUnique(geomInfo);
        queryFragShader = vd->device->createShaderModuleUnique(fragInfo);
        
        queryPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, queryVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eGeometry, queryGeomShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, queryFragShader.get(), "main")
        };
        queryPipelineProps.setShaderStageFlag();
        
        queryPipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, 3 * sizeof(uint32_t)),
            vk::VertexInputBindingDescription(1, 3 * sizeof(uint32_t))
        };
        queryPipelineProps.setInputBindingFlag();
        
        queryPipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Uint, 0),
            vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32G32B32Uint, 0)
        };
        queryPipelineProps.setInputAttrFlag();
        
        queryPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        queryPipelineProps.setInputAssemblyFlag();
        
        // Two-pass query: Pass 1 outputs [st, en) pairs to edgeBuffer
        // Fragment shader: bindings 0,1 = startAddr, extent; 2 = maxBuffer; 3 = edgeBuffer
        queryPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}, // startAddr
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}, // extent
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}, // maxBuffer
            vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}  // edgeBuffer
        };
        queryPipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 4}
        };
        // Push constants: resolution, nqueries
        queryPipelineProps.pushConstantRange = {
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, 2 * sizeof(uint32_t))
        };
        queryPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);
        
        vk::PipelineRenderingCreateInfo rpCreateInfo;
        rpCreateInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpCreateInfo.pColorAttachmentFormats = &colorFormat;
        
        vk::UniqueRenderPass dummyRenderPass;
        queryPipeline = queryPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
    }
    
    // ========== GRAPHICS PIPELINE: Query Pass 2 (Edge) ==========
    {
        std::vector<uint32_t> vertCode, geomCode, fragCode;
        validate(readShader(SHADER_FOLDER + "/equidepth_edge.vert.spv", vertCode), "equidepth edge vertex shader");
        validate(readShader(SHADER_FOLDER + "/equidepth_edge.geom.spv", geomCode), "equidepth edge geom shader");
        validate(readShader(SHADER_FOLDER + "/equidepth_edge.frag.spv", fragCode), "equidepth edge frag shader");
        
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
        
        // Vertex input: [st, en) pairs from edgeBuffer (uvec2)
        edgePipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, 2 * sizeof(uint32_t))
        };
        edgePipelineProps.setInputBindingFlag();
        
        edgePipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Uint, 0)
        };
        edgePipelineProps.setInputAttrFlag();
        
        edgePipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        edgePipelineProps.setInputAssemblyFlag();
        
        // Bindings: 0=result buffer, 1=query buffer
        edgePipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}
        };
        edgePipelineProps.poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 2}
        };
        // Push constants: res, ncols, dataBufferAddr[2]
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
    
    // ========== COMPUTE PIPELINE: Range Collection (Pass 1 alternative) ==========
    // This compute shader is faster than the graphics pipeline when bins are sparse
    {
        std::vector<uint32_t> code;
        validate(readShader(SHADER_FOLDER + "/equidepth_range_compute.comp.spv", code), "equidepth range compute shader");
        
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        rangeComputeShader = vd->device->createShaderModuleUnique(createInfo);
        
        // Descriptor set layout: 0=startAddr, 1=extent, 2=maxBuffer, 3=edgeBuffer
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
        rangeComputeDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
        
        // Push constants: resolution, totalBins
        vk::PushConstantRange pcRange(vk::ShaderStageFlagBits::eCompute, 0, 2 * sizeof(uint32_t));
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &rangeComputeDescSetLayout.get(), 1, &pcRange);
        rangeComputePipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
        
        // Descriptor pool
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eStorageBuffer, 4}
        };
        vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes);
        rangeComputeDescPool = vd->device->createDescriptorPoolUnique(poolInfo);
        
        // Allocate descriptor set
        vk::DescriptorSetAllocateInfo allocInfo(rangeComputeDescPool.get(), 1, &rangeComputeDescSetLayout.get());
        rangeComputeDescSet = std::move(vd->device->allocateDescriptorSetsUnique(allocInfo)[0]);
        
        // Create compute pipeline
        vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, rangeComputeShader.get(), "main");
        vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, rangeComputePipelineLayout.get());
        rangeComputePipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;
    }
    
    // ========== COMPUTE PIPELINE: Pure Compute Query (Optimized) ==========
    // This replaces the slow graphics pipeline (vertex + geometry + fragment)
    // with a single compute dispatch - same approach as BruteForceIndex
    {
        std::vector<uint32_t> code;
        validate(readShader(SHADER_FOLDER + "/equidepth_query_compute.comp.spv", code), "equidepth query compute shader");
        
        vk::ShaderModuleCreateInfo createInfo({}, code.size() * sizeof(uint32_t), code.data());
        queryComputeShader = vd->device->createShaderModuleUnique(createInfo);
        
        // Descriptor set layout: 0=queryBuffer, 1=resultBuffer
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
        queryComputeDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
        
        // Push constants: npoints (4 bytes) + padding (4 bytes) + dataBufferAddr (8 bytes) = 16 bytes
        vk::PushConstantRange pcRange(vk::ShaderStageFlagBits::eCompute, 0, 16);
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &queryComputeDescSetLayout.get(), 1, &pcRange);
        queryComputePipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
        
        // Descriptor pool
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eStorageBuffer, 2}
        };
        vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes);
        queryComputeDescPool = vd->device->createDescriptorPoolUnique(poolInfo);
        
        // Allocate descriptor set
        vk::DescriptorSetAllocateInfo allocInfo(queryComputeDescPool.get(), 1, &queryComputeDescSetLayout.get());
        queryComputeDescSet = std::move(vd->device->allocateDescriptorSetsUnique(allocInfo)[0]);
        
        // Create compute pipeline
        vk::PipelineShaderStageCreateInfo qcStageInfo({}, vk::ShaderStageFlagBits::eCompute, queryComputeShader.get(), "main");
        vk::ComputePipelineCreateInfo qcPipelineInfo({}, qcStageInfo, queryComputePipelineLayout.get());
        queryComputePipeline = vd->device->createComputePipelineUnique(nullptr, qcPipelineInfo).value;
    }
}

void EquiDepthIndex::buildHistograms(PBuffer pointsBuffer, uint32_t npoints, bool bindDescriptors) {
    // Push constants
    uint32_t pc[6] = {npoints, EQUIDEPTH_COARSE_BINS, minVal[0], maxVal[0], minVal[1], maxVal[1]};
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, histogramPipeline.get());
    if (bindDescriptors) {
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, histPipelineLayout.get(), 0, histDescSet.get(), nullptr);
    }
    vd->commandBuffer->pushConstants(histPipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), pc);
    
    uint32_t groups = (npoints + 255) / 256;
    vd->commandBuffer->dispatch(groups, 1, 1);
}

void EquiDepthIndex::updateHistogramDescriptors(PBuffer pointsBuffer, uint32_t npoints) {
    vk::DescriptorBufferInfo pointsXInfo(pointsBuffer->buf, 0, npoints * sizeof(uint32_t));
    vk::DescriptorBufferInfo pointsYInfo(pointsBuffer->buf, npoints * sizeof(uint32_t), npoints * sizeof(uint32_t));
    vk::DescriptorBufferInfo histXInfo(histXBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo histYInfo(histYBuffer->buf, 0, VK_WHOLE_SIZE);
    
    std::vector<vk::WriteDescriptorSet> writes = {
        {histDescSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &pointsXInfo},
        {histDescSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &pointsYInfo},
        {histDescSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &histXInfo},
        {histDescSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &histYInfo},
    };
    vd->device->updateDescriptorSets(writes, nullptr);
}

void EquiDepthIndex::computeQuantiles() {
    // Push constants
    uint32_t pc[7] = {npoints, EQUIDEPTH_COARSE_BINS, INDEX_RESOLUTION, minVal[0], maxVal[0], minVal[1], maxVal[1]};
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, quantilesPipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, quantPipelineLayout.get(), 0, quantDescSet.get(), nullptr);
    vd->commandBuffer->pushConstants(quantPipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), pc);
    
    uint32_t groups = (INDEX_RESOLUTION + 1 + 255) / 256;
    vd->commandBuffer->dispatch(groups, 1, 1);
}

void EquiDepthIndex::updateQuantileDescriptors() {
    vk::DescriptorBufferInfo cdfXInfo(histXBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo cdfYInfo(histYBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo quantXInfo(quantileXBuffer->buf, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo quantYInfo(quantileYBuffer->buf, 0, VK_WHOLE_SIZE);
    
    std::vector<vk::WriteDescriptorSet> writes = {
        {quantDescSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cdfXInfo},
        {quantDescSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cdfYInfo},
        {quantDescSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &quantXInfo},
        {quantDescSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &quantYInfo},
    };
    vd->device->updateDescriptorSets(writes, nullptr);
}

void EquiDepthIndex::buildIndex(PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal) {
    std::cerr << "[EquiDepthIndex] Starting buildIndex with " << npoints << " points...\n";
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Store min/max values
    for (int i = 0; i < 3; i++) {
        this->minVal[i] = minVal[i];
        this->maxVal[i] = maxVal[i];
    }
    
    // Allocate buffers
    allocateBuffers(npoints);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cerr << "[EquiDepthIndex] Buffer allocation time: " 
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
    
    initialize();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cerr << "[EquiDepthIndex] Pipeline initialization time: " 
              << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
    
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    
#if USE_MORTON_BINNING
    // Round-robin binning: Skip phases 1-3 entirely
    // - Phase 1 (histogram) not needed: each bin gets npoints/totalBins
    // - Phase 2 (prefix sum) not needed: trivial arithmetic
    // - Phase 3 (quantiles) not needed: bin = vertexIndex % totalBins
    std::cerr << "[EquiDepthIndex] Using round-robin binning - skipping phases 1-3...\n";
    auto t3 = t2;
#else
    // ========== PHASE 1: Build histograms ==========
    std::cerr << "[EquiDepthIndex] Step 1: Building coarse histograms...\n";
    updateHistogramDescriptors(pointsBuffer, npoints);
    
    vd->commandBuffer->begin(beginInfo);
    
    histXBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eComputeShader);
    histYBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eComputeShader);
    buildHistograms(pointsBuffer, npoints, true);
    
    vd->commandBuffer->end();
    vk::SubmitInfo submitInfo1;
    submitInfo1.commandBufferCount = 1;
    submitInfo1.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence1 = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo1, fence1, false);
    vd->waitForFences(fence1, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence1);
    
    // ========== PHASE 2: Prefix sum on histograms (uses cmdPrefixSum which manages its own command buffer) ==========
    std::cerr << "[EquiDepthIndex] Step 2: Computing CDFs via prefix sum...\n";
    if (scan) {
        uint32_t divisor = scan->getBufSizeDivisor();
        uint32_t paddedHistSize = ((EQUIDEPTH_COARSE_BINS + divisor - 1) / divisor) * divisor;
        scan->cmdPrefixSum(histXBuffer->buf, paddedHistSize);
        scan->cmdPrefixSum(histYBuffer->buf, paddedHistSize);
    }
    
    // ========== PHASE 3: Extract quantile boundaries ==========
    std::cerr << "[EquiDepthIndex] Step 3: Extracting quantile boundaries...\n";
    updateQuantileDescriptors();
    
    vd->commandBuffer->begin(beginInfo);
    computeQuantiles();
    vd->commandBuffer->end();
    vk::SubmitInfo submitInfo2;
    submitInfo2.commandBufferCount = 1;
    submitInfo2.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence2 = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo2, fence2, false);
    vd->waitForFences(fence2, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence2);
#endif
    
    // ========== PHASE 4: Count points per equi-depth bin ==========
    std::cerr << "[EquiDepthIndex] Step 4: Counting points per equi-depth bin...\n";
#if USE_MORTON_BINNING
    {
        // For round-robin, quantiles buffer is not used but we still need to bind something
        vk::DescriptorBufferInfo quantDesc{mortonQuantileBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo countDesc{extentBuffer->buf, 0, VK_WHOLE_SIZE};
        std::vector<vk::WriteDescriptorSet> writes = {
            {countPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &quantDesc},
            {countPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDesc},
        };
        vd->device->updateDescriptorSets(writes, nullptr);
    }
#else
    {
        vk::DescriptorBufferInfo quantXDesc{quantileXBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo quantYDesc{quantileYBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo countDesc{extentBuffer->buf, 0, VK_WHOLE_SIZE};
        std::vector<vk::WriteDescriptorSet> writes = {
            {countPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &quantXDesc},
            {countPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &quantYDesc},
            {countPipelineProps.descriptorSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDesc},
        };
        vd->device->updateDescriptorSets(writes, nullptr);
    }
#endif
    
    vd->commandBuffer->begin(beginInfo);
    extentBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader);
    
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, countPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, countPipelineProps.pipelineLayout.get(), 0, countPipelineProps.descriptorSet.get(), nullptr);
        
#if USE_MORTON_BINNING
        // Push constants: resolution, npoints, minX, maxX, minY, maxY
        uint32_t pc[6] = {INDEX_RESOLUTION, npoints, minVal[0], maxVal[0], minVal[1], maxVal[1]};
        vd->commandBuffer->pushConstants<uint32_t>(countPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, pc);
#else
        uint32_t pc[2] = {INDEX_RESOLUTION, npoints};
        vd->commandBuffer->pushConstants<uint32_t>(countPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, pc);
#endif
        
        vk::DeviceSize offset = 0;
        vd->commandBuffer->bindVertexBuffers(0, pointsBuffer->buf, offset);
        offset += npoints * sizeof(uint32_t);
        vd->commandBuffer->bindVertexBuffers(1, pointsBuffer->buf, offset);
        
        vd->commandBuffer->draw(npoints, 1, 0, 0);
        vd->commandBuffer->endRendering();
    }
    
    // Copy counts to startAddrBuffer
    extentBuffer->barrier(vk::PipelineStageFlagBits::eVertexShader, vk::PipelineStageFlagBits::eTransfer,
                          vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead);
    startAddrBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eTransfer);
    startAddrBuffer->copyFrom(totalBins * sizeof(uint32_t), 0, 0, extentBuffer);
    
    vd->commandBuffer->end();
    vk::SubmitInfo submitInfo3;
    submitInfo3.commandBufferCount = 1;
    submitInfo3.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence3 = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo3, fence3, false);
    vd->waitForFences(fence3, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence3);
    
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cerr << "[EquiDepthIndex] Step 4 (count) time: " 
              << std::chrono::duration<double, std::milli>(t4 - t3).count() << " ms\n";
    
    // ========== PHASE 5: Prefix sum on startAddrBuffer ==========
    std::cerr << "[EquiDepthIndex] Step 5: Computing prefix sum for offsets...\n";
    if (scan) {
        scan->cmdPrefixSum(startAddrBuffer->buf, countBufSize);
    }
    
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cerr << "[EquiDepthIndex] Step 5 (prefix sum) time: " 
              << std::chrono::duration<double, std::milli>(t5 - t4).count() << " ms\n";
    
    // ========== PHASE 6: Insert points into data buffer ==========
    std::cerr << "[EquiDepthIndex] Step 6: Inserting points into data buffer...\n";
#if USE_MORTON_BINNING
    {
        vk::DescriptorBufferInfo quantDesc{mortonQuantileBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo startDesc{startAddrBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo countDesc{countBuffer->buf, 0, VK_WHOLE_SIZE};
        std::vector<vk::WriteDescriptorSet> writes = {
            {buildPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &quantDesc},
            {buildPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &startDesc},
            {buildPipelineProps.descriptorSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDesc},
        };
        vd->device->updateDescriptorSets(writes, nullptr);
    }
#else
    {
        vk::DescriptorBufferInfo quantXDesc{quantileXBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo quantYDesc{quantileYBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo countDesc{countBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo startDesc{startAddrBuffer->buf, 0, VK_WHOLE_SIZE};
        std::vector<vk::WriteDescriptorSet> writes = {
            {buildPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &quantXDesc},
            {buildPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &quantYDesc},
            {buildPipelineProps.descriptorSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDesc},
            {buildPipelineProps.descriptorSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &startDesc},
        };
        vd->device->updateDescriptorSets(writes, nullptr);
    }
#endif
    
    vd->commandBuffer->begin(beginInfo);
    countBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader, 0);
    
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, buildPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, buildPipelineProps.pipelineLayout.get(), 0, buildPipelineProps.descriptorSet.get(), nullptr);
        
        uint64_t dataAddr = dataBuffer->getDeviceAddress();
#if USE_MORTON_BINNING
        // Push constants: resolution, npoints, minX, maxX, minY, maxY, dataBufferAddr[2]
        uint32_t pc[8] = {INDEX_RESOLUTION, npoints, minVal[0], maxVal[0], minVal[1], maxVal[1],
                         static_cast<uint32_t>(dataAddr & 0xFFFFFFFF),
                         static_cast<uint32_t>(dataAddr >> 32)};
        vd->commandBuffer->pushConstants<uint32_t>(buildPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, pc);
#else
        uint32_t pc[6] = {INDEX_RESOLUTION, npoints, 0, 0,
                         static_cast<uint32_t>(dataAddr & 0xFFFFFFFF),
                         static_cast<uint32_t>(dataAddr >> 32)};
        vd->commandBuffer->pushConstants<uint32_t>(buildPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, pc);
#endif
        
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
    vk::SubmitInfo submitInfo4;
    submitInfo4.commandBufferCount = 1;
    submitInfo4.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence4 = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo4, fence4, false);
    vd->waitForFences(fence4, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence4);
    
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cerr << "[EquiDepthIndex] Step 6 (insert) time: " 
              << std::chrono::duration<double, std::milli>(t6 - t5).count() << " ms\n";
    std::cerr << "[EquiDepthIndex] Total GPU work time: " 
              << std::chrono::duration<double, std::milli>(t6 - t3).count() << " ms\n";
    
    std::cerr << "[EquiDepthIndex] Build complete.\n";
}

void EquiDepthIndex::runRangeQueries(PBuffer queryBuffer, uint32_t nqueries, PBuffer resultBuffer) {
    // OPTIMIZED: Pure compute query - same approach as BruteForceIndex
    // Single dispatch over all entries in dataBuffer, check validity and range
    
    // Create cached fence on first use
    if (!queryFence) {
        queryFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    }
    
    // Only update descriptors if buffers changed (avoid redundant updates)
    if (queryBuffer->buf != lastQueryBuffer || resultBuffer->buf != lastResultBuffer) {
        vk::DescriptorBufferInfo queryDesc(queryBuffer->buf, 0, VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo resultDesc(resultBuffer->buf, 0, VK_WHOLE_SIZE);
        std::vector<vk::WriteDescriptorSet> writes = {
            {queryComputeDescSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &queryDesc},
            {queryComputeDescSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &resultDesc},
        };
        vd->device->updateDescriptorSets(writes, nullptr);
        lastQueryBuffer = queryBuffer->buf;
        lastResultBuffer = resultBuffer->buf;
    }
    
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);
    
    // Clear result buffer
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
    
    // Run compute query shader
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, queryComputePipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, queryComputePipelineLayout.get(), 0, queryComputeDescSet.get(), nullptr);
    
    // Push constants: npoints (4 bytes) + padding (4 bytes) + dataBufferAddr (8 bytes)
    uint64_t dataAddr = dataBuffer->getDeviceAddress();
    struct {
        uint32_t npoints;
        uint32_t padding;
        uint32_t dataAddrLo;
        uint32_t dataAddrHi;
    } pushData = {
        npoints,
        0,
        static_cast<uint32_t>(dataAddr & 0xFFFFFFFF),
        static_cast<uint32_t>(dataAddr >> 32)
    };
    vd->commandBuffer->pushConstants(queryComputePipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(pushData), &pushData);
    
    // Dispatch: 256 threads per workgroup
    uint32_t numWorkgroups = (npoints + 255) / 256;
    vd->commandBuffer->dispatch(numWorkgroups, 1, 1);
    
    vd->commandBuffer->end();
    
    // Submit and wait using cached fence
    vd->device->resetFences(queryFence.get());
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vd->submit(submitInfo, queryFence.get(), false);
    vd->waitForFences(queryFence.get(), VK_TRUE, UINT64_MAX);
}

uint32_t EquiDepthIndex::getMaxBinCount() {
    // Read extent buffer and find max
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    std::vector<uint32_t> extents(totalBins);
    
    PBuffer stagingBuf = std::make_shared<Buffer>(vd);
    stagingBuf->create(totalBins * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadOnly);
    
    readUsingStagingBuf((char*)extents.data(), totalBins * sizeof(uint32_t), extentBuffer, stagingBuf, vd);
    stagingBuf->destroy();
    
    uint32_t maxCount = 0;
    for (uint32_t e : extents) {
        if (e > maxCount) maxCount = e;
    }
    return maxCount;
}
