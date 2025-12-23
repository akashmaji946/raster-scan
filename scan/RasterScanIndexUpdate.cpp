// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "RasterScanIndexUpdate.hpp"

#include <core/GPUTimer.hpp>
#include <cmath>

using namespace vkcore;

// ============================================================================
// PageAllocator Implementation
// ============================================================================

PageAllocator::PageAllocator(PVkDevice vd, uint32_t maxPages) 
    : vd(vd), maxPages(maxPages), valid(false) {
    this->pageBufferSize = static_cast<size_t>(maxPages) * PAGE_SIZE_UINTS * sizeof(uint32_t);
    this->initialize();
}

PageAllocator::~PageAllocator() {
    this->destroy();
}

void PageAllocator::initialize() {
    // Create page buffer to store all pages
    pageBuffer.reset(new Buffer(vd));
    pageBuffer->create(pageBufferSize, 
        vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferDst | 
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::Internal);

    // Create allocation counter buffer (stores next free page index)
    allocCounterBuffer.reset(new Buffer(vd));
    allocCounterBuffer->create(sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferDst | 
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::LocalHostVisibleForce);

    valid = true;
}

void PageAllocator::destroy() {
    if (!valid) return;
    
    pageBuffer->destroy();
    pageBuffer.reset();
    allocCounterBuffer->destroy();
    allocCounterBuffer.reset();
    vd.reset();
    valid = false;
}

// ============================================================================
// LinkedListIndex Implementation
// ============================================================================

LinkedListIndex::LinkedListIndex(PVkDevice vd, uint32_t npoints, PPageAllocator pageAlloc)
    : vd(vd), npoints(npoints), pageAlloc(pageAlloc), valid(false) {
    this->indexSize = INDEX_RESOLUTION * INDEX_RESOLUTION;
    this->initialize();
}

LinkedListIndex::~LinkedListIndex() {
    this->destroy();
}

void LinkedListIndex::initialize() {
    // Create head pointer buffer: one uint32_t per cell
    // Initialized to NULL_PAGE_PTR (0xFFFFFFFF)
    headPtrBuffer.reset(new Buffer(vd));
    headPtrBuffer->create(indexSize * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::Internal);

    valid = true;
}

void LinkedListIndex::destroy() {
    if (!valid) return;
    
    headPtrBuffer->destroy();
    headPtrBuffer.reset();
    vd.reset();
    valid = false;
}

// ============================================================================
// RasterScanIndexUpdate Implementation
// ============================================================================

RasterScanIndexUpdate::RasterScanIndexUpdate(PVkDevice vd, PBufferCache bufs, int32_t ncols)
    : vd(vd), bufs(bufs), ncols(ncols) {
    this->initialize();
}

RasterScanIndexUpdate::~RasterScanIndexUpdate() {
    maxBuffer.reset();
}

void RasterScanIndexUpdate::initialize() {
    // Create page allocator
    pageAlloc.reset(new PageAllocator(vd, MAX_PAGES));
    
    this->initShaders();
    this->initBuffers();
    this->setupInsertPipeline();
    this->setupQueryTexturePipeline();
    this->setupQueryPagePipeline();
}

void RasterScanIndexUpdate::initShaders() {
    // Dummy fragment shader
    {
        std::vector<uint32_t> fshader;
        validate(readShader(SHADER_FOLDER + "/dummy.frag.spv", fshader), "dummy fragment shader");
        vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(), 
            fshader.size() * sizeof(uint32_t), fshader.data());
        dummyFragShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // Insert vertex shader
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/insert-page.vert.spv", vshader), "insert page vertex shader");
        vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(),
            vshader.size() * sizeof(uint32_t), vshader.data());
        insertVertShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // Query texture shaders
    {
        std::vector<uint32_t> vshader, gshader, fshader;
        
        validate(readShader(SHADER_FOLDER + "/query-tex-ll.vert.spv", vshader), "query texture ll vertex shader");
        vk::ShaderModuleCreateInfo vCreateInfo(vk::ShaderModuleCreateFlags(),
            vshader.size() * sizeof(uint32_t), vshader.data());
        queryTexVertShader = vd->device->createShaderModuleUnique(vCreateInfo);
        
        validate(readShader(SHADER_FOLDER + "/query-tex-ll.geom.spv", gshader), "query texture ll geom shader");
        vk::ShaderModuleCreateInfo gCreateInfo(vk::ShaderModuleCreateFlags(),
            gshader.size() * sizeof(uint32_t), gshader.data());
        queryTexGeomShader = vd->device->createShaderModuleUnique(gCreateInfo);
        
        validate(readShader(SHADER_FOLDER + "/query-tex-ll.frag.spv", fshader), "query texture ll frag shader");
        vk::ShaderModuleCreateInfo fCreateInfo(vk::ShaderModuleCreateFlags(),
            fshader.size() * sizeof(uint32_t), fshader.data());
        queryTexFragShader = vd->device->createShaderModuleUnique(fCreateInfo);
    }
    
    // Query page shaders
    {
        std::vector<uint32_t> vshader, gshader, fshader;
        
        validate(readShader(SHADER_FOLDER + "/query-page-ll.vert.spv", vshader), "query page ll vertex shader");
        vk::ShaderModuleCreateInfo vCreateInfo(vk::ShaderModuleCreateFlags(),
            vshader.size() * sizeof(uint32_t), vshader.data());
        queryPageVertShader = vd->device->createShaderModuleUnique(vCreateInfo);
        
        validate(readShader(SHADER_FOLDER + "/query-page-ll.geom.spv", gshader), "query page ll geom shader");
        vk::ShaderModuleCreateInfo gCreateInfo(vk::ShaderModuleCreateFlags(),
            gshader.size() * sizeof(uint32_t), gshader.data());
        queryPageGeomShader = vd->device->createShaderModuleUnique(gCreateInfo);
        
        validate(readShader(SHADER_FOLDER + "/query-page-ll.frag.spv", fshader), "query page ll frag shader");
        vk::ShaderModuleCreateInfo fCreateInfo(vk::ShaderModuleCreateFlags(),
            fshader.size() * sizeof(uint32_t), fshader.data());
        queryPageFragShader = vd->device->createShaderModuleUnique(fCreateInfo);
    }
}

void RasterScanIndexUpdate::initBuffers() {
    maxBuffer.reset(new Buffer(vd));
    maxBuffer->create(4 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eIndirectBuffer |
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::LocalHostVisibleForce);
}

void RasterScanIndexUpdate::setupInsertPipeline() {
    std::cerr << "setting up insert pipeline for linked list index\n";
    
    insertPipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eVertex, insertVertShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eFragment, dummyFragShader.get(), "main")
    };
    insertPipelineProps.setShaderStageFlag();

    // Input: x, y, z columns
    insertPipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription(0, sizeof(uint32_t)),
        vk::VertexInputBindingDescription(1, sizeof(uint32_t)),
        vk::VertexInputBindingDescription(2, sizeof(uint32_t)),
    };
    insertPipelineProps.setInputBindingFlag();

    insertPipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
        vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32Uint, 0),
        vk::VertexInputAttributeDescription(2, 2, vk::Format::eR32Uint, 0),
    };
    insertPipelineProps.setInputAttrFlag();

    insertPipelineProps.pipelineInputAssemblyStateCreateInfo = 
        vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), 
            vk::PrimitiveTopology::ePointList);
    insertPipelineProps.setInputAssemblyFlag();

    // Bindings: headPtrBuffer, pageBuffer, allocCounter
    insertPipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };

    insertPipelineProps.poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
    };

    // Push constants: minVal, binRange, res, pageDataSize
    insertPipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint32_t) * 6)
    };
    insertPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    insertPipeline = insertPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
}

void RasterScanIndexUpdate::setupQueryTexturePipeline() {
    std::cerr << "setting up query texture pipeline for linked list index\n";
    
    queryTexturePipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eVertex, queryTexVertShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eFragment, queryTexFragShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eGeometry, queryTexGeomShader.get(), "main"),
    };
    queryTexturePipelineProps.setShaderStageFlag();

    queryTexturePipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription(0, 4 * sizeof(uint32_t)),
        vk::VertexInputBindingDescription(1, 2 * sizeof(uint32_t)),
    };
    queryTexturePipelineProps.setInputBindingFlag();

    queryTexturePipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32A32Uint, 0),
        vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32G32Uint, 0),
    };
    queryTexturePipelineProps.setInputAttrFlag();

    queryTexturePipelineProps.pipelineInputAssemblyStateCreateInfo =
        vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(),
            vk::PrimitiveTopology::ePointList);
    queryTexturePipelineProps.setInputAssemblyFlag();

    // Bindings: headPtrBuffer, pageRangeBuffer (output), resCountBuffer
    queryTexturePipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
    };

    queryTexturePipelineProps.poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
    };

    queryTexturePipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 
            0, sizeof(uint32_t) * 5)
    };
    queryTexturePipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    queryTexturePipeline = queryTexturePipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
}

void RasterScanIndexUpdate::setupQueryPagePipeline() {
    std::cerr << "setting up query page pipeline for linked list index\n";
    
    queryPagePipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eVertex, queryPageVertShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eFragment, queryPageFragShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eGeometry, queryPageGeomShader.get(), "main"),
    };
    queryPagePipelineProps.setShaderStageFlag();

    // Input: page pointer from previous stage
    queryPagePipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription(0, sizeof(uint32_t)),
    };
    queryPagePipelineProps.setInputBindingFlag();

    queryPagePipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
    };
    queryPagePipelineProps.setInputAttrFlag();

    queryPagePipelineProps.pipelineInputAssemblyStateCreateInfo =
        vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(),
            vk::PrimitiveTopology::ePointList);
    queryPagePipelineProps.setInputAssemblyFlag();

    // Bindings: pageBuffer, resBuffer, rangeBuffer
    queryPagePipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
    };

    queryPagePipelineProps.poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
    };

    queryPagePipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, sizeof(uint32_t) * 3)
    };
    queryPagePipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    queryPagePipeline = queryPagePipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
}

inline vk::RenderingInfo setupRenderingLL(PVkDevice vd, PFrameBuffer fbo, vk::RenderingAttachmentInfo &colorInfo) {
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

PLinkedListIndex RasterScanIndexUpdate::buildIndex(PBuffer pointsBuffer, uint32_t npoints, 
                                                    uint32_t *minVal, uint32_t *maxVal) {
    PLinkedListIndex index(new LinkedListIndex(vd, npoints, pageAlloc));
    index->minVal[0] = minVal[0];
    index->maxVal[0] = maxVal[0];
    index->binRange[0] = uint32_t(ceil(double(maxVal[0] - minVal[0]) / INDEX_RESOLUTION));
    index->minVal[1] = minVal[1];
    index->maxVal[1] = maxVal[1];
    index->binRange[1] = uint32_t(ceil(double(maxVal[1] - minVal[1]) / INDEX_RESOLUTION));

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    // Initialize head pointer buffer to NULL_PAGE_PTR
    // Fill with 0xFFFFFFFF
    vd->commandBuffer->fillBuffer(index->headPtrBuffer->buf, 0, 
        index->indexSize * sizeof(uint32_t), NULL_PAGE_PTR);
    index->headPtrBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, 
        vk::PipelineStageFlagBits::eVertexShader,
        vk::AccessFlagBits::eTransferWrite, 
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

    // Initialize page buffer to zeros (important for count fields)
    pageAlloc->pageBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader);

    // Initialize allocation counter to 0
    pageAlloc->allocCounterBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader);

    // Run insert pipeline
    this->runInsertPipeline(pointsBuffer, index, npoints);

    vd->commandBuffer->end();
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);

    return index;
}

void RasterScanIndexUpdate::insertPoints(PLinkedListIndex index, PBuffer pointsBuffer, uint32_t npoints) {
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    this->runInsertPipeline(pointsBuffer, index, npoints);

    vd->commandBuffer->end();
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
}

void RasterScanIndexUpdate::deletePoints(PLinkedListIndex index, PBuffer pointsBuffer, uint32_t npoints) {
    // Delete is implemented as marking items in pages
    // For simplicity, we can use a separate delete shader or mark items with a special value
    // This is a placeholder - actual implementation would need a delete shader
    std::cerr << "Delete operation not yet implemented - use update (delete + insert)\n";
}

void RasterScanIndexUpdate::runRangeQueries(PLinkedListIndex index, PBuffer qranges, uint32_t nqueries) {
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());

    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    // Clear result buffer
    bufs->resBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eFragmentShader);
    
    // Single stage: Query texture and traverse linked lists directly
    this->runQueryTexturePipeline(index, qranges, nqueries);
    
    vd->commandBuffer->end();
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
}

void RasterScanIndexUpdate::runInsertPipeline(PBuffer pointsBuffer, PLinkedListIndex index, uint32_t npoints) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRenderingLL(vd, bufs->dummyFbo, colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, insertPipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo headPtrDescriptor{index->headPtrBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo allocDescriptor{pageAlloc->allocCounterBuffer->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{insertPipelineProps.descriptorSet.get(), 0, 0, 1, 
            vk::DescriptorType::eStorageBuffer, nullptr, &headPtrDescriptor},
        vk::WriteDescriptorSet{insertPipelineProps.descriptorSet.get(), 1, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
        vk::WriteDescriptorSet{insertPipelineProps.descriptorSet.get(), 2, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &allocDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, 
        insertPipelineProps.pipelineLayout.get(), 0, insertPipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t, 6> consts = {
        index->minVal[0], index->minVal[1], 
        index->binRange[0], index->binRange[1], 
        INDEX_RESOLUTION, PAGE_DATA_SIZE
    };
    vd->commandBuffer->pushConstants<uint32_t>(insertPipelineProps.pipelineLayout.get(),
        vk::ShaderStageFlagBits::eVertex, 0, consts);

    vk::DeviceSize offset = 0;
    vd->commandBuffer->bindVertexBuffers(0, pointsBuffer->buf, offset);
    offset += npoints * sizeof(uint32_t);
    vd->commandBuffer->bindVertexBuffers(1, pointsBuffer->buf, offset);
    if (ncols == 3) {
        offset += npoints * sizeof(uint32_t);
    }
    vd->commandBuffer->bindVertexBuffers(2, pointsBuffer->buf, offset);

    vd->commandBuffer->draw(npoints, 1, 0, 0);
    vd->commandBuffer->endRendering();
}

void RasterScanIndexUpdate::runQueryTexturePipeline(PLinkedListIndex index, PBuffer qranges, uint32_t nqueries) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRenderingLL(vd, bufs->dummyFbo, colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, queryTexturePipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    // Bindings: headPtrBuffer, pageBuffer, resBuffer
    vk::DescriptorBufferInfo headPtrDescriptor{index->headPtrBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo resDescriptor{bufs->resBuffer->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{queryTexturePipelineProps.descriptorSet.get(), 0, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &headPtrDescriptor},
        vk::WriteDescriptorSet{queryTexturePipelineProps.descriptorSet.get(), 1, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
        vk::WriteDescriptorSet{queryTexturePipelineProps.descriptorSet.get(), 2, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &resDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        queryTexturePipelineProps.pipelineLayout.get(), 0, queryTexturePipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t, 5> consts = {
        index->minVal[0], index->minVal[1],
        index->binRange[0], index->binRange[1],
        INDEX_RESOLUTION
    };
    vd->commandBuffer->pushConstants<uint32_t>(queryTexturePipelineProps.pipelineLayout.get(),
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, consts);

    vk::DeviceSize offset = 0;
    vd->commandBuffer->bindVertexBuffers(0, qranges->buf, offset);
    offset += nqueries * 4 * sizeof(uint32_t);
    vd->commandBuffer->bindVertexBuffers(1, qranges->buf, offset);

    vd->commandBuffer->draw(nqueries, 1, 0, 0);
    vd->commandBuffer->endRendering();
}

void RasterScanIndexUpdate::runQueryPagePipeline(PLinkedListIndex index, PBuffer qranges, uint32_t nqueries) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRenderingLL(vd, bufs->dummyFbo, colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, queryPagePipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo resDescriptor{bufs->resBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo rangeDescriptor{qranges->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{queryPagePipelineProps.descriptorSet.get(), 0, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
        vk::WriteDescriptorSet{queryPagePipelineProps.descriptorSet.get(), 1, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &resDescriptor},
        vk::WriteDescriptorSet{queryPagePipelineProps.descriptorSet.get(), 2, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &rangeDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        queryPagePipelineProps.pipelineLayout.get(), 0, queryPagePipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t, 3> consts = {
        INDEX_RESOLUTION, static_cast<uint32_t>(ncols), PAGE_DATA_SIZE
    };
    vd->commandBuffer->pushConstants<uint32_t>(queryPagePipelineProps.pipelineLayout.get(),
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, consts);

    vk::DeviceSize offset = 0;
    vd->commandBuffer->bindVertexBuffers(0, bufs->edgeBuffer->buf, offset);
    vd->commandBuffer->drawIndirect(maxBuffer->buf, 0, 1, 4 * sizeof(uint32_t));
    vd->commandBuffer->endRendering();
}
