// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "RasterScanIndexUpdate.hpp"
#include "GPUMemoryTool.hpp"

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
    GPU_MEM_PRINT("PageAllocator::pageBuffer", pageBufferSize);
    GPUMemoryTool::printGPUMemoryStatus(vd, "After pageBuffer allocation");

    // Create allocation counter buffer (stores next free page index)
    allocCounterBuffer.reset(new Buffer(vd));
    allocCounterBuffer->create(sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferDst | 
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::LocalHostVisibleForce);
    GPU_MEM_PRINT("PageAllocator::allocCounter", sizeof(uint32_t));
    GPUMemoryTool::printGPUMemoryStatus(vd, "After allocCounter allocation");

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
    size_t headBufferSize = indexSize * sizeof(uint32_t);
    headPtrBuffer.reset(new Buffer(vd));
    headPtrBuffer->create(headBufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::Internal);
    GPU_MEM_PRINT("LinkedListIndex::headPtrBuffer", headBufferSize);
    GPUMemoryTool::printGPUMemoryStatus(vd, "After headPtrBuffer allocation");

    // Initialize head pointer buffer to NULL_PAGE_PTR (0xFFFFFFFF)
    // This is critical - uninitialized device-local memory contains garbage
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vd->commandBuffer->fillBuffer(headPtrBuffer->buf, 0, headBufferSize, NULL_PAGE_PTR);
    headPtrBuffer->barrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eVertexShader,
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    vd->commandBuffer->end();
    vk::UniqueFence fence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence.get(), false);
    vd->device->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);

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
    // NOTE: PageAllocator is created lazily in buildIndex() to avoid
    // allocating GPU memory until it's actually needed
    
    this->initShaders();
    this->initBuffers();
    this->setupInsertPipeline();
    this->setupDeletePipeline();
    this->setupDeleteRangePipeline();
    this->setupQueryTexturePipeline();
    this->setupQueryPagePipeline();
    this->setupInsertBitmapPipeline();
    this->setupInsertBitmapV2Pipeline();
    this->setupDeleteByDataPipeline();
    this->setupMarkFreePipeline();
    this->setupCompactPipeline();
    this->setupVerifyIndexPipeline();
    
    // Print GPU memory after pipeline setup (no large buffers allocated yet)
    GPUMemoryTool::printGPUMemoryStatus(vd, "After RasterScanIndexUpdate init (pipelines only)");
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
    
    // Delete vertex shader
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/delete-page.vert.spv", vshader), "delete page vertex shader");
        vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(),
            vshader.size() * sizeof(uint32_t), vshader.data());
        deleteVertShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // Delete range vertex shader
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/delete-range.vert.spv", vshader), "delete range vertex shader");
        vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(),
            vshader.size() * sizeof(uint32_t), vshader.data());
        deleteRangeVertShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // Insert bitmap vertex shader
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/insert-page-bitmap.vert.spv", vshader), "insert page bitmap vertex shader");
        vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(),
            vshader.size() * sizeof(uint32_t), vshader.data());
        insertBitmapVertShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // Mark free vertex shader
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/mark-free-bitmap.vert.spv", vshader), "mark free bitmap vertex shader");
        vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(),
            vshader.size() * sizeof(uint32_t), vshader.data());
        markFreeVertShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // Compact pages vertex shader
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/compact-pages.vert.spv", vshader), "compact pages vertex shader");
        vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(),
            vshader.size() * sizeof(uint32_t), vshader.data());
        compactVertShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // Insert bitmap V2 vertex shader (stores pageId in count field)
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/insert-page-bitmap-v2.vert.spv", vshader), "insert page bitmap v2 vertex shader");
        vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(),
            vshader.size() * sizeof(uint32_t), vshader.data());
        insertBitmapV2VertShader = vd->device->createShaderModuleUnique(createInfo);
    }
    
    // Delete by data vertex shader
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/delete-by-data.vert.spv", vshader), "delete by data vertex shader");
        vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(),
            vshader.size() * sizeof(uint32_t), vshader.data());
        deleteByDataVertShader = vd->device->createShaderModuleUnique(createInfo);
    }

    // Verify index compute shader
    {
        std::vector<uint32_t> cshader;
        validate(readShader(SHADER_FOLDER + "/verify-index.comp.spv", cshader), "verify index compute shader");
        vk::ShaderModuleCreateInfo createInfo(vk::ShaderModuleCreateFlags(),
            cshader.size() * sizeof(uint32_t), cshader.data());
        verifyIndexCompShader = vd->device->createShaderModuleUnique(createInfo);
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

    // Push constants: minVal, binRange, res, pageDataSize, rowIdOffset
    insertPipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint32_t) * 7)
    };
    insertPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    insertPipeline = insertPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
}

void RasterScanIndexUpdate::setupDeletePipeline() {
    std::cerr << "setting up delete pipeline for linked list index\n";
    
    deletePipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eVertex, deleteVertShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eFragment, dummyFragShader.get(), "main")
    };
    deletePipelineProps.setShaderStageFlag();

    // Input: rowId to delete
    deletePipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription(0, sizeof(uint32_t)),
    };
    deletePipelineProps.setInputBindingFlag();

    deletePipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
    };
    deletePipelineProps.setInputAttrFlag();

    deletePipelineProps.pipelineInputAssemblyStateCreateInfo = 
        vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), 
            vk::PrimitiveTopology::ePointList);
    deletePipelineProps.setInputAssemblyFlag();

    // Bindings: only pageBuffer (binding 1 to match shader)
    deletePipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };

    deletePipelineProps.poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
    };

    // No push constants needed - we directly use rowId as page index
    deletePipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    deletePipeline = deletePipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
}

void RasterScanIndexUpdate::setupDeleteRangePipeline() {
    std::cerr << "setting up delete range pipeline for linked list index\n";
    
    deleteRangePipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eVertex, deleteRangeVertShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eFragment, dummyFragShader.get(), "main")
    };
    deleteRangePipelineProps.setShaderStageFlag();

    // No vertex input - we use gl_VertexIndex to iterate over cells
    deleteRangePipelineProps.vertexInputBindingDescriptions = {};
    deleteRangePipelineProps.setInputBindingFlag();
    deleteRangePipelineProps.vertexInputAttributeDescriptions = {};
    deleteRangePipelineProps.setInputAttrFlag();
    
    deleteRangePipelineProps.pipelineInputAssemblyStateCreateInfo = 
        vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), 
            vk::PrimitiveTopology::ePointList);
    deleteRangePipelineProps.setInputAssemblyFlag();

    // Bindings: headPtrBuffer, pageBuffer
    deleteRangePipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };

    deleteRangePipelineProps.poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
    };

    // Push constants: x1, x2, y1, y2, z1, z2, res (7 uint32_t)
    deleteRangePipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint32_t) * 7)
    };
    deleteRangePipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    deleteRangePipeline = deleteRangePipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
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
    // Print memory requirements for this index
    std::cerr << "\n";
    GPUMemoryTool::printLinkedListIndexMemory(MAX_PAGES, PAGE_SIZE_UINTS, INDEX_RESOLUTION, npoints, ncols);
    
    // Create page allocator lazily (only when buildIndex is called)
    // This allows RasterScan2D to release its memory first
    if (!pageAlloc) {
        GPUMemoryTool::printGPUMemoryStatus(vd, "Before PageAllocator creation");
        pageAlloc.reset(new PageAllocator(vd, MAX_PAGES));
        GPUMemoryTool::printGPUMemoryStatus(vd, "After PageAllocator creation");
    }
    
    PLinkedListIndex index(new LinkedListIndex(vd, npoints, pageAlloc));
    index->minVal[0] = minVal[0];
    index->maxVal[0] = maxVal[0];
    index->binRange[0] = uint32_t(floor(double(maxVal[0] - minVal[0]) / INDEX_RESOLUTION)) + 1;
    index->minVal[1] = minVal[1];
    index->maxVal[1] = maxVal[1];
    index->binRange[1] = uint32_t(floor(double(maxVal[1] - minVal[1]) / INDEX_RESOLUTION)) + 1;

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

    // Print GPU memory status after building index
    GPUMemoryTool::printGPUMemoryStatus(vd, "After buildIndex complete");

    return index;
}

void RasterScanIndexUpdate::insertPoints(PLinkedListIndex index, PBuffer pointsBuffer, uint32_t npoints, uint32_t rowIdOffset) {
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    this->runInsertPipeline(pointsBuffer, index, npoints, rowIdOffset);

    vd->commandBuffer->end();
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
}

void RasterScanIndexUpdate::deletePoints(PLinkedListIndex index, PBuffer rowIdBuffer, uint32_t ndeletes) {
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    this->runDeletePipeline(rowIdBuffer, index, ndeletes);

    vd->commandBuffer->end();
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
}

void RasterScanIndexUpdate::deleteRange(PLinkedListIndex index, uint32_t* range) {
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    this->runDeleteRangePipeline(index, range);

    vd->commandBuffer->end();
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
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

void RasterScanIndexUpdate::runInsertPipeline(PBuffer pointsBuffer, PLinkedListIndex index, uint32_t npoints, uint32_t rowIdOffset) {
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

    std::array<uint32_t, 7> consts = {
        index->minVal[0], index->minVal[1], 
        index->binRange[0], index->binRange[1], 
        INDEX_RESOLUTION, PAGE_DATA_SIZE, rowIdOffset
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

void RasterScanIndexUpdate::runDeletePipeline(PBuffer rowIdBuffer, PLinkedListIndex index, uint32_t ndeletes) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRenderingLL(vd, bufs->dummyFbo, colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, deletePipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    // Only need pageBuffer - we directly access page by rowId
    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{deletePipelineProps.descriptorSet.get(), 1, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, 
        deletePipelineProps.pipelineLayout.get(), 0, deletePipelineProps.descriptorSet.get(), nullptr);

    // No push constants needed

    vk::DeviceSize offset = 0;
    vd->commandBuffer->bindVertexBuffers(0, rowIdBuffer->buf, offset);

    vd->commandBuffer->draw(ndeletes, 1, 0, 0);
    vd->commandBuffer->endRendering();
}

void RasterScanIndexUpdate::runDeleteRangePipeline(PLinkedListIndex index, uint32_t* range) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRenderingLL(vd, bufs->dummyFbo, colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, deleteRangePipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    // Bindings: headPtrBuffer, pageBuffer
    vk::DescriptorBufferInfo headPtrDescriptor{index->headPtrBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{deleteRangePipelineProps.descriptorSet.get(), 0, 0, 1, 
            vk::DescriptorType::eStorageBuffer, nullptr, &headPtrDescriptor},
        vk::WriteDescriptorSet{deleteRangePipelineProps.descriptorSet.get(), 1, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, 
        deleteRangePipelineProps.pipelineLayout.get(), 0, deleteRangePipelineProps.descriptorSet.get(), nullptr);

    // Push constants: x1, x2, y1, y2, z1, z2, res
    std::array<uint32_t, 7> consts = {
        range[0], range[1], range[2], range[3], range[4], range[5], INDEX_RESOLUTION
    };
    vd->commandBuffer->pushConstants<uint32_t>(deleteRangePipelineProps.pipelineLayout.get(),
        vk::ShaderStageFlagBits::eVertex, 0, consts);

    // Draw one vertex per grid cell
    uint32_t totalCells = INDEX_RESOLUTION * INDEX_RESOLUTION;
    vd->commandBuffer->draw(totalCells, 1, 0, 0);
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

// ============================================================================
// FreeBitmap Implementation
// ============================================================================

FreeBitmap::FreeBitmap(PVkDevice vd, uint32_t maxPages)
    : vd(vd), maxPages(maxPages), valid(false) {
    this->bitmapSizeUints = maxPages / 32;
    this->initialize();
}

FreeBitmap::~FreeBitmap() {
    this->destroy();
}

void FreeBitmap::initialize() {
    // Create bitmap buffer (1 = free, 0 = allocated)
    size_t bitmapSize = bitmapSizeUints * sizeof(uint32_t);
    bitmapBuffer.reset(new Buffer(vd));
    bitmapBuffer->create(bitmapSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::Internal);
    GPU_MEM_PRINT("FreeBitmap::bitmapBuffer", bitmapSize);
    
    // Create free count buffer
    freeCountBuffer.reset(new Buffer(vd));
    freeCountBuffer->create(sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::LocalHostVisibleForce);
    GPU_MEM_PRINT("FreeBitmap::freeCountBuffer", sizeof(uint32_t));
    
    // Create next free hint buffer
    nextFreeHintBuffer.reset(new Buffer(vd));
    nextFreeHintBuffer->create(sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::LocalHostVisibleForce);
    GPU_MEM_PRINT("FreeBitmap::nextFreeHintBuffer", sizeof(uint32_t));
    
    // Initialize all buffers to 0 using command buffer
    // bitmap = 0 means all pages are allocated initially
    // freeCount = 0, nextFreeHint = 0
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    // Clear bitmap buffer to 0 (all pages allocated)
    vd->commandBuffer->fillBuffer(bitmapBuffer->buf, 0, bitmapSize, 0);
    bitmapBuffer->barrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eVertexShader,
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    
    // Clear freeCount to 0
    vd->commandBuffer->fillBuffer(freeCountBuffer->buf, 0, sizeof(uint32_t), 0);
    freeCountBuffer->barrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eVertexShader,
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    
    // Clear nextFreeHint to 0
    vd->commandBuffer->fillBuffer(nextFreeHintBuffer->buf, 0, sizeof(uint32_t), 0);
    nextFreeHintBuffer->barrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eVertexShader,
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    
    vd->commandBuffer->end();
    vk::UniqueFence fence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence.get(), false);
    vd->device->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
    
    valid = true;
}

void FreeBitmap::initializeWithAllocatedCount(uint32_t allocatedPages) {
    // Initialize bitmap: first allocatedPages bits = 0 (allocated), rest = 1 (free)
    // This is called after initial index build to set up the bitmap correctly
    
    if (!valid) {
        initialize();
    }
    
    uint32_t freePages = maxPages - allocatedPages;
    
    // Calculate how many full words are allocated (all bits = 0)
    uint32_t fullAllocatedWords = allocatedPages / 32;
    uint32_t remainingAllocatedBits = allocatedPages % 32;
    
    // Prepare bitmap data on CPU
    std::vector<uint32_t> bitmapData(bitmapSizeUints, 0xFFFFFFFF);  // All free initially
    
    // Set first fullAllocatedWords to 0 (all allocated)
    for (uint32_t i = 0; i < fullAllocatedWords; i++) {
        bitmapData[i] = 0;
    }
    
    // Handle partial word at boundary
    if (remainingAllocatedBits > 0 && fullAllocatedWords < bitmapSizeUints) {
        // Lower remainingAllocatedBits are allocated (0), upper bits are free (1)
        uint32_t mask = 0xFFFFFFFF << remainingAllocatedBits;
        bitmapData[fullAllocatedWords] = mask;
    }
    
    // Upload bitmap to GPU using staging buffer approach
    // Since bitmapBuffer is device-local, we need to use a staging buffer
    PBuffer stagingBuffer(new Buffer(vd));
    uint32_t bitmapSize = bitmapSizeUints * sizeof(uint32_t);
    stagingBuffer->create(bitmapSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::ReadWrite);
    
    // Write data to staging buffer
    stagingBuffer->loadData((char*)bitmapData.data(), bitmapSize, 0);
    
    // Copy from staging to device-local bitmap buffer
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    vk::BufferCopy copyRegion(0, 0, bitmapSize);
    vd->commandBuffer->copyBuffer(stagingBuffer->buf, bitmapBuffer->buf, copyRegion);
    
    bitmapBuffer->barrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eVertexShader,
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    
    vd->commandBuffer->end();
    vk::UniqueFence fence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence.get(), false);
    vd->device->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
    
    // Clean up staging buffer
    stagingBuffer->destroy();
    stagingBuffer.reset();
    
    // Set freeCount
    freeCountBuffer->loadData((char*)&freePages, sizeof(uint32_t), 0);
    
    // Reset nextFreeHint to point to first free word
    uint32_t hint = fullAllocatedWords;
    nextFreeHintBuffer->loadData((char*)&hint, sizeof(uint32_t), 0);
    
    std::cerr << "[FreeBitmap] Initialized with " << allocatedPages << " allocated pages, " 
              << freePages << " free pages\n";
}

void FreeBitmap::destroy() {
    if (!valid) return;
    
    bitmapBuffer->destroy();
    bitmapBuffer.reset();
    freeCountBuffer->destroy();
    freeCountBuffer.reset();
    nextFreeHintBuffer->destroy();
    nextFreeHintBuffer.reset();
    vd.reset();
    valid = false;
}

// ============================================================================
// Bitmap-based Pipeline Setup
// ============================================================================

void RasterScanIndexUpdate::setupInsertBitmapPipeline() {
    std::cerr << "setting up insert bitmap pipeline for linked list index\n";
    
    insertBitmapPipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eVertex, insertBitmapVertShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eFragment, dummyFragShader.get(), "main")
    };
    insertBitmapPipelineProps.setShaderStageFlag();

    insertBitmapPipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription(0, sizeof(uint32_t)),
        vk::VertexInputBindingDescription(1, sizeof(uint32_t)),
        vk::VertexInputBindingDescription(2, sizeof(uint32_t)),
    };
    insertBitmapPipelineProps.setInputBindingFlag();

    insertBitmapPipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
        vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32Uint, 0),
        vk::VertexInputAttributeDescription(2, 2, vk::Format::eR32Uint, 0),
    };
    insertBitmapPipelineProps.setInputAttrFlag();

    insertBitmapPipelineProps.pipelineInputAssemblyStateCreateInfo = 
        vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), 
            vk::PrimitiveTopology::ePointList);
    insertBitmapPipelineProps.setInputAssemblyFlag();

    // Bindings: headPtrBuffer, pageBuffer, freeBitmap, freeCount, nextFreeHint
    insertBitmapPipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };

    insertBitmapPipelineProps.poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 5},
    };

    // Push constants: minVal, binRange, res, pageDataSize, rowIdOffset, maxPages
    insertBitmapPipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint32_t) * 8)
    };
    insertBitmapPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    insertBitmapPipeline = insertBitmapPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
}

void RasterScanIndexUpdate::setupInsertBitmapV2Pipeline() {
    std::cerr << "setting up insert bitmap V2 pipeline for linked list index\n";
    
    insertBitmapV2PipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eVertex, insertBitmapV2VertShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eFragment, dummyFragShader.get(), "main")
    };
    insertBitmapV2PipelineProps.setShaderStageFlag();

    insertBitmapV2PipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription(0, sizeof(uint32_t)),
        vk::VertexInputBindingDescription(1, sizeof(uint32_t)),
        vk::VertexInputBindingDescription(2, sizeof(uint32_t)),
    };
    insertBitmapV2PipelineProps.setInputBindingFlag();

    insertBitmapV2PipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
        vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32Uint, 0),
        vk::VertexInputAttributeDescription(2, 2, vk::Format::eR32Uint, 0),
    };
    insertBitmapV2PipelineProps.setInputAttrFlag();

    insertBitmapV2PipelineProps.pipelineInputAssemblyStateCreateInfo = 
        vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), 
            vk::PrimitiveTopology::ePointList);
    insertBitmapV2PipelineProps.setInputAssemblyFlag();

    // Bindings: headPtrBuffer, pageBuffer, freeBitmap, freeCount, nextFreeHint
    insertBitmapV2PipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };

    insertBitmapV2PipelineProps.poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 5},
    };

    // Push constants: minVal, binRange, res, pageDataSize, rowIdOffset, maxPages
    insertBitmapV2PipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint32_t) * 8)
    };
    insertBitmapV2PipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    insertBitmapV2Pipeline = insertBitmapV2PipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
}

void RasterScanIndexUpdate::setupDeleteByDataPipeline() {
    std::cerr << "setting up delete-by-data pipeline for linked list index\n";
    
    deleteByDataPipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eVertex, deleteByDataVertShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eFragment, dummyFragShader.get(), "main")
    };
    deleteByDataPipelineProps.setShaderStageFlag();

    // Input: x, y, z columns (same as insert)
    deleteByDataPipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription(0, sizeof(uint32_t)),
        vk::VertexInputBindingDescription(1, sizeof(uint32_t)),
        vk::VertexInputBindingDescription(2, sizeof(uint32_t)),
    };
    deleteByDataPipelineProps.setInputBindingFlag();

    deleteByDataPipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),
        vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32Uint, 0),
        vk::VertexInputAttributeDescription(2, 2, vk::Format::eR32Uint, 0),
    };
    deleteByDataPipelineProps.setInputAttrFlag();

    deleteByDataPipelineProps.pipelineInputAssemblyStateCreateInfo = 
        vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), 
            vk::PrimitiveTopology::ePointList);
    deleteByDataPipelineProps.setInputAssemblyFlag();

    // Bindings: headPtrBuffer, pageBuffer, freeBitmap, freeCount
    deleteByDataPipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };

    deleteByDataPipelineProps.poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 4},
    };

    // Push constants: minVal, binRange, res, maxPages
    deleteByDataPipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint32_t) * 6)
    };
    deleteByDataPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    deleteByDataPipeline = deleteByDataPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
}

void RasterScanIndexUpdate::setupMarkFreePipeline() {
    std::cerr << "setting up mark-free pipeline for linked list index\n";
    
    markFreePipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eVertex, markFreeVertShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eFragment, dummyFragShader.get(), "main")
    };
    markFreePipelineProps.setShaderStageFlag();

    // No vertex input - we use gl_VertexIndex as page index
    markFreePipelineProps.vertexInputBindingDescriptions = {};
    markFreePipelineProps.setInputBindingFlag();
    markFreePipelineProps.vertexInputAttributeDescriptions = {};
    markFreePipelineProps.setInputAttrFlag();

    markFreePipelineProps.pipelineInputAssemblyStateCreateInfo = 
        vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), 
            vk::PrimitiveTopology::ePointList);
    markFreePipelineProps.setInputAssemblyFlag();

    // Bindings: pageBuffer, freeBitmap, freeCount
    markFreePipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };

    markFreePipelineProps.poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 3},
    };

    // Push constants: maxPages, allocCounter
    markFreePipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint32_t) * 2)
    };
    markFreePipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    markFreePipeline = markFreePipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
}

void RasterScanIndexUpdate::setupCompactPipeline() {
    std::cerr << "setting up compact pipeline for linked list index\n";
    
    compactPipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eVertex, compactVertShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
            vk::ShaderStageFlagBits::eFragment, dummyFragShader.get(), "main")
    };
    compactPipelineProps.setShaderStageFlag();

    // No vertex input - we use gl_VertexIndex as cell index
    compactPipelineProps.vertexInputBindingDescriptions = {};
    compactPipelineProps.setInputBindingFlag();
    compactPipelineProps.vertexInputAttributeDescriptions = {};
    compactPipelineProps.setInputAttrFlag();

    compactPipelineProps.pipelineInputAssemblyStateCreateInfo = 
        vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), 
            vk::PrimitiveTopology::ePointList);
    compactPipelineProps.setInputAssemblyFlag();

    // Bindings: headPtrBuffer, pageBuffer, freeBitmap, freeCount
    compactPipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };

    compactPipelineProps.poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 4},
    };

    // Push constants: res
    compactPipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint32_t))
    };
    compactPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPass;
    compactPipeline = compactPipelineProps.createPipeline(vd, dummyRenderPass, &rpCreateInfo);
}

// ============================================================================
// Bitmap-based Operations
// ============================================================================

void RasterScanIndexUpdate::initializeBitmapWithAllocation(uint32_t allocatedPages) {
    // Create and initialize the bitmap with the correct allocation state
    // First allocatedPages are marked as allocated (bit=0), rest are free (bit=1)
    // Use pageAlloc->maxPages if available, otherwise fall back to MAX_PAGES
    uint32_t maxPages = pageAlloc ? pageAlloc->maxPages : MAX_PAGES;
    if (!freeBitmap) {
        freeBitmap.reset(new FreeBitmap(vd, maxPages));
    }
    freeBitmap->initializeWithAllocatedCount(allocatedPages);
}

void RasterScanIndexUpdate::markDeletedPagesAsFree(PLinkedListIndex index) {
    // Ensure freeBitmap is initialized
    if (!freeBitmap) {
        uint32_t maxPages = pageAlloc ? pageAlloc->maxPages : MAX_PAGES;
        freeBitmap.reset(new FreeBitmap(vd, maxPages));
    }
    
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    this->runMarkFreePipeline(index);
    
    vd->commandBuffer->end();
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
}

void RasterScanIndexUpdate::compactPages(PLinkedListIndex index) {
    // Ensure freeBitmap is initialized
    if (!freeBitmap) {
        uint32_t maxPages = pageAlloc ? pageAlloc->maxPages : MAX_PAGES;
        freeBitmap.reset(new FreeBitmap(vd, maxPages));
    }
    
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    this->runCompactPipeline(index);
    
    vd->commandBuffer->end();
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
}

void RasterScanIndexUpdate::insertPointsWithBitmap(PLinkedListIndex index, PBuffer pointsBuffer, 
                                                    uint32_t npoints, uint32_t rowIdOffset) {
    // Ensure freeBitmap is initialized
    if (!freeBitmap) {
        uint32_t maxPages = pageAlloc ? pageAlloc->maxPages : MAX_PAGES;
        freeBitmap.reset(new FreeBitmap(vd, maxPages));
    }
    
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    this->runInsertBitmapPipeline(pointsBuffer, index, npoints, rowIdOffset);
    
    vd->commandBuffer->end();
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
}

void RasterScanIndexUpdate::insertPointsWithBitmapV2(PLinkedListIndex index, PBuffer pointsBuffer, 
                                                      uint32_t npoints, uint32_t rowIdOffset) {
    // Ensure freeBitmap is initialized
    if (!freeBitmap) {
        uint32_t maxPages = pageAlloc ? pageAlloc->maxPages : MAX_PAGES;
        freeBitmap.reset(new FreeBitmap(vd, maxPages));
    }
    
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    this->runInsertBitmapV2Pipeline(pointsBuffer, index, npoints, rowIdOffset);
    
    vd->commandBuffer->end();
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
}

void RasterScanIndexUpdate::deletePointsByData(PLinkedListIndex index, PBuffer dataBuffer, uint32_t ndeletes) {
    // Ensure freeBitmap is initialized
    if (!freeBitmap) {
        uint32_t maxPages = pageAlloc ? pageAlloc->maxPages : MAX_PAGES;
        freeBitmap.reset(new FreeBitmap(vd, maxPages));
    }
    
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    this->runDeleteByDataPipeline(dataBuffer, index, ndeletes);
    
    vd->commandBuffer->end();
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, drawFence.get(), false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
}

RasterScanIndexUpdate::AllocationStats RasterScanIndexUpdate::getAllocationStats(PLinkedListIndex index) {
    AllocationStats stats = {0, 0, 0, 0, 0, 0};
    stats.totalPages = MAX_PAGES;
    
    // Read free count from bitmap
    uint32_t freeCount = 0;
    if (freeBitmap) {
        freeBitmap->freeCountBuffer->readData((char*)&freeCount, sizeof(uint32_t));
    }
    stats.freePages = freeCount;
    
    // With pure bitmap approach: allocated = total - free
    stats.allocatedPages = MAX_PAGES - freeCount;
    stats.unallocatedPages = 0;  // All pages are tracked in bitmap
    
    // For pure bitmap approach, valid = allocated (pages in use)
    // Invalid pages are freed back to bitmap, so invalidPages = 0
    // The bitmap tracks: free (bit=1) vs allocated (bit=0)
    // Allocated pages are either valid (in linked list) or will be freed by compact
    stats.validPages = stats.allocatedPages;
    stats.invalidPages = 0;
    
    return stats;
}

void RasterScanIndexUpdate::countValidInvalidPages(uint32_t allocCounter, uint32_t& validCount, uint32_t& invalidCount) {
    validCount = 0;
    invalidCount = 0;
    
    if (allocCounter == 0) return;
    
    // Read page buffer in chunks to count valid/invalid pages
    // Each page is PAGE_SIZE_UINTS (6) uint32_t values
    // The rowId with valid bit is at offset 3 within each page
    const uint32_t CHUNK_SIZE = 256 * 1024;  // 256K pages per chunk (to fit in staging buffer)
    const uint32_t VALID_BIT_MASK = 0x80000000u;
    
    // Create a staging buffer for reading from device-local memory
    PBuffer stagingBuffer(new Buffer(vd));
    uint32_t stagingSize = CHUNK_SIZE * PAGE_SIZE_UINTS * sizeof(uint32_t);
    stagingBuffer->create(stagingSize,
        vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::ReadWrite);
    
    std::vector<uint32_t> pageData(CHUNK_SIZE * PAGE_SIZE_UINTS);
    
    for (uint32_t chunkStart = 0; chunkStart < allocCounter; chunkStart += CHUNK_SIZE) {
        uint32_t chunkEnd = std::min(chunkStart + CHUNK_SIZE, allocCounter);
        uint32_t chunkPages = chunkEnd - chunkStart;
        uint32_t bytesToRead = chunkPages * PAGE_SIZE_UINTS * sizeof(uint32_t);
        uint32_t offsetBytes = chunkStart * PAGE_SIZE_UINTS * sizeof(uint32_t);
        
        // Copy from device-local pageBuffer to staging buffer using command buffer
        vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
        vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        
        vk::BufferCopy copyRegion(offsetBytes, 0, bytesToRead);
        vd->commandBuffer->copyBuffer(pageAlloc->pageBuffer->buf, stagingBuffer->buf, copyRegion);
        
        vd->commandBuffer->end();
        vk::UniqueFence fence = vd->device->createFenceUnique(vk::FenceCreateInfo());
        vd->submit(submitInfo, fence.get(), false);
        vd->device->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
        
        // Read from staging buffer (host-visible)
        stagingBuffer->readData((char*)pageData.data(), bytesToRead, 0);
        
        // Count valid/invalid pages in this chunk
        for (uint32_t i = 0; i < chunkPages; i++) {
            uint32_t pageOffset = i * PAGE_SIZE_UINTS;
            uint32_t rowIdWithValid = pageData[pageOffset + 3];  // rowId is at offset 3
            
            if (rowIdWithValid & VALID_BIT_MASK) {
                validCount++;
            } else {
                invalidCount++;
            }
        }
    }
    
    stagingBuffer->destroy();
    stagingBuffer.reset();
}

void RasterScanIndexUpdate::traverseLinkedListsAndCount(PLinkedListIndex index, uint32_t& validCount, uint32_t& invalidCount) {
    validCount = 0;
    invalidCount = 0;
    
    const uint32_t VALID_BIT_MASK = 0x80000000u;
    uint32_t totalCells = INDEX_RESOLUTION * INDEX_RESOLUTION;
    
    // Read head pointers
    std::vector<uint32_t> headPtrs(totalCells);
    PBuffer stagingHead(new Buffer(vd));
    stagingHead->create(totalCells * sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadWrite);
    
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy copyRegion(0, 0, totalCells * sizeof(uint32_t));
    vd->commandBuffer->copyBuffer(index->headPtrBuffer->buf, stagingHead->buf, copyRegion);
    vd->commandBuffer->end();
    vk::UniqueFence fence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence.get(), false);
    vd->device->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
    stagingHead->readData((char*)headPtrs.data(), totalCells * sizeof(uint32_t), 0);
    stagingHead->destroy();
    stagingHead.reset();
    
    // Read page buffer (need to read in chunks due to size)
    uint32_t maxPagesToRead = std::min((uint32_t)MAX_PAGES, 25000000u);  // Limit for memory
    std::vector<uint32_t> pageData(maxPagesToRead * PAGE_SIZE_UINTS);
    
    PBuffer stagingPages(new Buffer(vd));
    uint32_t bytesToRead = maxPagesToRead * PAGE_SIZE_UINTS * sizeof(uint32_t);
    stagingPages->create(bytesToRead, vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadWrite);
    
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy copyRegion2(0, 0, bytesToRead);
    vd->commandBuffer->copyBuffer(pageAlloc->pageBuffer->buf, stagingPages->buf, copyRegion2);
    vd->commandBuffer->end();
    vk::UniqueFence fence2 = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence2.get(), false);
    vd->device->waitForFences(fence2.get(), VK_TRUE, UINT64_MAX);
    stagingPages->readData((char*)pageData.data(), bytesToRead, 0);
    stagingPages->destroy();
    stagingPages.reset();
    
    // Traverse all linked lists
    uint32_t cellsWithPages = 0;
    for (uint32_t cell = 0; cell < totalCells; cell++) {
        uint32_t currentPtr = headPtrs[cell];
        uint32_t iterations = 0;
        const uint32_t MAX_ITERATIONS = 10000;
        
        while (currentPtr != NULL_PAGE_PTR && iterations < MAX_ITERATIONS) {
            iterations++;
            
            if (currentPtr >= maxPagesToRead) {
                // std::cerr << "[WARNING] Page index " << currentPtr << " exceeds read buffer\n";
                break;
            }
            
            uint32_t pageOffset = currentPtr * PAGE_SIZE_UINTS;
            uint32_t rowIdWithValid = pageData[pageOffset + 3];
            uint32_t nextPtr = pageData[pageOffset + PAGE_SIZE_UINTS - 1];  // nextPtr is last field
            
            if (rowIdWithValid & VALID_BIT_MASK) {
                validCount++;
            } else {
                invalidCount++;
            }
            
            // Safety check for self-loop
            if (nextPtr == currentPtr) {
                std::cerr << "[WARNING] Self-loop detected at page " << currentPtr << " in cell " << cell << "\n";
                break;
            }
            
            currentPtr = nextPtr;
        }
        
        if (headPtrs[cell] != NULL_PAGE_PTR) {
            cellsWithPages++;
        }
    }
    
    std::cerr << "[Traverse] Cells with pages: " << cellsWithPages << " / " << totalCells << "\n";
}

// ============================================================================
// Bitmap-based Pipeline Run Methods
// ============================================================================

void RasterScanIndexUpdate::runInsertBitmapPipeline(PBuffer pointsBuffer, PLinkedListIndex index, 
                                                     uint32_t npoints, uint32_t rowIdOffset) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRenderingLL(vd, bufs->dummyFbo, colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, insertBitmapPipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo headPtrDescriptor{index->headPtrBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo bitmapDescriptor{freeBitmap->bitmapBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo freeCountDescriptor{freeBitmap->freeCountBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo nextFreeDescriptor{freeBitmap->nextFreeHintBuffer->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{insertBitmapPipelineProps.descriptorSet.get(), 0, 0, 1, 
            vk::DescriptorType::eStorageBuffer, nullptr, &headPtrDescriptor},
        vk::WriteDescriptorSet{insertBitmapPipelineProps.descriptorSet.get(), 1, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
        vk::WriteDescriptorSet{insertBitmapPipelineProps.descriptorSet.get(), 2, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &bitmapDescriptor},
        vk::WriteDescriptorSet{insertBitmapPipelineProps.descriptorSet.get(), 3, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &freeCountDescriptor},
        vk::WriteDescriptorSet{insertBitmapPipelineProps.descriptorSet.get(), 4, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &nextFreeDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, 
        insertBitmapPipelineProps.pipelineLayout.get(), 0, insertBitmapPipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t, 8> consts = {
        index->minVal[0], index->minVal[1], 
        index->binRange[0], index->binRange[1], 
        INDEX_RESOLUTION, PAGE_DATA_SIZE, rowIdOffset, MAX_PAGES
    };
    vd->commandBuffer->pushConstants<uint32_t>(insertBitmapPipelineProps.pipelineLayout.get(),
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

void RasterScanIndexUpdate::runInsertBitmapV2Pipeline(PBuffer pointsBuffer, PLinkedListIndex index, 
                                                       uint32_t npoints, uint32_t rowIdOffset) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRenderingLL(vd, bufs->dummyFbo, colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, insertBitmapV2Pipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo headPtrDescriptor{index->headPtrBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo bitmapDescriptor{freeBitmap->bitmapBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo freeCountDescriptor{freeBitmap->freeCountBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo nextFreeDescriptor{freeBitmap->nextFreeHintBuffer->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{insertBitmapV2PipelineProps.descriptorSet.get(), 0, 0, 1, 
            vk::DescriptorType::eStorageBuffer, nullptr, &headPtrDescriptor},
        vk::WriteDescriptorSet{insertBitmapV2PipelineProps.descriptorSet.get(), 1, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
        vk::WriteDescriptorSet{insertBitmapV2PipelineProps.descriptorSet.get(), 2, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &bitmapDescriptor},
        vk::WriteDescriptorSet{insertBitmapV2PipelineProps.descriptorSet.get(), 3, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &freeCountDescriptor},
        vk::WriteDescriptorSet{insertBitmapV2PipelineProps.descriptorSet.get(), 4, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &nextFreeDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, 
        insertBitmapV2PipelineProps.pipelineLayout.get(), 0, insertBitmapV2PipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t, 8> consts = {
        index->minVal[0], index->minVal[1], 
        index->binRange[0], index->binRange[1], 
        INDEX_RESOLUTION, PAGE_DATA_SIZE, rowIdOffset, pageAlloc->maxPages
    };
    vd->commandBuffer->pushConstants<uint32_t>(insertBitmapV2PipelineProps.pipelineLayout.get(),
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

void RasterScanIndexUpdate::runDeleteByDataPipeline(PBuffer dataBuffer, PLinkedListIndex index, uint32_t ndeletes) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRenderingLL(vd, bufs->dummyFbo, colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, deleteByDataPipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo headPtrDescriptor{index->headPtrBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo bitmapDescriptor{freeBitmap->bitmapBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo freeCountDescriptor{freeBitmap->freeCountBuffer->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{deleteByDataPipelineProps.descriptorSet.get(), 0, 0, 1, 
            vk::DescriptorType::eStorageBuffer, nullptr, &headPtrDescriptor},
        vk::WriteDescriptorSet{deleteByDataPipelineProps.descriptorSet.get(), 1, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
        vk::WriteDescriptorSet{deleteByDataPipelineProps.descriptorSet.get(), 2, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &bitmapDescriptor},
        vk::WriteDescriptorSet{deleteByDataPipelineProps.descriptorSet.get(), 3, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &freeCountDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, 
        deleteByDataPipelineProps.pipelineLayout.get(), 0, deleteByDataPipelineProps.descriptorSet.get(), nullptr);

    // Push constants: minVal, binRange, res, maxPages
    std::array<uint32_t, 6> consts = {
        index->minVal[0], index->minVal[1], 
        index->binRange[0], index->binRange[1], 
        INDEX_RESOLUTION, pageAlloc->maxPages
    };
    vd->commandBuffer->pushConstants<uint32_t>(deleteByDataPipelineProps.pipelineLayout.get(),
        vk::ShaderStageFlagBits::eVertex, 0, consts);

    // Bind data buffer (x, y, z columns)
    vk::DeviceSize offset = 0;
    vd->commandBuffer->bindVertexBuffers(0, dataBuffer->buf, offset);
    offset += ndeletes * sizeof(uint32_t);
    vd->commandBuffer->bindVertexBuffers(1, dataBuffer->buf, offset);
    offset += ndeletes * sizeof(uint32_t);
    vd->commandBuffer->bindVertexBuffers(2, dataBuffer->buf, offset);

    vd->commandBuffer->draw(ndeletes, 1, 0, 0);
    vd->commandBuffer->endRendering();
}

void RasterScanIndexUpdate::runMarkFreePipeline(PLinkedListIndex index) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRenderingLL(vd, bufs->dummyFbo, colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, markFreePipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo bitmapDescriptor{freeBitmap->bitmapBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo freeCountDescriptor{freeBitmap->freeCountBuffer->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{markFreePipelineProps.descriptorSet.get(), 0, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
        vk::WriteDescriptorSet{markFreePipelineProps.descriptorSet.get(), 1, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &bitmapDescriptor},
        vk::WriteDescriptorSet{markFreePipelineProps.descriptorSet.get(), 2, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &freeCountDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, 
        markFreePipelineProps.pipelineLayout.get(), 0, markFreePipelineProps.descriptorSet.get(), nullptr);

    // Read current allocation counter using readData
    uint32_t allocCounter = 0;
    pageAlloc->allocCounterBuffer->readData((char*)&allocCounter, sizeof(uint32_t));

    std::array<uint32_t, 2> consts = {MAX_PAGES, allocCounter};
    vd->commandBuffer->pushConstants<uint32_t>(markFreePipelineProps.pipelineLayout.get(),
        vk::ShaderStageFlagBits::eVertex, 0, consts);

    // Draw ALL pages to ensure we catch those allocated by bitmap allocator
    vd->commandBuffer->draw(MAX_PAGES, 1, 0, 0);
    vd->commandBuffer->endRendering();
}

void RasterScanIndexUpdate::runCompactPipeline(PLinkedListIndex index) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRenderingLL(vd, bufs->dummyFbo, colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, compactPipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo headPtrDescriptor{index->headPtrBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo bitmapDescriptor{freeBitmap->bitmapBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo freeCountDescriptor{freeBitmap->freeCountBuffer->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{compactPipelineProps.descriptorSet.get(), 0, 0, 1, 
            vk::DescriptorType::eStorageBuffer, nullptr, &headPtrDescriptor},
        vk::WriteDescriptorSet{compactPipelineProps.descriptorSet.get(), 1, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
        vk::WriteDescriptorSet{compactPipelineProps.descriptorSet.get(), 2, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &bitmapDescriptor},
        vk::WriteDescriptorSet{compactPipelineProps.descriptorSet.get(), 3, 0, 1,
            vk::DescriptorType::eStorageBuffer, nullptr, &freeCountDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, 
        compactPipelineProps.pipelineLayout.get(), 0, compactPipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t, 1> consts = {INDEX_RESOLUTION};
    vd->commandBuffer->pushConstants<uint32_t>(compactPipelineProps.pipelineLayout.get(),
        vk::ShaderStageFlagBits::eVertex, 0, consts);

    // Draw one vertex per grid cell
    uint32_t totalCells = INDEX_RESOLUTION * INDEX_RESOLUTION;
    vd->commandBuffer->draw(totalCells, 1, 0, 0);
    vd->commandBuffer->endRendering();
}

void RasterScanIndexUpdate::setupVerifyIndexPipeline() {
    std::cerr << "setting up verify index pipeline for linked list index (Compute)\n";
    
    // 1. Descriptor Set Layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings.size(), bindings.data());
    verifyIndexDescSetLayout = vd->device->createDescriptorSetLayoutUnique(layoutInfo);
    
    // 2. Pipeline Layout
    vk::PushConstantRange pushConst(vk::ShaderStageFlagBits::eCompute, 0, sizeof(int));
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &verifyIndexDescSetLayout.get(), 1, &pushConst);
    verifyIndexPipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutInfo);
    
    // 3. Compute Pipeline
    vk::ComputePipelineCreateInfo pipelineInfo({}, 
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, verifyIndexCompShader.get(), "main"),
        verifyIndexPipelineLayout.get());
    
    // Result<Value> - use .value
    verifyIndexPipeline = vd->device->createComputePipelineUnique(nullptr, pipelineInfo).value;

    // 4. Descriptor Pool & Set
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 4}
    };
    vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes.size(), poolSizes.data());
    verifyIndexDescPool = vd->device->createDescriptorPoolUnique(poolInfo);
    
    vk::DescriptorSetAllocateInfo allocInfo(verifyIndexDescPool.get(), 1, &verifyIndexDescSetLayout.get());
    std::vector<vk::UniqueDescriptorSet> sets = vd->device->allocateDescriptorSetsUnique(allocInfo);
    verifyIndexDescSet = std::move(sets[0]);
}

void RasterScanIndexUpdate::runVerifyIndexPipeline(PLinkedListIndex index, vkcore::PBuffer resultBuffer, vkcore::PBuffer countBuffer) {
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, verifyIndexPipeline.get());

    vk::DescriptorBufferInfo headPtrDescriptor{index->headPtrBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo pageDescriptor{pageAlloc->pageBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo resultDescriptor{resultBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo countDescriptor{countBuffer->buf, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> descriptorWrites = {
        vk::WriteDescriptorSet{verifyIndexDescSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &headPtrDescriptor},
        vk::WriteDescriptorSet{verifyIndexDescSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &pageDescriptor},
        vk::WriteDescriptorSet{verifyIndexDescSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &resultDescriptor},
        vk::WriteDescriptorSet{verifyIndexDescSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorWrites, nullptr);
    
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, 
        verifyIndexPipelineLayout.get(), 0, 1, &verifyIndexDescSet.get(), 0, nullptr);

    int dim = INDEX_RESOLUTION;
    vd->commandBuffer->pushConstants<int>(verifyIndexPipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, dim);

    uint32_t totalCells = INDEX_RESOLUTION * INDEX_RESOLUTION;
    // Local size is 256
    uint32_t groupCountX = (totalCells + 255) / 256;
    vd->commandBuffer->dispatch(groupCountX, 1, 1);
}

void RasterScanIndexUpdate::verifyIndex(PLinkedListIndex index, vkcore::PBuffer resultBuffer, vkcore::PBuffer countBuffer) {
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vd->commandBuffer->begin(beginInfo);

    // Clear count buffer
    vd->commandBuffer->fillBuffer(countBuffer->buf, 0, VK_WHOLE_SIZE, 0);
    
    std::vector<vk::BufferMemoryBarrier> barriers;

    // Memory barrier to ensure clear is visible
    vk::BufferMemoryBarrier countBarrier;
    countBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    countBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    countBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    countBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    countBarrier.buffer = countBuffer->buf;
    countBarrier.offset = 0;
    countBarrier.size = VK_WHOLE_SIZE;
    barriers.push_back(countBarrier);
    
    // Page buffer visibility barrier (Compute -> Compute)
    vk::BufferMemoryBarrier pageBarrier;
    pageBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
    pageBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    pageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pageBarrier.buffer = pageAlloc->pageBuffer->buf;
    pageBarrier.offset = 0;
    pageBarrier.size = VK_WHOLE_SIZE;
    barriers.push_back(pageBarrier);
    
    // Head ptr visibility barrier
    vk::BufferMemoryBarrier headBarrier;
    headBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
    headBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    headBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    headBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    headBarrier.buffer = index->headPtrBuffer->buf;
    headBarrier.offset = 0;
    headBarrier.size = VK_WHOLE_SIZE;
    barriers.push_back(headBarrier);
    
    vd->commandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags(),
        0, nullptr,
        barriers.size(), barriers.data(),
        0, nullptr
    );

    runVerifyIndexPipeline(index, resultBuffer, countBuffer);
    
    vd->commandBuffer->end();
    
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->device->waitForFences(1, &fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
}
