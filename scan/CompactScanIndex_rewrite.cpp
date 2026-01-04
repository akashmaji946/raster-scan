#include "CompactScanIndex.hpp"
#include <common/utils.h>
#include <iostream>
#include <cmath>

using namespace vkcore;

// Helper function for graphics pipeline rendering (same as RasterScan2D)
inline vk::RenderingInfo setupRendering(PVkDevice vd, PFrameBuffer fbo, vk::RenderingAttachmentInfo &colorInfo) {
    colorInfo.imageView = fbo->colorView;
    colorInfo.imageLayout = vk::ImageLayout::eGeneral;
    colorInfo.loadOp = vk::AttachmentLoadOp::eDontCare;
    colorInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
    
    vk::Rect2D renderArea({0, 0}, {INDEX_RESOLUTION, INDEX_RESOLUTION});
    vk::RenderingInfo renderingInfo;
    renderingInfo.renderArea = renderArea;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorInfo;
    
    vk::Viewport viewport(0, 0, INDEX_RESOLUTION, INDEX_RESOLUTION, 0.0f, 1.0f);
    vk::Rect2D scissor({0, 0}, {INDEX_RESOLUTION, INDEX_RESOLUTION});
    vd->commandBuffer->setViewport(0, viewport);
    vd->commandBuffer->setScissor(0, scissor);
    
    return renderingInfo;
}

CompactScanIndex::CompactScanIndex(PVkDevice vd, int32_t ncols, SinglePassScan* scan)
    : vd(vd), ncols(ncols), scan(scan), npoints(0) {
}

CompactScanIndex::~CompactScanIndex() {
    if(dummyFbo) { dummyFbo->destroy(); dummyFbo.reset(); }
    if(cstartBuffer) { cstartBuffer->destroy(); cstartBuffer.reset(); }
    if(cendBuffer) { cendBuffer->destroy(); cendBuffer.reset(); }
    if(dataBuffer) { dataBuffer->destroy(); dataBuffer.reset(); }
}

void CompactScanIndex::initialize() {
    setupPipelines();
}

void CompactScanIndex::allocateBuffers(uint32_t npoints) {
    this->npoints = npoints;
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    
    size_t scanBufSize = scan ? scan->getBufSizeDivisor() : 4096;
    countBufSize = size_t(std::ceil(double(totalBins + 1) / scanBufSize) * scanBufSize);
    
    // cstartBuffer: prefix sum result (read-only after build)
    cstartBuffer = std::make_shared<Buffer>(vd);
    cstartBuffer->create(countBufSize * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::Internal);

    // cendBuffer: working offsets (atomicAdd during insert)
    cendBuffer = std::make_shared<Buffer>(vd);
    cendBuffer->create(countBufSize * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::Internal);

    // dataBuffer: CompactEntry array (4 uints per entry = 16 bytes)
    uint32_t dataSize = npoints * COMPACT_SCALE_FACTOR;
    dataBuffer = std::make_shared<Buffer>(vd);
    dataBuffer->create(dataSize * sizeof(CompactEntry), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::Internal);
    
    std::cout << "[CompactIndex] Buffers: cstart=" << (countBufSize*4/1024) << "KB, cend=" 
              << (countBufSize*4/1024) << "KB, data=" << (dataSize*16/1024/1024) << "MB\n";
}

void CompactScanIndex::setupPipelines() {
    // Create dummy FBO (same as RasterScan2D)
    dummyFbo = std::make_shared<FrameBuffer>(vd);
    dummyFbo->create(INDEX_RESOLUTION, INDEX_RESOLUTION, vk::Format::eR8Sint, false);

    std::vector<uint32_t> shader;
    
    // ========== DUMMY FRAGMENT SHADER ==========
    validate(readShader(SHADER_FOLDER + "/dummy.frag.spv", shader), "dummy fragment shader");
    fragmentShader = vd->device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, shader.size() * sizeof(uint32_t), shader.data()));

    // ========== BUILD COUNT PIPELINE (like RasterScan2D) ==========
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/compact_count_gfx.vert.spv", vshader), "compact count vertex shader");
        bcVertexShader = vd->device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, vshader.size() * sizeof(uint32_t), vshader.data()));
        
        bcPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, bcVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, fragmentShader.get(), "main")
        };
        bcPipelineProps.setShaderStageFlag();
        
        // 2 vertex bindings: X, Y (Z not needed for count)
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
        bcPipelineProps.poolSizes = { vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1} };
        bcPipelineProps.pushConstantRange = { vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, 5 * sizeof(uint32_t)) };
        bcPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);
        
        vk::PipelineRenderingCreateInfo rpInfo;
        rpInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpInfo.pColorAttachmentFormats = &colorFormat;
        bcPipelineProps.pipelineRenderingCreateInfo = rpInfo;
        bcPipelineProps.setRenderingFlag();
        
        bcPipelineProps.createDescriptorSetLayout(vd);
        bcPipelineProps.createPipelineLayout(vd);
        bcPipelineProps.createDescriptorPool(vd);
        bcPipelineProps.createDescriptorSet(vd);
        bcPipeline = bcPipelineProps.createPipeline(vd);
    }

    // ========== BUILD INSERT PIPELINE (like RasterScan2D) ==========
    {
        std::vector<uint32_t> vshader;
        validate(readShader(SHADER_FOLDER + "/compact_build_gfx.vert.spv", vshader), "compact build vertex shader");
        bVertexShader = vd->device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, vshader.size() * sizeof(uint32_t), vshader.data()));
        
        bPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, bVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, fragmentShader.get(), "main")
        };
        bPipelineProps.setShaderStageFlag();
        
        // 3 vertex bindings: X, Y, Z
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
        bPipelineProps.poolSizes = { vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 2} };
        bPipelineProps.pushConstantRange = { vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, 5 * sizeof(uint32_t)) };
        bPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);
        
        vk::PipelineRenderingCreateInfo rpInfo;
        rpInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpInfo.pColorAttachmentFormats = &colorFormat;
        bPipelineProps.pipelineRenderingCreateInfo = rpInfo;
        bPipelineProps.setRenderingFlag();
        
        bPipelineProps.createDescriptorSetLayout(vd);
        bPipelineProps.createPipelineLayout(vd);
        bPipelineProps.createDescriptorPool(vd);
        bPipelineProps.createDescriptorSet(vd);
        bPipeline = bPipelineProps.createPipeline(vd);
    }

    // ========== QUERY PIPELINE ==========
    {
        std::vector<uint32_t> vshader, gshader, fshader;
        validate(readShader(SHADER_FOLDER + "/compact_query_gfx.vert.spv", vshader), "compact query vertex shader");
        validate(readShader(SHADER_FOLDER + "/compact_query_gfx.geom.spv", gshader), "compact query geometry shader");
        validate(readShader(SHADER_FOLDER + "/compact_query_gfx.frag.spv", fshader), "compact query fragment shader");
        
        queryVertexShader = vd->device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, vshader.size() * sizeof(uint32_t), vshader.data()));
        queryGeomShader = vd->device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, gshader.size() * sizeof(uint32_t), gshader.data()));
        queryFragShader = vd->device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, fshader.size() * sizeof(uint32_t), fshader.data()));
        
        queryPipelineProps.pipelineShaderStageCreateInfos = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, queryVertexShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eGeometry, queryGeomShader.get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, queryFragShader.get(), "main")
        };
        queryPipelineProps.setShaderStageFlag();
        
        queryPipelineProps.vertexInputBindingDescriptions = {
            vk::VertexInputBindingDescription(0, 6 * sizeof(uint32_t), vk::VertexInputRate::eInstance)
        };
        queryPipelineProps.setInputBindingFlag();
        
        queryPipelineProps.vertexInputAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Uint, 0),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Uint, 3 * sizeof(uint32_t))
        };
        queryPipelineProps.setInputAttrFlag();
        
        queryPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::ePointList);
        queryPipelineProps.setInputAssemblyFlag();
        
        queryPipelineProps.setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}
        };
        queryPipelineProps.poolSizes = { vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 3} };
        queryPipelineProps.pushConstantRange = { 
            vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, 8 * sizeof(uint32_t)) 
        };
        queryPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);
        
        vk::PipelineRenderingCreateInfo rpInfo;
        rpInfo.colorAttachmentCount = 1;
        vk::Format colorFormat = vk::Format::eR8Sint;
        rpInfo.pColorAttachmentFormats = &colorFormat;
        queryPipelineProps.pipelineRenderingCreateInfo = rpInfo;
        queryPipelineProps.setRenderingFlag();
        
        queryPipelineProps.createDescriptorSetLayout(vd);
        queryPipelineProps.createPipelineLayout(vd);
        queryPipelineProps.createDescriptorPool(vd);
        queryPipelineProps.createDescriptorSet(vd);
        queryPipeline = queryPipelineProps.createPipeline(vd);
    }

    // ========== COMPUTE PIPELINES (for delete/insert) ==========
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)
        };
        descSetLayout = vd->device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, bindings));
        
        vk::PushConstantRange pcRange(vk::ShaderStageFlagBits::eCompute, 0, 12 * sizeof(uint32_t));
        pipelineLayout = vd->device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo({}, descSetLayout.get(), pcRange));
        
        std::vector<vk::DescriptorPoolSize> poolSizes = { vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 4) };
        descPool = vd->device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes));
        descSet = std::move(vd->device->allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(descPool.get(), descSetLayout.get()))[0]);
        
        std::vector<uint32_t> delShader, insShader;
        validate(readShader(SHADER_FOLDER + "/compact_delete.comp.spv", delShader), "compact delete shader");
        validate(readShader(SHADER_FOLDER + "/compact_insert.comp.spv", insShader), "compact insert shader");
        
        deleteShader = vd->device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, delShader.size() * sizeof(uint32_t), delShader.data()));
        insertShader = vd->device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, insShader.size() * sizeof(uint32_t), insShader.data()));
        
        vk::PipelineShaderStageCreateInfo delStage({}, vk::ShaderStageFlagBits::eCompute, deleteShader.get(), "main");
        vk::PipelineShaderStageCreateInfo insStage({}, vk::ShaderStageFlagBits::eCompute, insertShader.get(), "main");
        
        deletePipeline = vd->device->createComputePipelineUnique({}, vk::ComputePipelineCreateInfo({}, delStage, pipelineLayout.get())).value;
        insertPipeline = vd->device->createComputePipelineUnique({}, vk::ComputePipelineCreateInfo({}, insStage, pipelineLayout.get())).value;
    }
}

void CompactScanIndex::buildIndex(PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal) {
    allocateBuffers(npoints);
    
    for(int i = 0; i < 3; i++) {
        this->minVal[i] = minVal[i];
        this->maxVal[i] = maxVal[i];
        uint32_t range = maxVal[i] - minVal[i];
        binWidth[i] = (range + INDEX_RESOLUTION - 1) / INDEX_RESOLUTION;
        if(binWidth[i] == 0) binWidth[i] = 1;
    }
    
    // Push constants: minVal[2], binWidth[2], resolution (like RasterScan2D)
    std::array<uint32_t, 5> pc = {minVal[0], minVal[1], binWidth[0], binWidth[1], INDEX_RESOLUTION};
    
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    // Clear cstartBuffer
    cstartBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader);
    
    // ========== PASS 1: Count ==========
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, bcPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        
        vk::DescriptorBufferInfo countDesc{cstartBuffer->buf, 0, VK_WHOLE_SIZE};
        vd->device->updateDescriptorSets({
            vk::WriteDescriptorSet{bcPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDesc}
        }, nullptr);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bcPipelineProps.pipelineLayout.get(), 0, bcPipelineProps.descriptorSet.get(), nullptr);
        vd->commandBuffer->pushConstants<uint32_t>(bcPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, pc);
        
        vk::DeviceSize offset = 0;
        vd->commandBuffer->bindVertexBuffers(0, pointsBuffer->buf, offset);
        offset += npoints * sizeof(uint32_t);
        vd->commandBuffer->bindVertexBuffers(1, pointsBuffer->buf, offset);
        
        vd->commandBuffer->draw(npoints, 1, 0, 0);
        vd->commandBuffer->endRendering();
    }
    
    // Barrier: Count -> Prefix Sum
    cstartBuffer->barrier(vk::PipelineStageFlagBits::eVertexShader, vk::PipelineStageFlagBits::eComputeShader,
                          vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    
    // ========== PASS 2: Prefix Sum ==========
    scan->prefixSum(cstartBuffer->buf, countBufSize);
    
    // Barrier: Prefix Sum -> Copy
    cstartBuffer->barrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                          vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead);
    
    // Copy to cendBuffer (for insert atomicAdd)
    cendBuffer->copyFrom(countBufSize * sizeof(uint32_t), 0, 0, cstartBuffer);
    
    // Barrier: Copy -> Insert
    cendBuffer->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eVertexShader,
                        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    
    // ========== PASS 3: Insert ==========
    {
        vk::RenderingAttachmentInfo colorInfo;
        vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
        vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, bPipeline.get());
        vd->commandBuffer->beginRendering(&renderingInfo);
        
        vk::DescriptorBufferInfo countDesc{cendBuffer->buf, 0, VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo dataDesc{dataBuffer->buf, 0, VK_WHOLE_SIZE};
        vd->device->updateDescriptorSets({
            vk::WriteDescriptorSet{bPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDesc},
            vk::WriteDescriptorSet{bPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dataDesc}
        }, nullptr);
        vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bPipelineProps.pipelineLayout.get(), 0, bPipelineProps.descriptorSet.get(), nullptr);
        vd->commandBuffer->pushConstants<uint32_t>(bPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, pc);
        
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
    vk::UniqueFence fence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence.get(), false);
    vd->device->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
}

void CompactScanIndex::runRangeQueries(PBuffer queryBuffer, uint32_t nqueries, PBuffer resultBuffer) {
    std::array<uint32_t, 8> pc = {minVal[0], minVal[1], minVal[2], INDEX_RESOLUTION, 
                                   binWidth[0], binWidth[1], binWidth[2], nqueries};
    
    vk::DescriptorBufferInfo cstartDesc{cstartBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo dataDesc{dataBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo resultDesc{resultBuffer->buf, 0, VK_WHOLE_SIZE};
    
    vd->device->updateDescriptorSets({
        vk::WriteDescriptorSet{queryPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cstartDesc},
        vk::WriteDescriptorSet{queryPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dataDesc},
        vk::WriteDescriptorSet{queryPipelineProps.descriptorSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &resultDesc}
    }, nullptr);
    
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRendering(vd, dummyFbo, colorInfo);
    
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, queryPipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, queryPipelineProps.pipelineLayout.get(), 0, queryPipelineProps.descriptorSet.get(), nullptr);
    vd->commandBuffer->pushConstants<uint32_t>(queryPipelineProps.pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
    
    vd->commandBuffer->bindVertexBuffers(0, queryBuffer->buf, vk::DeviceSize(0));
    vd->commandBuffer->draw(1, nqueries, 0, 0);
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

void CompactScanIndex::deletePoints(PBuffer deleteDataBuffer, uint32_t ndeletes) {
    vk::DescriptorBufferInfo cstartDesc{cstartBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo cendDesc{cendBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo dataDesc{dataBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo inputDesc{deleteDataBuffer->buf, 0, VK_WHOLE_SIZE};
    
    vd->device->updateDescriptorSets({
        vk::WriteDescriptorSet{descSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cstartDesc},
        vk::WriteDescriptorSet{descSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cendDesc},
        vk::WriteDescriptorSet{descSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dataDesc},
        vk::WriteDescriptorSet{descSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &inputDesc}
    }, nullptr);
    
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, deletePipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descSet.get(), 0, nullptr);
    
    uint32_t pc[12] = {minVal[0], minVal[1], minVal[2], INDEX_RESOLUTION, maxVal[0], maxVal[1], maxVal[2], ndeletes, binWidth[0], binWidth[1], binWidth[2], 0};
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, 12 * sizeof(uint32_t), pc);
    vd->commandBuffer->dispatch((ndeletes + 255) / 256, 1, 1);
    vd->commandBuffer->end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
}

void CompactScanIndex::insertPoints(PBuffer pointsBuffer, uint32_t npoints) {
    vk::DescriptorBufferInfo cstartDesc{cstartBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo cendDesc{cendBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo dataDesc{dataBuffer->buf, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo inputDesc{pointsBuffer->buf, 0, VK_WHOLE_SIZE};
    
    vd->device->updateDescriptorSets({
        vk::WriteDescriptorSet{descSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cstartDesc},
        vk::WriteDescriptorSet{descSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cendDesc},
        vk::WriteDescriptorSet{descSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dataDesc},
        vk::WriteDescriptorSet{descSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &inputDesc}
    }, nullptr);
    
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, insertPipeline.get());
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descSet.get(), 0, nullptr);
    
    uint32_t pc[12] = {minVal[0], minVal[1], minVal[2], INDEX_RESOLUTION, maxVal[0], maxVal[1], maxVal[2], npoints, binWidth[0], binWidth[1], binWidth[2], 0};
    vd->commandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, 12 * sizeof(uint32_t), pc);
    vd->commandBuffer->dispatch((npoints + 255) / 256, 1, 1);
    vd->commandBuffer->end();
    
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vd->commandBuffer.get();
    vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
    vd->submit(submitInfo, fence, false);
    vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
    vd->device->destroyFence(fence);
}
