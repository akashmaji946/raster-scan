// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "RasterScan2D.hpp"

#include <core/GPUTimer.hpp>
#include <cmath>

using namespace vkcore;

RasterScan2D::RasterScan2D(PVkDevice vd, PBufferCache bufs, SinglePassScan *scan, ReduceMax *reduce, int32_t ncols) {
    this->vd = vd;
    this->bufs = bufs;
    this->scan = scan;
    this->reduce = reduce;
    this->ncols = ncols;

    this->initalize();
}

RasterScan2D::~RasterScan2D() {
    maxBuffer.reset();
}

void RasterScan2D::initalize() {
    this->initShaders();
    this->initBuffers();

    this->setupBuildCountPipeline();
    this->setupBuildPipeline();
    this->setupRQTPipeline();
    this->setupRQEPipeline();
}

PRasterIndex RasterScan2D::buildIndex(PBuffer pointsBuffer, uint32_t npoints, uint32_t *minVal, uint32_t *maxVal) {
    PRasterIndex index(new IndexBuffers(vd, npoints, false, scan->getBufSizeDivisor()));
    index->minVal[0] = minVal[0];
    index->maxVal[0] = maxVal[0];
    index->binRange[0] = uint32_t(ceil(double(maxVal[0] - minVal[0])/ (INDEX_RESOLUTION)));
    index->minVal[1] = minVal[1];
    index->maxVal[1] = maxVal[1];
    index->binRange[1] = uint32_t(ceil(double(maxVal[1] - minVal[1])/ (INDEX_RESOLUTION)));

    GPUTimer countTimer, copyTimer, prefixTimer, buildTimer;

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());
    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    index->cstartBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eVertexShader);

    // first compute the layer count fbo
    countTimer.start(vd);
    this->buildHistogram(pointsBuffer,index);
    countTimer.stop();

    index->cstartBuffer->barrier(vk::PipelineStageFlagBits::eVertexShader,vk::PipelineStageFlagBits::eComputeShader,vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eShaderRead);
    reduce->reduce(index->cstartBuffer, maxBuffer, index->indexSize);
    index->cstartBuffer->barrier(vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eComputeShader,vk::AccessFlagBits::eShaderRead,vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

    // prefix sum to get offsets to write
    prefixTimer.start(vd);
    scan->prefixSum(index->cstartBuffer->buf,index->countBufSize);
    index->cstartBuffer->barrier(vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eTransfer,vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eTransferWrite);
    index->cendBuffer->copyFrom(index->countBufSize * sizeof(uint32_t),0,0,index->cstartBuffer);
    index->cendBuffer->barrier(vk::PipelineStageFlagBits::eTransfer,vk::PipelineStageFlagBits::eVertexShader,vk::AccessFlagBits::eTransferWrite,vk::AccessFlagBits::eShaderRead);
    prefixTimer.stop();

    // build hash table
    buildTimer.start(vd);
    this->build(pointsBuffer, index);
    buildTimer.stop();


    vd->commandBuffer->end();
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vd->submit(submitInfo,drawFence.get(),false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);

    maxBuffer->readData((char *)&(index->maxBinCt), sizeof(uint32_t));

    // TODO store timings

    return index;
}

void RasterScan2D::runRangeQueries(PRasterIndex index, PBuffer qranges, uint32_t nqueries) {
    vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->commandBuffer.get());

    vd->commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    maxBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eFragmentShader);
    this->runRQTPipeline(index,qranges,nqueries);
    maxBuffer->barrier(vk::PipelineStageFlagBits::eFragmentShader,vk::PipelineStageFlagBits::eDrawIndirect,vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eIndirectCommandRead);
    this->runRQEPipeline(index,qranges,nqueries);
    vd->commandBuffer->end();
    vd->submit(submitInfo,drawFence.get(),false);
    vd->device->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);
}

void RasterScan2D::initShaders() {
    {
        std::vector<uint32_t> fshader;
        validate(readShader(SHADER_FOLDER + "/dummy.frag.spv",fshader),"dummy 2d fragment shader");

        vk::ShaderModuleCreateInfo dummyFragmentModuleCreateInfo(vk::ShaderModuleCreateFlags(), fshader.size() * sizeof(uint32_t), fshader.data());
        fragmentShader = vd->device->createShaderModuleUnique(dummyFragmentModuleCreateInfo);
    }
    // build-count
    {
        std::vector<uint32_t> cvshader;
        validate(readShader(SHADER_FOLDER + "/build-count-2D.vert.spv",cvshader),"build count 2d vertex shader");

        vk::ShaderModuleCreateInfo countVertexModuleCreateInfo(vk::ShaderModuleCreateFlags(), cvshader.size() * sizeof(uint32_t), cvshader.data());
        bcVertexShader = vd->device->createShaderModuleUnique(countVertexModuleCreateInfo);

    }
    // build
    {
        std::vector<uint32_t> bvshader;
        validate(readShader(SHADER_FOLDER + "/build2D.vert.spv",bvshader),"build 2d vertex shader");

        vk::ShaderModuleCreateInfo buildVertexModuleCreateInfo(vk::ShaderModuleCreateFlags(), bvshader.size() * sizeof(uint32_t), bvshader.data());
        bVertexShader = vd->device->createShaderModuleUnique(buildVertexModuleCreateInfo);
    }
    // range query
    {
        std::vector<uint32_t> rqvshader, rqgshader, rqfshader;

        validate(readShader(SHADER_FOLDER + "/range2D.vert.spv",rqvshader),"range query 2d vertex shader");
        vk::ShaderModuleCreateInfo rqvModuleCreateInfo(vk::ShaderModuleCreateFlags(), rqvshader.size() * sizeof(uint32_t), rqvshader.data());
        rqVertexShader = vd->device->createShaderModuleUnique(rqvModuleCreateInfo);

        validate(readShader(SHADER_FOLDER + "/range2D.geom.spv",rqgshader),"range query 2d geometric shader");
        vk::ShaderModuleCreateInfo rqgModuleCreateInfo(vk::ShaderModuleCreateFlags(), rqgshader.size() * sizeof(uint32_t), rqgshader.data());
        rqGeomShader = vd->device->createShaderModuleUnique(rqgModuleCreateInfo);

        validate(readShader(SHADER_FOLDER + "/range2D.frag.spv",rqfshader),"range query 2d frag shader");
        vk::ShaderModuleCreateInfo rqfModuleCreateInfo(vk::ShaderModuleCreateFlags(), rqfshader.size() * sizeof(uint32_t), rqfshader.data());
        rqFragShader = vd->device->createShaderModuleUnique(rqfModuleCreateInfo);
    }
    // edges for query
    {
        std::vector<uint32_t> evshader, egshader, efshader;

        validate(readShader(SHADER_FOLDER + "/edge2D.vert.spv",evshader),"range query 2d edge vertex shader");
        vk::ShaderModuleCreateInfo evModuleCreateInfo(vk::ShaderModuleCreateFlags(), evshader.size() * sizeof(uint32_t), evshader.data());
        eVertexShader = vd->device->createShaderModuleUnique(evModuleCreateInfo);

        validate(readShader(SHADER_FOLDER + "/edge2D.geom.spv",egshader),"range query 2d  edge geometric shader");
        vk::ShaderModuleCreateInfo egModuleCreateInfo(vk::ShaderModuleCreateFlags(), egshader.size() * sizeof(uint32_t), egshader.data());
        eGeomShader = vd->device->createShaderModuleUnique(egModuleCreateInfo);

        validate(readShader(SHADER_FOLDER + "/edge2D.frag.spv",efshader),"range query 2d  edge frag shader");
        vk::ShaderModuleCreateInfo efModuleCreateInfo(vk::ShaderModuleCreateFlags(), efshader.size() * sizeof(uint32_t), efshader.data());
        eFragShader = vd->device->createShaderModuleUnique(efModuleCreateInfo);
    }

}

void RasterScan2D::initBuffers() {
    maxBuffer.reset(new Buffer(vd));
    maxBuffer->create(4 * sizeof(uint32_t),vk::BufferUsageFlagBits::eIndirectBuffer|vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc,MemoryType::LocalHostVisibleForce);
}

void RasterScan2D::setupBuildCountPipeline() {
    std::cerr << "setting up build count pipeline\n";
    bcPipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, bcVertexShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, fragmentShader.get(), "main")
    };
    bcPipelineProps.setShaderStageFlag();

    bcPipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription (0, sizeof(uint32_t)),
        vk::VertexInputBindingDescription (1, sizeof(uint32_t))
    };
    bcPipelineProps.setInputBindingFlag();

    bcPipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),  // coordinates
        vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32Uint, 0),  // coordinates
    };
    bcPipelineProps.setInputAttrFlag();

    bcPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::ePointList);
    bcPipelineProps.setInputAssemblyFlag();

    bcPipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };

    bcPipelineProps.poolSizes = {
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
    };

    bcPipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex,0,sizeof(uint32_t) * 5)
    };
    bcPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPassBC;
    bcPipeline = bcPipelineProps.createPipeline(vd, dummyRenderPassBC, &rpCreateInfo);
}

void RasterScan2D::setupBuildPipeline() {
    std::cerr << "setting up build pipeline\n";
    bPipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, bVertexShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, fragmentShader.get(), "main")
    };
    bPipelineProps.setShaderStageFlag();

    bPipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription (0, sizeof(uint32_t)),
        vk::VertexInputBindingDescription (1, sizeof(uint32_t)),
        vk::VertexInputBindingDescription (2, sizeof(uint32_t)),
    };
    bPipelineProps.setInputBindingFlag();

    bPipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32Uint, 0),  // coordinates
        vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32Uint, 0),  // coordinates
        vk::VertexInputAttributeDescription(2, 2, vk::Format::eR32Uint, 0),  // coordinates
    };
    bPipelineProps.setInputAttrFlag();

    bPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::ePointList);
    bPipelineProps.setInputAssemblyFlag();

    bPipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    };

    bPipelineProps.poolSizes = {
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1}
    };

    bPipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex,0,sizeof(int32_t) * 5)
    };
    bPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPassB;
    bPipeline = bPipelineProps.createPipeline(vd, dummyRenderPassB, &rpCreateInfo);
}

void RasterScan2D::setupRQTPipeline() {
    std::cerr << "setting up range query triangle pipeline\n";
    rqtPipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, rqVertexShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, rqFragShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eGeometry, rqGeomShader.get(), "main"),
    };
    rqtPipelineProps.setShaderStageFlag();

    rqtPipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription (0, 4 * sizeof(uint32_t)),
        vk::VertexInputBindingDescription (1, 2 * sizeof(uint32_t)),
    };
    rqtPipelineProps.setInputBindingFlag();

    rqtPipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32A32Uint, 0),  // coordinates
        vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32G32Uint, 0),  // coordinates
    };
    rqtPipelineProps.setInputAttrFlag();

    rqtPipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::ePointList);
    rqtPipelineProps.setInputAssemblyFlag();

    rqtPipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{ 3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{ 4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
    };

    rqtPipelineProps.poolSizes = {
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1}
    };

    rqtPipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex|vk::ShaderStageFlagBits::eFragment,0,sizeof(int32_t) * 5)
    };
    rqtPipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPassRQT;
    rqtPipeline = rqtPipelineProps.createPipeline(vd, dummyRenderPassRQT, &rpCreateInfo);
}

void RasterScan2D::setupRQEPipeline() {
    std::cerr << "setting up range query triangle pipeline\n";
    rqePipelineProps.pipelineShaderStageCreateInfos = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, eVertexShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, eFragShader.get(), "main"),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eGeometry, eGeomShader.get(), "main"),
    };
    rqePipelineProps.setShaderStageFlag();

    rqePipelineProps.vertexInputBindingDescriptions = {
        vk::VertexInputBindingDescription (0, 2 * sizeof(uint32_t)),
    };
    rqePipelineProps.setInputBindingFlag();

    rqePipelineProps.vertexInputAttributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Uint, 0),  // coordinates
    };
    rqePipelineProps.setInputAttrFlag();

    rqePipelineProps.pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::ePointList);
    rqePipelineProps.setInputAssemblyFlag();

    rqePipelineProps.setLayoutBindings = {
        vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
    };

    rqePipelineProps.poolSizes = {
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
        vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1},
    };

    rqePipelineProps.pushConstantRange = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex|vk::ShaderStageFlagBits::eFragment,0,2 * sizeof(int32_t))
    };
    rqePipelineProps.setBlendFunction(BlendFunc::BLEND_NONE);

    vk::PipelineRenderingCreateInfo rpCreateInfo;
    rpCreateInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = vk::Format::eR8Sint;
    rpCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::UniqueRenderPass dummyRenderPassRQE;
    rqePipeline = rqePipelineProps.createPipeline(vd, dummyRenderPassRQE, &rpCreateInfo);

}

inline vk::RenderingInfo setupRendering(PVkDevice vd, PFrameBuffer fbo, vk::RenderingAttachmentInfo &colorInfo) {
    colorInfo.imageView = fbo->colorView;
    colorInfo.imageLayout = vk::ImageLayout::eGeneral;
    colorInfo.loadOp = vk::AttachmentLoadOp::eDontCare;

    int width = INDEX_RESOLUTION; int height = INDEX_RESOLUTION;
    vk::Rect2D renderArea(vk::Offset2D(0, 0), vk::Extent2D(width, height));
    vk::RenderingInfo renderingInfo;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorInfo;
    renderingInfo.renderArea = renderArea;
    renderingInfo.layerCount = 1;

    vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
    vk::Rect2D scissor(vk::Offset2D(0, 0), vk::Extent2D(width,height));
    vd->commandBuffer->setViewport(0,1,&viewport);
    vd->commandBuffer->setScissor(0,1,&scissor);

    return renderingInfo;
}

void RasterScan2D::buildHistogram(vkcore::PBuffer pointsBuffer, PRasterIndex index) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRendering(vd,bufs->dummyFbo,colorInfo);

    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, bcPipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo countDescriptor{ index->cstartBuffer->buf, 0, VK_WHOLE_SIZE };

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{ bcPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bcPipelineProps.pipelineLayout.get(), 0, bcPipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t,5> consts = {index->minVal[0], index->minVal[1], index->binRange[0], index->binRange[1], INDEX_RESOLUTION};
    vd->commandBuffer->pushConstants<uint32_t>(bcPipelineProps.pipelineLayout.get(),vk::ShaderStageFlagBits::eVertex,0,consts);

    vk::DeviceSize offset = 0;
    vd->commandBuffer->bindVertexBuffers(0, pointsBuffer->buf, offset);
    offset += index->npoints * sizeof(uint32_t);
    vd->commandBuffer->bindVertexBuffers(1, pointsBuffer->buf, offset);

    vd->commandBuffer->draw(index->npoints,1,0,0);
    vd->commandBuffer->endRendering();
}

void RasterScan2D::build(vkcore::PBuffer pointsBuffer, PRasterIndex index) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRendering(vd,bufs->dummyFbo,colorInfo);
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, bPipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo countDescriptor{ index->cendBuffer->buf, 0, VK_WHOLE_SIZE };
    vk::DescriptorBufferInfo indexDescriptor{ index->indexBuffer->buf, 0, VK_WHOLE_SIZE };

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{ bPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &countDescriptor},
        vk::WriteDescriptorSet{ bPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &indexDescriptor},
    };
    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bPipelineProps.pipelineLayout.get(), 0, bPipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t,5> consts = {index->minVal[0], index->minVal[1], index->binRange[0], index->binRange[1], INDEX_RESOLUTION};
    vd->commandBuffer->pushConstants<uint32_t>(bPipelineProps.pipelineLayout.get(),vk::ShaderStageFlagBits::eVertex,0,consts);

    vk::DeviceSize offset = 0;
    vd->commandBuffer->bindVertexBuffers(0, pointsBuffer->buf, offset);
    offset += index->npoints * sizeof(uint32_t);
    vd->commandBuffer->bindVertexBuffers(1, pointsBuffer->buf, offset);
    // hack: use 2nd column as 3rd column if the index is on 2 columns
    if(ncols == 3) {
        offset += index->npoints * sizeof(uint32_t);
    }
    vd->commandBuffer->bindVertexBuffers(2, pointsBuffer->buf, offset);

    vd->commandBuffer->draw(index->npoints,1,0,0);
    vd->commandBuffer->endRendering();
}

void RasterScan2D::runRQTPipeline(PRasterIndex index, vkcore::PBuffer qranges, uint32_t nqueries) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRendering(vd,bufs->dummyFbo,colorInfo);
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, rqtPipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo cstartDescriptor{ index->cstartBuffer->buf, 0, VK_WHOLE_SIZE };
    vk::DescriptorBufferInfo cendDescriptor{ index->cendBuffer->buf, 0, VK_WHOLE_SIZE };
    vk::DescriptorBufferInfo rctDescriptor{ maxBuffer->buf, 0, VK_WHOLE_SIZE };
    vk::DescriptorBufferInfo edgeDescriptor{ bufs->edgeBuffer->buf, 0, VK_WHOLE_SIZE };

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{ rqtPipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cstartDescriptor},
        vk::WriteDescriptorSet{ rqtPipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &cendDescriptor},
        vk::WriteDescriptorSet{ rqtPipelineProps.descriptorSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &rctDescriptor},
        vk::WriteDescriptorSet{ rqtPipelineProps.descriptorSet.get(), 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &edgeDescriptor},
    };

    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, rqtPipelineProps.pipelineLayout.get(), 0, rqtPipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t,5> consts = {index->minVal[0], index->minVal[1], index->binRange[0], index->binRange[1], INDEX_RESOLUTION};
    vd->commandBuffer->pushConstants<uint32_t>(rqtPipelineProps.pipelineLayout.get(),vk::ShaderStageFlagBits::eVertex|vk::ShaderStageFlagBits::eFragment,0,consts);

    vk::DeviceSize offset = 0;
    vd->commandBuffer->bindVertexBuffers(0, qranges->buf, offset);
    offset += nqueries * 4 * sizeof(uint32_t);
    vd->commandBuffer->bindVertexBuffers(1, qranges->buf, offset);

    vd->commandBuffer->draw(nqueries,1,0,0);
    vd->commandBuffer->endRendering();
}

void RasterScan2D::runRQEPipeline(PRasterIndex index, vkcore::PBuffer qranges, uint32_t nqueries) {
    vk::RenderingAttachmentInfo colorInfo;
    vk::RenderingInfo renderingInfo = setupRendering(vd,bufs->dummyFbo,colorInfo);
    vd->commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, rqePipeline.get());
    vd->commandBuffer->beginRendering(&renderingInfo);

    vk::DescriptorBufferInfo indexDescriptor{ index->indexBuffer->buf, 0, VK_WHOLE_SIZE };
    vk::DescriptorBufferInfo resDescriptor{ bufs->resBuffer->buf, 0, VK_WHOLE_SIZE };
    vk::DescriptorBufferInfo rangeDescriptor{ qranges->buf, 0, VK_WHOLE_SIZE };

    std::vector<vk::WriteDescriptorSet> descriptorSets = {
        vk::WriteDescriptorSet{ rqePipelineProps.descriptorSet.get(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &indexDescriptor},
        vk::WriteDescriptorSet{ rqePipelineProps.descriptorSet.get(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &resDescriptor},
        vk::WriteDescriptorSet{ rqePipelineProps.descriptorSet.get(), 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &rangeDescriptor},
    };

    vd->device->updateDescriptorSets(descriptorSets, nullptr);
    vd->commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, rqePipelineProps.pipelineLayout.get(), 0, rqePipelineProps.descriptorSet.get(), nullptr);

    std::array<uint32_t,2> consts = {INDEX_RESOLUTION, static_cast<uint32_t>(ncols)};
    vd->commandBuffer->pushConstants<uint32_t>(rqePipelineProps.pipelineLayout.get(),vk::ShaderStageFlagBits::eVertex|vk::ShaderStageFlagBits::eFragment,0,consts);

    vk::DeviceSize offset = 0;
    vd->commandBuffer->bindVertexBuffers(0, bufs->edgeBuffer->buf, offset);
    vd->commandBuffer->drawIndirect(maxBuffer->buf,0,1,4 * sizeof(uint32_t));
    vd->commandBuffer->endRendering();
}


