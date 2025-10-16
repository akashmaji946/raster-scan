// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "GraphicsPipelineProperties.hpp"

#include <iostream>

namespace vkcore {

GraphicsPipelineProperties::GraphicsPipelineProperties() {
    // flags to ensure required properties are set
    shaderStageFlag = false;
    inputBindingFlag = false;
    inputAttrFlag = false;
    inputAssemblyFlag = false;

    setLayoutBindings.clear();
    pushConstantRange.clear();

    // fill rest with default valaues
    pipelineViewportStateCreateInfo = vk::PipelineViewportStateCreateInfo(
        vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1, nullptr
    );

    pipelineRasterizationStateCreateInfo = vk::PipelineRasterizationStateCreateInfo(
        vk::PipelineRasterizationStateCreateFlags(),  // flags
        false,                                        // depthClampEnable
        false,                                         // rasterizerDiscardEnable
        vk::PolygonMode::eFill,                       // polygonMode
        vk::CullModeFlagBits::eNone,                  // cullMode
        vk::FrontFace::eClockwise,                    // frontFace
        false,                                        // depthBiasEnable
        0.0f,                                         // depthBiasConstantFactor
        0.0f,                                         // depthBiasClamp
        0.0f,                                         // depthBiasSlopeFactor
        1.0f                                          // lineWidth
    );

    pipelineDepthStencilStateCreateInfo = vk::PipelineDepthStencilStateCreateInfo(
        vk::PipelineDepthStencilStateCreateFlags(), // flags
        false                                      // depthTestEnable
    );

    colorComponentFlags = vk::ColorComponentFlags(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

    pipelineColorBlendAttachmentState = vk::PipelineColorBlendAttachmentState(
        VK_FALSE                        // blendEnable
    );
    pipelineColorBlendAttachmentState.colorWriteMask = colorComponentFlags;

    dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };

    pipelineTessellationStateCreateInfo = nullptr;

    logicalOpEnable = VK_FALSE;
    logicalOp = vk::LogicOp::eNoOp;
    blendAttachmentCount = 1;
}

vk::UniquePipeline GraphicsPipelineProperties::createPipeline(PVkDevice vd, vk::UniqueRenderPass &renderPass, vk::PipelineRenderingCreateInfo *renderingPipelineCreateInfo) {
    if(!shaderStageFlag) {
        std::cerr << "shader stages not set" << std::endl;
        exit(-3);
    }
    if(!inputBindingFlag) {
        std::cerr << "shader input bindings not set" << std::endl;
        exit(-3);
    }
    if(!inputAttrFlag) {
        std::cerr << "shader input attributes not set" << std::endl;
        exit(-3);
    }
    if(!inputAssemblyFlag) {
        std::cerr << "shader input assembly not set" << std::endl;
        exit(-3);
    }


    // create the derived properties variables
    pipelineVertexInputStateCreateInfo = vk::PipelineVertexInputStateCreateInfo (
        vk::PipelineVertexInputStateCreateFlags(),  // flags
        (uint32_t)vertexInputBindingDescriptions.size(),      // vertexBindingDescriptionCount
        vertexInputBindingDescriptions.data(),      // pVertexBindingDescription
        (uint32_t)vertexInputAttributeDescriptions.size(),    // vertexAttributeDescriptionCount
        vertexInputAttributeDescriptions.data()     // pVertexAttributeDescriptions
    );


    std::vector<vk::PipelineColorBlendAttachmentState> pipelineColorBlendAttachmentStates(blendAttachmentCount);
    for(int i = 0;i < blendAttachmentCount; i++) {
        pipelineColorBlendAttachmentStates[i] = pipelineColorBlendAttachmentState;
    }
    pipelineColorBlendStateCreateInfo = vk::PipelineColorBlendStateCreateInfo(
        vk::PipelineColorBlendStateCreateFlags(),   // flags
        logicalOpEnable,                            // logicOpEnable
        logicalOp,                                  // logicOp
        blendAttachmentCount,                       // attachmentCount
        pipelineColorBlendAttachmentStates.data(),  // pAttachments
        { { (1.0f, 1.0f, 1.0f, 1.0f) } }            // blendConstants
    );

    pipelineDynamicStateCreateInfo = vk::PipelineDynamicStateCreateInfo(
        vk::PipelineDynamicStateCreateFlags(),
        (uint32_t)dynamicStates.size(),
        dynamicStates.data()
    );

    descriptorSetLayout = vd->device->createDescriptorSetLayoutUnique(
        {
            {},
            uint32_t(setLayoutBindings.size()),
            setLayoutBindings.data()
        }
    );
    if(setLayoutBindings.size() > 0) {
        descriptorPool = vd->device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo{ vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });;
        std::vector<vk::UniqueDescriptorSet> descriptorSets = vd->device->allocateDescriptorSetsUnique({ descriptorPool.get(), 1, &descriptorSetLayout.get() });
        descriptorSet = std::move(descriptorSets[0]);
    }
    // TODO can there be more than 1 descriptor set??
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
        vk::PipelineLayoutCreateFlags(),
        1, &descriptorSetLayout.get(),    // descriptorSetLayout
        (uint32_t)pushConstantRange.size(), pushConstantRange.data()      // constantRange
    );

    pipelineLayout = vd->device->createPipelineLayoutUnique(pipelineLayoutCreateInfo);

    if(renderingPipelineCreateInfo == nullptr) {
        graphicsPipelineCreateInfo = vk::GraphicsPipelineCreateInfo(
            vk::PipelineCreateFlags(),                      // flags
            int32_t(pipelineShaderStageCreateInfos.size()), // stageCount
            pipelineShaderStageCreateInfos.data(),          // pStages
            &pipelineVertexInputStateCreateInfo,            // pVertexInputState
            &pipelineInputAssemblyStateCreateInfo,          // pInputAssemblyState
            pipelineTessellationStateCreateInfo,            // pTessellationState
            &pipelineViewportStateCreateInfo,               // pViewportState
            &pipelineRasterizationStateCreateInfo,          // pRasterizationState
            &pipelineMultisampleStateCreateInfo,            // pMultisampleState
            &pipelineDepthStencilStateCreateInfo,           // pDepthStencilState
            &pipelineColorBlendStateCreateInfo,             // pColorBlendState
            &pipelineDynamicStateCreateInfo,                // pDynamicState
            pipelineLayout.get(),                           // layout
            renderPass.get()                                // renderPass
        );
    } else {
        graphicsPipelineCreateInfo = vk::GraphicsPipelineCreateInfo(
            vk::PipelineCreateFlags(),                      // flags
            int32_t(pipelineShaderStageCreateInfos.size()), // stageCount
            pipelineShaderStageCreateInfos.data(),          // pStages
            &pipelineVertexInputStateCreateInfo,            // pVertexInputState
            &pipelineInputAssemblyStateCreateInfo,          // pInputAssemblyState
            pipelineTessellationStateCreateInfo,            // pTessellationState
            &pipelineViewportStateCreateInfo,               // pViewportState
            &pipelineRasterizationStateCreateInfo,          // pRasterizationState
            &pipelineMultisampleStateCreateInfo,            // pMultisampleState
            &pipelineDepthStencilStateCreateInfo,           // pDepthStencilState
            &pipelineColorBlendStateCreateInfo,             // pColorBlendState
            &pipelineDynamicStateCreateInfo,                // pDynamicState
            pipelineLayout.get(),                           // layout
            VK_NULL_HANDLE                                  // renderPass
        );
        graphicsPipelineCreateInfo.pNext = renderingPipelineCreateInfo;
    }
    return vd->device->createGraphicsPipelineUnique(vd->pipelineCache.get(), graphicsPipelineCreateInfo).value;
}

// TODO test
void GraphicsPipelineProperties::enableDepthTest() {
    std::cerr << "------------> not sure depth test is going to work.. double check when you use this!!!" << std::endl;

    vk::StencilOpState stencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways);
    pipelineDepthStencilStateCreateInfo = vk::PipelineDepthStencilStateCreateInfo(
        vk::PipelineDepthStencilStateCreateFlags(), // flags
        true,                                       // depthTestEnable
        true,                                       // depthWriteEnable
        vk::CompareOp::eLessOrEqual,                // depthCompareOp
        false,                                      // depthBoundTestEnable
        false,                                      // stencilTestEnable
        stencilOpState,                             // front
        stencilOpState                              // back
    );
}

void GraphicsPipelineProperties::updateBlendProperties(vk::PipelineColorBlendAttachmentState blendState) {
    this->pipelineColorBlendAttachmentState = blendState;
}

void GraphicsPipelineProperties::updateBlendAttachmentCount(int attachmentCount) {
    blendAttachmentCount = attachmentCount;
}

void GraphicsPipelineProperties::setInputAssemblyFlag() {
    inputAssemblyFlag = true;
}

void GraphicsPipelineProperties::setBlendFunction(BlendFunc blendFunc) {
    switch(blendFunc) {
    case BlendFunc::BLEND_NONE:
        pipelineColorBlendAttachmentState.blendEnable = VK_FALSE;
        break;

    case BlendFunc::BLEND_DEFAULT_ALPHA:
        pipelineColorBlendAttachmentState.blendEnable = VK_TRUE;
        pipelineColorBlendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        pipelineColorBlendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrc1Alpha;
        pipelineColorBlendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        pipelineColorBlendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
        pipelineColorBlendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrc1Alpha;
        pipelineColorBlendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
        break;

    case BlendFunc::BLEND_ADD:
        pipelineColorBlendAttachmentState.blendEnable = VK_TRUE;
        pipelineColorBlendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        pipelineColorBlendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
        break;

    case BlendFunc::BLEND_MIN:
        pipelineColorBlendAttachmentState.blendEnable = VK_TRUE;
        pipelineColorBlendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.colorBlendOp = vk::BlendOp::eMin;
        pipelineColorBlendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.alphaBlendOp = vk::BlendOp::eMin;
        break;

    case BlendFunc::BLEND_MAX:
        pipelineColorBlendAttachmentState.blendEnable = VK_TRUE;
        pipelineColorBlendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.colorBlendOp = vk::BlendOp::eMax;
        pipelineColorBlendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.alphaBlendOp = vk::BlendOp::eMax;
        break;

    case BlendFunc::BLEND_OVERWRITE:
        pipelineColorBlendAttachmentState.blendEnable = VK_TRUE;
        pipelineColorBlendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eZero;
        pipelineColorBlendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        pipelineColorBlendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        pipelineColorBlendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
        pipelineColorBlendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
        break;

    default:
        std::cerr << "blend operation not supported" << std::endl;
        exit(-3);
    }

}

void GraphicsPipelineProperties::setLogicalOp(vk::LogicOp lop) {
    this->logicalOp = lop;
    this->logicalOpEnable = VK_TRUE;

    pipelineColorBlendAttachmentState.blendEnable = VK_FALSE;
}

void GraphicsPipelineProperties::setInputAttrFlag() {
    inputAttrFlag = true;
}

void GraphicsPipelineProperties::setInputBindingFlag() {
    inputBindingFlag = true;
}

void GraphicsPipelineProperties::setShaderStageFlag() {
    shaderStageFlag = true;
}

} // namespace
