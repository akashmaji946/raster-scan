# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set(src
    common/utils.cpp
    core/ComputePipelineProperties.cpp
    core/GraphicsPipelineProperties.cpp
    core/GPUTimer.cpp
    core/VkData.cpp
    core/VkEngine.cpp
    core/VulkanDevice.cpp
    core/vkutils.cpp
    operators/OperatorCache.cpp
    operators/ReduceMax.cpp
    operators/SinglePassScan.cpp
    vma/vma.cpp
)

set(headers
    common/constants.h
    common/ShaderDefines.h
    common/utils.h
    core/ComputePipelineProperties.hpp
    core/GraphicsPipelineProperties.hpp
    core/GPUTimer.hpp
    core/VkData.hpp
    core/VkEngine.hpp
    core/VkInclude.hpp
    core/VulkanDevice.hpp
    core/vkutils.h
    operators/OperatorCache.hpp
    operators/ReduceMax.hpp
    operators/SinglePassScan.hpp
    vma/vma.hpp
)

set(shader_src
    shaders/prefixsum.comp
    shaders/reduce-max.comp
)

add_library(VkCore STATIC ${src} ${headers} ${shader_src})

target_include_directories(VkCore PUBLIC ./)
target_include_directories(VkCore PUBLIC ${Vulkan_INCLUDE_DIRS})

target_link_libraries(VkCore ${Vulkan_LIBRARIES})
target_link_libraries(VkCore ${CMAKE_DL_LIBS})

# compile shaders
foreach(shader ${shader_src})
    get_filename_component(p ${shader} NAME)
    add_shader(VkCore ${p})
    source_group("Shaders" FILES ${shader})
endforeach(shader)

if(MSVC)
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
    add_definitions(-DNOMINMAX)
endif()

if (UNIX)
    target_link_libraries(VkCore -pthread)
endif()
