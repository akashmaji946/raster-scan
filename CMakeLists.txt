# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.10)

project(RasterScan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_DEBUG_POSTFIX d)

# Setup vcpkg
include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

## Dependencies (supports vcpkg manifest mode)
find_package(Vulkan REQUIRED COMPONENTS glslc)
find_program(GLSLC NAMES glslc HINTS Vulkan::glslc)

find_package(VulkanMemoryAllocator CONFIG QUIET)
if(VulkanMemoryAllocator_FOUND)
    message(STATUS "Using VulkanMemoryAllocator from vcpkg")
    set(VMA_SYSTEM_AVAILABLE ON)
    add_definitions(-DUSE_SYSTEM_VMA)
else()
    message(STATUS "VulkanMemoryAllocator not found; using bundled implementation")
endif()

# script to compile vulkan shaders
include(shaders.cmake)

# use vulkan validation
option(VULKAN_DEV "enable vulkan debugging" ON)

# enable vulkan validation.
if(VULKAN_DEV)
    add_definitions(-DDEV_BUILD)
endif()

add_subdirectory(vulkan)
add_subdirectory(scan)


