cmake_minimum_required(VERSION 3.12)
project(radix_sort)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# Shader Compilation
set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders/spv")
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(SHADERS
    ${SHADER_SOURCE_DIR}/histogram.comp
    ${SHADER_SOURCE_DIR}/scan_add.comp
    ${SHADER_SOURCE_DIR}/scatter.comp
    ${SHADER_SOURCE_DIR}/shuffle.comp
    ${SHADER_SOURCE_DIR}/scan_global.comp
)

set(SPV_SHADERS
    ${SHADER_BINARY_DIR}/histogram.spv
    ${SHADER_BINARY_DIR}/scan.spv
    ${SHADER_BINARY_DIR}/scatter.spv
    ${SHADER_BINARY_DIR}/scan_reduce.spv
    ${SHADER_BINARY_DIR}/scan_global.spv
    ${SHADER_BINARY_DIR}/scan_add.spv
    ${SHADER_BINARY_DIR}/shuffle.spv
)

find_program(GLSLC_EXECUTABLE NAMES glslc HINTS ${Vulkan_GLSLC_EXECUTABLE})
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found! Please install Vulkan SDK or make sure it's in PATH.")
endif()

# Custom command to compile shaders
add_custom_command(
    OUTPUT ${SPV_SHADERS}
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE_DIR}/histogram.comp -o ${SHADER_BINARY_DIR}/histogram.spv
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE_DIR}/scan.comp -o ${SHADER_BINARY_DIR}/scan.spv
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE_DIR}/scatter.comp -o ${SHADER_BINARY_DIR}/scatter.spv
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE_DIR}/scan_reduce.comp -o ${SHADER_BINARY_DIR}/scan_reduce.spv
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE_DIR}/scan_global.comp -o ${SHADER_BINARY_DIR}/scan_global.spv
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE_DIR}/scan_add.comp -o ${SHADER_BINARY_DIR}/scan_add.spv
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE_DIR}/shuffle.comp -o ${SHADER_BINARY_DIR}/shuffle.spv
    DEPENDS ${SHADERS}
    COMMENT "Compiling Shaders..."
)

add_custom_target(compile_shaders DEPENDS ${SPV_SHADERS})

# Main Executable
add_executable(main
    main.cpp
    radix.cpp
    GPURadixSort.cpp
    vulkan_common.cpp
    vulkan_common.hpp
)

add_dependencies(main compile_shaders)

target_link_libraries(main 
    PRIVATE 
    Vulkan::Vulkan
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Optional X11 libs if needed for Vulkan surface on Linux (though we are compute only)
# target_link_libraries(main PRIVATE X11 Xxf86vm Xrandr Xi) 
