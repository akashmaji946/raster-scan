# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Function to compile shaders
function(add_shader TARGET SHADER)
    # All shaders are found here.
    set(current-shader-path ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER})
    set(current-output-path ${CMAKE_BINARY_DIR}/shaders/${SHADER}.spv)

    # Add a custom command to compile GLSL to SPIR-V.
    get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
    file(MAKE_DIRECTORY ${current-output-dir})
    add_custom_command(
        OUTPUT ${current-output-path}
        COMMAND ${GLSLC} -o ${current-output-path} ${current-shader-path} --target-env=vulkan1.1
        DEPENDS ${current-shader-path}
        IMPLICIT_DEPENDS CXX ${current-shader-path}
        VERBATIM)

    # Make sure our native build depends on this output.
    set_source_files_properties(${current-output-path} PROPERTIES GENERATED TRUE)
    target_sources(${TARGET} PRIVATE ${current-output-path})
endfunction(add_shader)

