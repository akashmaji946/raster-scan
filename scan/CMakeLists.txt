# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set(scan_headers
    RasterScan2D.hpp
    RasterScanIndexUpdate.hpp
    BufferPool.hpp
    CPUTimer.hpp
)

set(scan_src
    scan.cpp
    RasterScan2D.cpp
    RasterScanIndexUpdate.cpp
    BufferPool.cpp
    CPUTimer.cpp
)

set(encode_src
    encode.cpp
)

set(scan_shader_src
    shaders/dummy.frag
    shaders/build-count-2D.vert
    shaders/build2D.vert
    shaders/range2D.vert
    shaders/range2D.frag
    shaders/range2D.geom
    shaders/edge2D.vert
    shaders/edge2D.frag
    shaders/edge2D.geom
    shaders/insert-page.vert
    shaders/delete-page.vert
    shaders/delete-range.vert
    shaders/query-tex-ll.vert
    shaders/query-tex-ll.geom
    shaders/query-tex-ll.frag
    shaders/query-page-ll.vert
    shaders/query-page-ll.geom
    shaders/query-page-ll.frag
)

add_executable(EncodeData ${encode_src})
add_executable(RasterScan ${scan_src} ${scan_shader_src} ${scan_headers})

target_include_directories(RasterScan PUBLIC ./)
target_include_directories(RasterScan PUBLIC ${Vulkan_INCLUDE_DIRS})

target_link_libraries(RasterScan ${Vulkan_LIBRARIES})
target_link_libraries(RasterScan ${CMAKE_DL_LIBS})
target_link_libraries(RasterScan VkCore)


# compile shaders
foreach(shader ${scan_shader_src})
    get_filename_component(p ${shader} NAME)
    add_shader(RasterScan ${p})
    source_group("Shaders" FILES ${shader})
endforeach(shader)

if(MSVC)
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
    add_definitions(-DNOMINMAX)
endif()
