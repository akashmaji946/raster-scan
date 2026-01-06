# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set(scan_headers
    RasterScan2D.hpp
    RasterScanIndexUpdate.hpp
    BufferPool.hpp
    CPUTimer.hpp
    modes/ModeUtils.hpp
    modes/RunModes.hpp
    CompactScanIndex.hpp
)

set(scan_src
    scan.cpp
    RasterScan2D.cpp
    RasterScanIndexUpdate.cpp
    BufferPool.cpp
    CPUTimer.cpp
    modes/ModeUtils.cpp
    modes/Mode0.cpp
    modes/Mode1.cpp
    modes/Mode2.cpp
    modes/Mode3.cpp
    modes/Mode4.cpp
    modes/Mode5.cpp
    modes/Mode6.cpp
    modes/Mode7.cpp
    modes/Mode8.cpp
    modes/Mode9.cpp
    modes/Mode10.cpp
    modes/Mode11.cpp
    modes/Mode12.cpp
    modes/Mode13.cpp
    modes/Mode21.cpp
    modes/Mode22.cpp
    modes/Mode23.cpp
    CompactScanIndex.cpp
)

set(encode_src
    encode.cpp
)

set(cpu_scan_src
    cpu_scan.cpp
)

set(compare_src
    compare_cpu_gpu.cpp
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
    shaders/insert-page-bitmap.vert
    shaders/insert-page-bitmap-v2.vert
    shaders/delete-by-data.vert
    shaders/mark-free-bitmap.vert
    shaders/compact-pages.vert
    shaders/verify-index.comp
    shaders/compact_insert.comp
    shaders/compact_query.comp
    shaders/compact_delete.comp
    shaders/compact_count.comp
    shaders/compact_stats.comp
    shaders/compact_count_gfx.vert
    shaders/compact_build_gfx.vert
    shaders/compact_scale_counts.comp
    shaders/compact_query_gfx.vert
    shaders/compact_query_gfx.geom
    shaders/compact_query_gfx.frag
    shaders/compact_range.frag
    shaders/compact_edge.vert
    shaders/compact_edge.geom
    shaders/compact_edge.frag
)

add_executable(EncodeData ${encode_src})
add_executable(RasterScan ${scan_src} ${scan_shader_src} ${scan_headers})
add_executable(CPUScan ${cpu_scan_src})
add_executable(Compare ${compare_src})

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
