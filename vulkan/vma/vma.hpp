// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#if !defined(VK_USE_PLATFORM_WIN32_KHR)
    #define VK_USE_PLATFORM_WIN32_KHR
#endif // #if !defined(VK_USE_PLATFORM_WIN32_KHR)

// Handle for external memory
typedef HANDLE Handle;
#define ExternalMemoryHandleBit vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32

#else  // #ifdef _WIN32

#include <vulkan/vulkan.h>

#define ExternalMemoryHandleBit vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd
typedef int Handle;

#endif  // #ifdef _WIN32

#ifdef _MSVC_LANG

// Uncomment to test including `vulkan.h` on your own before including VMA.
//#include <vulkan/vulkan.h>

/*
In every place where you want to use Vulkan Memory Allocator, define appropriate
macros if you want to configure the library and then include its header to
include all public interface declarations. Example:
*/

//#define VMA_HEAVY_ASSERT(expr) assert(expr)
//#define VMA_DEDICATED_ALLOCATION 0
//#define VMA_DEBUG_MARGIN 16
//#define VMA_DEBUG_DETECT_CORRUPTION 1
//#define VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY 256
//#define VMA_USE_STL_SHARED_MUTEX 0
//#define VMA_MEMORY_BUDGET 0
//#define VMA_STATS_STRING_ENABLED 0
//#define VMA_MAPPING_HYSTERESIS_ENABLED 0

#define VMA_VULKAN_VERSION 1003000 // Vulkan 1.3
//#define VMA_VULKAN_VERSION 1002000 // Vulkan 1.2
//#define VMA_VULKAN_VERSION 1001000 // Vulkan 1.1
//#define VMA_VULKAN_VERSION 1000000 // Vulkan 1.0

/*
#define VMA_DEBUG_LOG(format, ...) do { \
        printf(format, __VA_ARGS__); \
        printf("\n"); \
    } while(false)
*/

#pragma warning(push, 4)
#pragma warning(disable: 4127) // conditional expression is constant
#pragma warning(disable: 4100) // unreferenced formal parameter
#pragma warning(disable: 4189) // local variable is initialized but not referenced
#pragma warning(disable: 4324) // structure was padded due to alignment specifier

#endif  // #ifdef _MSVC_LANG


#include "vma/vk_mem_alloc.h"


#ifdef _MSVC_LANG
    #pragma warning(pop)
#endif


