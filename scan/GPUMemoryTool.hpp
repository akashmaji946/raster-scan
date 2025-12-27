// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef GPU_MEMORY_TOOL_HPP
#define GPU_MEMORY_TOOL_HPP

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>

// Forward declarations for Vulkan/VMA types
#include <core/VulkanDevice.hpp>

/**
 * GPUMemoryTool - Utility class for tracking and reporting GPU memory usage
 * 
 * Provides functions to:
 * - Calculate memory requirements for buffers/textures
 * - Track allocated memory
 * - Print memory usage reports
 * - Convert between bytes, MB, and GB
 */
class GPUMemoryTool {
public:
    // Singleton access
    static GPUMemoryTool& getInstance() {
        static GPUMemoryTool instance;
        return instance;
    }

    // Memory conversion utilities
    static double bytesToMB(uint64_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }

    static double bytesToGB(uint64_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    }

    static uint64_t MBToBytes(double mb) {
        return static_cast<uint64_t>(mb * 1024.0 * 1024.0);
    }

    static uint64_t GBToBytes(double gb) {
        return static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
    }

    // Format bytes as human-readable string
    static std::string formatBytes(uint64_t bytes) {
        if (bytes >= GBToBytes(1.0)) {
            return std::to_string(bytesToGB(bytes)).substr(0, 5) + " GB";
        } else if (bytes >= MBToBytes(1.0)) {
            return std::to_string(bytesToMB(bytes)).substr(0, 6) + " MB";
        } else if (bytes >= 1024) {
            return std::to_string(bytes / 1024) + " KB";
        }
        return std::to_string(bytes) + " B";
    }

    // Calculate buffer memory requirements
    static uint64_t calcBufferSize(uint32_t elementCount, uint32_t elementSize) {
        return static_cast<uint64_t>(elementCount) * elementSize;
    }

    // Calculate page buffer size for linked list index
    static uint64_t calcPageBufferSize(uint32_t maxPages, uint32_t pageSizeUints) {
        return static_cast<uint64_t>(maxPages) * pageSizeUints * sizeof(uint32_t);
    }

    // Calculate head pointer buffer size
    static uint64_t calcHeadBufferSize(uint32_t resolution) {
        return static_cast<uint64_t>(resolution) * resolution * sizeof(uint32_t);
    }

    // Calculate points buffer size
    static uint64_t calcPointsBufferSize(uint32_t npoints, uint32_t ncols) {
        return static_cast<uint64_t>(npoints) * ncols * sizeof(uint32_t);
    }

    // Calculate result bitmap buffer size
    static uint64_t calcResultBufferSize(uint32_t npoints) {
        uint32_t arrsize = (npoints + 31) / 32;  // ceil(npoints / 32)
        return static_cast<uint64_t>(arrsize) * sizeof(uint32_t);
    }

    // Track memory allocation
    void trackAllocation(const std::string& name, uint64_t bytes) {
        allocations.push_back({name, bytes});
        totalAllocated += bytes;
    }

    // Clear tracked allocations
    void clearAllocations() {
        allocations.clear();
        totalAllocated = 0;
    }

    // Get total allocated memory
    uint64_t getTotalAllocated() const {
        return totalAllocated;
    }

    // Print single allocation info
    static void printAllocation(const std::string& name, uint64_t bytes) {
        std::cerr << "[GPU Memory] " << std::setw(30) << std::left << name 
                  << ": " << std::setw(12) << std::right << formatBytes(bytes) 
                  << " (" << bytes << " bytes)" << std::endl;
    }

    // Print allocation with calculation details
    static void printAllocationDetails(const std::string& name, uint64_t count, 
                                        uint32_t elementSize, const std::string& formula = "") {
        uint64_t bytes = count * elementSize;
        std::cerr << "[GPU Memory] " << std::setw(30) << std::left << name 
                  << ": " << std::setw(12) << std::right << formatBytes(bytes);
        if (!formula.empty()) {
            std::cerr << " (" << formula << ")";
        }
        std::cerr << std::endl;
    }

    // Print memory summary
    void printSummary() const {
        std::cerr << "\n========== GPU Memory Summary ==========\n";
        for (const auto& alloc : allocations) {
            std::cerr << "  " << std::setw(30) << std::left << alloc.name 
                      << ": " << std::setw(12) << std::right << formatBytes(alloc.bytes) 
                      << std::endl;
        }
        std::cerr << "----------------------------------------\n";
        std::cerr << "  " << std::setw(30) << std::left << "TOTAL" 
                  << ": " << std::setw(12) << std::right << formatBytes(totalAllocated) 
                  << std::endl;
        std::cerr << "=========================================\n\n";
    }

    // Print memory budget check
    static void printBudgetCheck(uint64_t required, uint64_t available, const std::string& gpuName = "") {
        std::cerr << "\n[GPU Memory Budget Check]";
        if (!gpuName.empty()) {
            std::cerr << " - " << gpuName;
        }
        std::cerr << std::endl;
        std::cerr << "  Required:  " << formatBytes(required) << std::endl;
        std::cerr << "  Available: " << formatBytes(available) << std::endl;
        
        if (required <= available) {
            double usage = (static_cast<double>(required) / available) * 100.0;
            std::cerr << "  Status:    OK (" << std::fixed << std::setprecision(1) 
                      << usage << "% of available)" << std::endl;
        } else {
            std::cerr << "  Status:    EXCEEDS BUDGET by " 
                      << formatBytes(required - available) << std::endl;
        }
        std::cerr << std::endl;
    }

    // Print linked list index memory requirements
    static void printLinkedListIndexMemory(uint32_t maxPages, uint32_t pageSizeUints, 
                                            uint32_t resolution, uint32_t npoints, uint32_t ncols) {
        std::cerr << "\n[GPU Memory] Linked List Index Requirements:\n";
        
        uint64_t pageBuffer = calcPageBufferSize(maxPages, pageSizeUints);
        uint64_t headBuffer = calcHeadBufferSize(resolution);
        uint64_t pointsBuffer = calcPointsBufferSize(npoints, ncols);
        uint64_t resultBuffer = calcResultBufferSize(npoints);
        uint64_t allocCounter = sizeof(uint32_t);
        
        std::cerr << "  Page Buffer:      " << std::setw(10) << formatBytes(pageBuffer) 
                  << " (" << maxPages << " pages * " << pageSizeUints << " uints * 4 bytes)\n";
        std::cerr << "  Head Ptr Buffer:  " << std::setw(10) << formatBytes(headBuffer) 
                  << " (" << resolution << " * " << resolution << " cells * 4 bytes)\n";
        std::cerr << "  Points Buffer:    " << std::setw(10) << formatBytes(pointsBuffer) 
                  << " (" << npoints << " points * " << ncols << " cols * 4 bytes)\n";
        std::cerr << "  Result Buffer:    " << std::setw(10) << formatBytes(resultBuffer) 
                  << " (bitmap for " << npoints << " points)\n";
        std::cerr << "  Alloc Counter:    " << std::setw(10) << formatBytes(allocCounter) << "\n";
        
        uint64_t total = pageBuffer + headBuffer + pointsBuffer + resultBuffer + allocCounter;
        std::cerr << "  ----------------------------------------\n";
        std::cerr << "  TOTAL:            " << std::setw(10) << formatBytes(total) << "\n\n";
    }

    // Print RasterScan2D index memory requirements
    static void printRasterScan2DMemory(uint32_t npoints, uint32_t ncols, uint32_t resolution) {
        std::cerr << "\n[GPU Memory] RasterScan2D Index Requirements:\n";
        
        uint64_t pointsBuffer = calcPointsBufferSize(npoints, ncols);
        uint64_t countBuffer = static_cast<uint64_t>(resolution) * resolution * sizeof(uint32_t);
        uint64_t offsetBuffer = countBuffer;
        uint64_t resultBuffer = calcResultBufferSize(npoints);
        
        std::cerr << "  Points Buffer:    " << std::setw(10) << formatBytes(pointsBuffer) 
                  << " (" << npoints << " points * " << ncols << " cols * 4 bytes)\n";
        std::cerr << "  Count Buffer:     " << std::setw(10) << formatBytes(countBuffer) 
                  << " (" << resolution << " * " << resolution << " cells)\n";
        std::cerr << "  Offset Buffer:    " << std::setw(10) << formatBytes(offsetBuffer) 
                  << " (" << resolution << " * " << resolution << " cells)\n";
        std::cerr << "  Result Buffer:    " << std::setw(10) << formatBytes(resultBuffer) 
                  << " (bitmap for " << npoints << " points)\n";
        
        uint64_t total = pointsBuffer + countBuffer + offsetBuffer + resultBuffer;
        std::cerr << "  ----------------------------------------\n";
        std::cerr << "  TOTAL:            " << std::setw(10) << formatBytes(total) << "\n\n";
    }

    // Estimate total memory for comparison mode (both indexes)
    static uint64_t estimateComparisonModeMemory(uint32_t maxPages, uint32_t pageSizeUints,
                                                  uint32_t resolution, uint32_t npoints, uint32_t ncols) {
        // Linked list index
        uint64_t llPageBuffer = calcPageBufferSize(maxPages, pageSizeUints);
        uint64_t llHeadBuffer = calcHeadBufferSize(resolution);
        
        // RasterScan2D index
        uint64_t rs2dCountBuffer = static_cast<uint64_t>(resolution) * resolution * sizeof(uint32_t);
        uint64_t rs2dOffsetBuffer = rs2dCountBuffer;
        
        // Shared
        uint64_t pointsBuffer = calcPointsBufferSize(npoints, ncols);
        uint64_t resultBuffer = calcResultBufferSize(npoints);
        uint64_t stagingBuffer = 64 * 1024 * 1024;  // 64MB staging
        
        return llPageBuffer + llHeadBuffer + rs2dCountBuffer + rs2dOffsetBuffer + 
               pointsBuffer + resultBuffer + stagingBuffer;
    }

    // =========================================================================
    // VMA-based GPU Memory Query Functions (query actual GPU usage)
    // =========================================================================

    /**
     * Query actual GPU memory usage from VMA allocator
     * Returns a struct with usage and budget info for each heap
     */
    struct HeapBudgetInfo {
        uint64_t usage;           // Current usage in bytes
        uint64_t budget;          // Available budget in bytes
        uint64_t allocationCount; // Number of allocations
        uint64_t blockCount;      // Number of memory blocks
    };

    /**
     * Get heap budget info from VMA allocator
     * @param vd Vulkan device with VMA allocator
     * @param heapIndex Memory heap index (usually 0 for device local)
     * @return HeapBudgetInfo with usage statistics
     */
    static HeapBudgetInfo getHeapBudget(vkcore::PVkDevice vd, uint32_t heapIndex = 0) {
        HeapBudgetInfo info = {0, 0, 0, 0};
        
        if (!vd || !vd->allocator) {
            return info;
        }

        // Get number of memory heaps
        uint32_t heapCount = vd->memProps.memoryHeapCount;
        if (heapIndex >= heapCount) {
            return info;
        }

        // Query VMA for heap budgets
        std::vector<VmaBudget> budgets(heapCount);
        vmaGetHeapBudgets(vd->allocator, budgets.data());

        info.usage = budgets[heapIndex].usage;
        info.budget = budgets[heapIndex].budget;
        info.allocationCount = budgets[heapIndex].statistics.allocationCount;
        info.blockCount = budgets[heapIndex].statistics.blockCount;

        return info;
    }

    /**
     * Print current GPU memory usage queried from VMA
     * @param vd Vulkan device with VMA allocator
     */
    static void printGPUMemoryUsage(vkcore::PVkDevice vd) {
        if (!vd || !vd->allocator) {
            std::cerr << "[GPU Memory] Error: Invalid device or allocator\n";
            return;
        }

        uint32_t heapCount = vd->memProps.memoryHeapCount;
        std::vector<VmaBudget> budgets(heapCount);
        vmaGetHeapBudgets(vd->allocator, budgets.data());

        std::cerr << "\n========== GPU Memory Usage (VMA Query) ==========\n";
        std::cerr << "Device: " << vd->props.deviceName << "\n";
        std::cerr << "--------------------------------------------------\n";

        uint64_t totalUsage = 0;
        uint64_t totalBudget = 0;

        for (uint32_t i = 0; i < heapCount; i++) {
            vk::MemoryHeapFlags flags = vd->memProps.memoryHeaps[i].flags;
            bool isDeviceLocal = (flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{};
            
            std::string heapType = isDeviceLocal ? "Device Local" : "Host Visible";
            
            std::cerr << "Heap " << i << " (" << heapType << "):\n";
            std::cerr << "  Usage:       " << std::setw(12) << formatBytes(budgets[i].usage) 
                      << " / " << formatBytes(budgets[i].budget) << "\n";
            std::cerr << "  Allocations: " << budgets[i].statistics.allocationCount << "\n";
            std::cerr << "  Blocks:      " << budgets[i].statistics.blockCount << "\n";
            
            if (budgets[i].budget > 0) {
                double usagePercent = (static_cast<double>(budgets[i].usage) / budgets[i].budget) * 100.0;
                std::cerr << "  Utilization: " << std::fixed << std::setprecision(1) 
                          << usagePercent << "%\n";
            }
            std::cerr << "\n";

            if (isDeviceLocal) {
                totalUsage += budgets[i].usage;
                totalBudget += budgets[i].budget;
            }
        }

        std::cerr << "--------------------------------------------------\n";
        std::cerr << "Total Device Local Memory:\n";
        std::cerr << "  Used:      " << std::setw(12) << formatBytes(totalUsage) << "\n";
        std::cerr << "  Available: " << std::setw(12) << formatBytes(totalBudget - totalUsage) << "\n";
        std::cerr << "  Budget:    " << std::setw(12) << formatBytes(totalBudget) << "\n";
        if (totalBudget > 0) {
            double usagePercent = (static_cast<double>(totalUsage) / totalBudget) * 100.0;
            std::cerr << "  Usage:     " << std::fixed << std::setprecision(1) 
                      << usagePercent << "%\n";
        }
        std::cerr << "==================================================\n\n";
    }

    /**
     * Print a brief one-line GPU memory status with optional message
     * @param vd Vulkan device with VMA allocator
     * @param message Optional message to print before the status
     */
    static void printGPUMemoryStatus(vkcore::PVkDevice vd, const std::string& message = "") {
        if (!vd || !vd->allocator) {
            return;
        }

        uint32_t heapCount = vd->memProps.memoryHeapCount;
        std::vector<VmaBudget> budgets(heapCount);
        vmaGetHeapBudgets(vd->allocator, budgets.data());

        // Find device local heap
        uint64_t usage = 0, budget = 0;
        for (uint32_t i = 0; i < heapCount; i++) {
            vk::MemoryHeapFlags flags = vd->memProps.memoryHeaps[i].flags;
            if ((flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{}) {
                usage += budgets[i].usage;
                budget += budgets[i].budget;
            }
        }

        double usagePercent = budget > 0 ? (static_cast<double>(usage) / budget) * 100.0 : 0.0;
        
        if (!message.empty()) {
            std::cout << "[GPU Memory] " << message << ": ";
        } else {
            std::cout << "[GPU Memory] ";
        }
        std::cout << formatBytes(usage) << " / " << formatBytes(budget) 
                  << " (" << std::fixed << std::setprecision(1) << usagePercent << "% used)" << std::endl;
    }

    /**
     * Get device local memory usage in bytes
     * @param vd Vulkan device with VMA allocator
     * @return Current device local memory usage in bytes
     */
    static uint64_t getDeviceLocalUsage(vkcore::PVkDevice vd) {
        if (!vd || !vd->allocator) {
            return 0;
        }

        uint32_t heapCount = vd->memProps.memoryHeapCount;
        std::vector<VmaBudget> budgets(heapCount);
        vmaGetHeapBudgets(vd->allocator, budgets.data());

        uint64_t usage = 0;
        for (uint32_t i = 0; i < heapCount; i++) {
            vk::MemoryHeapFlags flags = vd->memProps.memoryHeaps[i].flags;
            if ((flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{}) {
                usage += budgets[i].usage;
            }
        }
        return usage;
    }

    /**
     * Get device local memory budget (available) in bytes
     * @param vd Vulkan device with VMA allocator
     * @return Device local memory budget in bytes
     */
    static uint64_t getDeviceLocalBudget(vkcore::PVkDevice vd) {
        if (!vd || !vd->allocator) {
            return 0;
        }

        uint32_t heapCount = vd->memProps.memoryHeapCount;
        std::vector<VmaBudget> budgets(heapCount);
        vmaGetHeapBudgets(vd->allocator, budgets.data());

        uint64_t budget = 0;
        for (uint32_t i = 0; i < heapCount; i++) {
            vk::MemoryHeapFlags flags = vd->memProps.memoryHeaps[i].flags;
            if ((flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{}) {
                budget += budgets[i].budget;
            }
        }
        return budget;
    }

    /**
     * Check if there's enough GPU memory available
     * @param vd Vulkan device
     * @param requiredBytes Required memory in bytes
     * @return true if enough memory is available
     */
    static bool hasEnoughMemory(vkcore::PVkDevice vd, uint64_t requiredBytes) {
        uint64_t usage = getDeviceLocalUsage(vd);
        uint64_t budget = getDeviceLocalBudget(vd);
        return (usage + requiredBytes) <= budget;
    }

private:
    GPUMemoryTool() : totalAllocated(0) {}
    GPUMemoryTool(const GPUMemoryTool&) = delete;
    GPUMemoryTool& operator=(const GPUMemoryTool&) = delete;

    struct Allocation {
        std::string name;
        uint64_t bytes;
    };

    std::vector<Allocation> allocations;
    uint64_t totalAllocated;
};

// Convenience macros for memory tracking
#define GPU_MEM_PRINT(name, bytes) GPUMemoryTool::printAllocation(name, bytes)
#define GPU_MEM_TRACK(name, bytes) GPUMemoryTool::getInstance().trackAllocation(name, bytes)
#define GPU_MEM_SUMMARY() GPUMemoryTool::getInstance().printSummary()
#define GPU_MEM_CLEAR() GPUMemoryTool::getInstance().clearAllocations()

#endif // GPU_MEMORY_TOOL_HPP
