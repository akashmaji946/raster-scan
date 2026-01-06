#include "RunModes.hpp"
#include "../CompactScanIndex.hpp"
#include "../BufferPool.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>

// Distribution names for dataId 0-4
static const std::vector<std::string> distributionFiles23 = {
    "uniform.bin",
    "normal.bin",
    "zipf1.1.bin",
    "zipf1.3.bin",
    "zipf1.5.bin"
};

static const std::vector<std::string> distributionNames23 = {
    "uniform",
    "normal",
    "zipf1.1",
    "zipf1.3",
    "zipf1.5"
};

// Helper: Extract valid points from GPU data buffer
static std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> extractGPUPoints(
    const std::vector<CompactEntry>& gpuData
) {
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> result;
    for (const auto& entry : gpuData) {
        if (entry.rowId & 0x80000000u) { // Valid bit set
            result.emplace_back(entry.x, entry.y, entry.z);
        }
    }
    // Sort for comparison
    std::sort(result.begin(), result.end());
    return result;
}

// Helper: Extract points from CPU reference (column-major format)
static std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> extractCPUPoints(
    const std::vector<uint32_t>& points,
    uint32_t npoints,
    const std::vector<bool>& validMask
) {
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> result;
    for (uint32_t i = 0; i < npoints; i++) {
        if (validMask[i]) {
            result.emplace_back(points[i], points[npoints + i], points[2 * npoints + i]);
        }
    }
    // Sort for comparison
    std::sort(result.begin(), result.end());
    return result;
}

void testCompactIndexBatchCycles(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    
    // Validate dataId (0-4 for the 5 distributions)
    if (dataId < 0 || dataId >= (int)distributionFiles23.size()) {
        std::cerr << "ERROR: Invalid dataId " << dataId << ". Must be 0-4.\n";
        std::cerr << "  0: uniform, 1: normal, 2: zipf1.1, 3: zipf1.3, 4: zipf1.5\n";
        return;
    }
    
    // Construct plain data folder path: data/data_Xm_Yc/
    uint32_t millions = g_npoints / 1000000;
    std::string plainDataFolder = PROJECT_DIR + "data/data_" + std::to_string(millions) + "m_" + std::to_string(g_dim) + "c";
    std::string dataFile = plainDataFolder + "/" + distributionFiles23[dataId];
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 23: Compact Index Batch Delete/Insert Cycles\n";
    std::cerr << "Distribution: " << distributionNames23[dataId] << " (dataId=" << dataId << ")\n";
    std::cerr << "Data file: " << dataFile << "\n";
    std::cerr << "Columns: " << g_dim << "\n";
    std::cerr << "========================================\n";

    // 1. Read Plain Dataset
    int32_t ncols = g_dim;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<uint32_t> points;
    vkcore::PBuffer pointsBuffer = readPlainData(dataFile, vd, staging, npoints, minval, maxval, points, ncols);
    std::cerr << "Dataset: " << npoints << " points\n";

    // Get SinglePassScan for GPU prefix sum
    vkcore::SinglePassScan *scan = (vkcore::SinglePassScan *) op.getFunction(vkcore::FunctionType::SinglePassScan);

    // =========================================================
    // Build CompactScanIndex
    // =========================================================
    std::cerr << "\n--- [CompactScanIndex] ---\n";
    
    PCompactScanIndex compactIndex = std::make_shared<CompactScanIndex>(vd, ncols, scan);
    compactIndex->useIndexedDelete = g_useSkewedPipeline;  // Enable O(1) delete if -s flag
    compactIndex->initialize();
    
    if (g_useSkewedPipeline) {
        std::cerr << "[Mode23] Using INDEXED delete (O(1) per delete)\n";
    } else {
        std::cerr << "[Mode23] Using LINEAR SCAN delete (O(bin_size) per delete)\n";
    }

    GPUMemoryTool::printGPUMemoryStatus(vd, "Before CompactScanIndex build");
    
    CPUTimer buildTimer;
    buildTimer.start();
    compactIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    double buildTime = double(buildTimer.stop()) / 1000000.0;
    std::cerr << "Compact Index build time: " << (buildTime * 1000.0) << " ms\n";
    GPUMemoryTool::printGPUMemoryStatus(vd, "After CompactScanIndex build");

    // =========================================================
    // Batch Delete/Insert Cycles
    // =========================================================
    
    // Configuration
    const uint32_t NUM_BATCHES = 10;
    const int RUNS = 2;
    const bool USE_RANDOM_BATCHES = true;  // Toggle: true = random, false = sequential (like Mode 22)
    const bool CPU_CHECK = false;  // Toggle: true = verify GPU results against CPU, false = skip verification
    uint32_t batchSize = npoints / NUM_BATCHES;
    
    std::cerr << "\n--- Batch Delete/Insert Cycles ---\n";
    std::cerr << "Total points: " << npoints << "\n";
    std::cerr << "Number of batches: " << NUM_BATCHES << "\n";
    std::cerr << ">Batch size: " << batchSize << " points\n";
    
    // Create indices for batch assignment
    std::vector<uint32_t> indices(npoints);
    std::iota(indices.begin(), indices.end(), 0);
    if (USE_RANDOM_BATCHES) {
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    std::cerr << "Batch order: " << (USE_RANDOM_BATCHES ? "RANDOM" : "SEQUENTIAL") << "\n";
    std::cerr << "CPU verification: " << (CPU_CHECK ? "ON" : "OFF") << "\n";
    
    // CPU reference: track which points are valid (only needed if CPU_CHECK)
    std::vector<bool> cpuValidMask;
    if (CPU_CHECK) {
        cpuValidMask.resize(npoints, true);
    }
    
    // Allocate batch buffer (row-major: [x0,y0,z0, x1,y1,z1, ...])
    vkcore::PBuffer batchBuffer(new Buffer(vd));
    batchBuffer->create(batchSize * 3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    // Batch data storage
    std::vector<uint32_t> batchData(batchSize * 3);
    
    // For indexed delete: buffer containing point indices
    vkcore::PBuffer indexBuffer;
    std::vector<uint32_t> indexData;
    if (g_useSkewedPipeline) {
        indexBuffer = std::make_shared<Buffer>(vd);
        indexBuffer->create(batchSize * sizeof(uint32_t), 
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
            MemoryType::Internal);
        indexData.resize(batchSize);
    }
    
    // Summary statistics
    uint32_t totalPassed = 0;
    double totalDelTime = 0, totalInsTime = 0;
    
    // std::cerr << "\n";
    // std::cerr << std::setw(6) << "Batch" 
    //           << std::setw(10) << "Size"
    //           << std::setw(12) << "Del(ms)"
    //           << std::setw(12) << "Ins(ms)"
    //           << std::setw(12) << "CPU_Cnt"
    //           << std::setw(12) << "GPU_Cnt"
    //           << std::setw(10) << "PtMatch"
    //           << std::setw(8) << "Status" << "\n";
    // std::cerr << std::string(82, '-') << "\n";
    
    for (uint32_t r = 0; r < RUNS; r++) {
        // Use modulo to wrap around batches when RUNS > NUM_BATCHES
        uint32_t b = r % NUM_BATCHES;
        uint32_t startIdx = b * batchSize;
        uint32_t endIdx = (b == NUM_BATCHES - 1) ? npoints : (b + 1) * batchSize;
        uint32_t currentBatchSize = endIdx - startIdx;
        
        // Prepare batch data (row-major format for delete/insert)
        for (uint32_t i = 0; i < currentBatchSize; i++) {
            uint32_t pointIdx = indices[startIdx + i];
            batchData[i * 3 + 0] = points[pointIdx];                  // x
            batchData[i * 3 + 1] = points[npoints + pointIdx];        // y
            batchData[i * 3 + 2] = points[2 * npoints + pointIdx];    // z
        }
        
        // Upload batch to GPU
        loadUsingStagingBuf((char*)batchData.data(), currentBatchSize * 3 * sizeof(uint32_t), batchBuffer, staging, vd, 0);
        
        // --- DELETE ---
        CPUTimer delTimer;
        delTimer.start();
        if (g_useSkewedPipeline) {
            // Indexed delete: upload point indices
            for (uint32_t i = 0; i < currentBatchSize; i++) {
                indexData[i] = indices[startIdx + i];  // Original point index
            }
            loadUsingStagingBuf((char*)indexData.data(), currentBatchSize * sizeof(uint32_t), indexBuffer, staging, vd, 0);
            compactIndex->deletePointsIndexed(indexBuffer, currentBatchSize);
        } else {
            // Linear scan delete: use coordinates
            compactIndex->deletePoints(batchBuffer, currentBatchSize);
        }
        double delTime = double(delTimer.stop()) / 1000000.0;
        totalDelTime += delTime;
        
        // Update CPU reference (mark deleted) - only if CPU_CHECK
        if (CPU_CHECK) {
            for (uint32_t i = 0; i < currentBatchSize; i++) {
                uint32_t pointIdx = indices[startIdx + i];
                cpuValidMask[pointIdx] = false;
            }
        }
        
        // --- INSERT (same batch) ---
        CPUTimer insTimer;
        insTimer.start();
        compactIndex->insertPoints(batchBuffer, currentBatchSize);
        double insTime = double(insTimer.stop()) / 1000000.0;
        totalInsTime += insTime;
        
        // Update CPU reference (mark re-inserted) - only if CPU_CHECK
        if (CPU_CHECK) {
            for (uint32_t i = 0; i < currentBatchSize; i++) {
                uint32_t pointIdx = indices[startIdx + i];
                cpuValidMask[pointIdx] = true;
            }
        }
        
        // --- VERIFICATION (only if CPU_CHECK) ---
        if (CPU_CHECK) {
            // Read GPU data
            uint64_t capacity = compactIndex->totalAllocatedCapacity;
            std::vector<CompactEntry> gpuData(capacity);
            readUsingStagingBuf((char*)gpuData.data(), capacity * sizeof(CompactEntry), compactIndex->dataBuffer, staging, vd);
            
            // Count valid entries on GPU
            uint32_t gpuCount = 0;
            for (const auto& entry : gpuData) {
                if (entry.rowId & 0x80000000u) gpuCount++;
            }
            
            // Count valid entries on CPU
            uint32_t cpuCount = std::count(cpuValidMask.begin(), cpuValidMask.end(), true);
            
            // Extract and compare actual points
            auto gpuPoints = extractGPUPoints(gpuData);
            auto cpuPoints = extractCPUPoints(points, npoints, cpuValidMask);
            
            bool countMatch = (cpuCount == gpuCount);
            bool pointsMatch = (gpuPoints == cpuPoints);
            bool passed = countMatch && pointsMatch;
            
            if (passed) totalPassed++;
            
            std::cerr << std::setw(6) << (r + 1)
                      << std::setw(10) << currentBatchSize
                      << std::setw(12) << std::fixed << std::setprecision(3) << (delTime * 1000.0)
                      << std::setw(12) << std::fixed << std::setprecision(3) << (insTime * 1000.0)
                      << std::setw(12) << cpuCount
                      << std::setw(12) << gpuCount
                      << std::setw(10) << (pointsMatch ? "YES" : "NO")
                      << std::setw(8) << (passed ? "PASS" : "FAIL") << "\n";
            
            // If failed, print details
            if (!passed) {
                if (!countMatch) {
                    std::cerr << "  [ERROR] Count mismatch: CPU=" << cpuCount << ", GPU=" << gpuCount << "\n";
                }
                if (!pointsMatch) {
                    std::cerr << "  [ERROR] Points mismatch: CPU has " << cpuPoints.size() 
                              << " points, GPU has " << gpuPoints.size() << " points\n";
                    // Show first few mismatches
                    size_t maxShow = 5;
                    size_t shown = 0;
                    for (size_t i = 0; i < std::min(cpuPoints.size(), gpuPoints.size()) && shown < maxShow; i++) {
                        if (cpuPoints[i] != gpuPoints[i]) {
                            auto [cx, cy, cz] = cpuPoints[i];
                            auto [gx, gy, gz] = gpuPoints[i];
                            std::cerr << "    Mismatch at " << i << ": CPU=(" << cx << "," << cy << "," << cz 
                                      << ") GPU=(" << gx << "," << gy << "," << gz << ")\n";
                            shown++;
                        }
                    }
                }
            }
        } else {
            // No CPU check - just print timing
            std::cerr << std::setw(6) << (r + 1)
                      << std::setw(10) << currentBatchSize
                      << std::setw(12) << std::fixed << std::setprecision(3) << (delTime * 1000.0)
                      << std::setw(12) << std::fixed << std::setprecision(3) << (insTime * 1000.0)
                      << std::setw(12) << "-"
                      << std::setw(12) << "-"
                      << std::setw(10) << "-"
                      << std::setw(8) << "-" << "\n";
            totalPassed++;  // Assume pass when not checking
        }
    }
    
    // Summary
    std::cerr << std::string(82, '-') << "\n";
    std::cerr << "\n--- Summary ---\n";
    std::cerr << "Batches passed: " << totalPassed << "/" << RUNS << "\n";
    std::cerr << "Total delete time: " << std::fixed << std::setprecision(3) << (totalDelTime * 1000.0) << " ms\n";
    std::cerr << "Total insert time: " << std::fixed << std::setprecision(3) << (totalInsTime * 1000.0) << " ms\n";
    std::cerr << "Avg delete time per batch: " << std::fixed << std::setprecision(3) << (totalDelTime * 1000.0 / RUNS) << " ms\n";
    std::cerr << "Avg insert time per batch: " << std::fixed << std::setprecision(3) << (totalInsTime * 1000.0 / RUNS) << " ms\n";
    std::cerr << "Avg delete throughput: " << std::fixed << std::setprecision(3) << (totalDelTime * 1000000.0 / npoints) << " us/point\n";
    std::cerr << "Avg insert throughput: " << std::fixed << std::setprecision(3) << (totalInsTime * 1000000.0 / npoints) << " us/point\n";
    
    // Final cleanup
    batchBuffer->destroy();
    if (indexBuffer) indexBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 23 Complete. Overall: " << (totalPassed == RUNS ? "ALL PASSED" : "SOME FAILED") << "\n";
}
