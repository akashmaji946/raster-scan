#include "RunModes.hpp"
#include "../CompactScanIndex.hpp"
#include "../BufferPool.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>

// TPC-C Constants
static constexpr int32_t kDistrictsPerWarehouse = 10;
static constexpr int32_t kCustomerPerDistrict   = 3000;
static constexpr int64_t kCustomersPerWarehouse = kDistrictsPerWarehouse * kCustomerPerDistrict; // 30,000

// Generate TPC-C Customer table data with 3 columns: W_ID, D_ID, C_ID
// Returns data in column-major format: [W_ID...][D_ID...][C_ID...]
static void generateTPCCData(
    int64_t targetCustomers,
    std::vector<uint32_t>& data,
    uint32_t& minW, uint32_t& maxW,
    uint32_t& minD, uint32_t& maxD,
    uint32_t& minC, uint32_t& maxC
) {
    const int64_t warehouseCount = (targetCustomers + kCustomersPerWarehouse - 1) / kCustomersPerWarehouse;
    
    std::cerr << "[TPC-C] Target customers: " << targetCustomers << "\n";
    std::cerr << "[TPC-C] Warehouses needed: " << warehouseCount << "\n";
    std::cerr << "[TPC-C] Customers per warehouse: " << kCustomersPerWarehouse << "\n";
    
    // Allocate column-major storage
    data.resize(targetCustomers * 3);
    uint32_t* W = data.data();
    uint32_t* D = data.data() + targetCustomers;
    uint32_t* C = data.data() + 2 * targetCustomers;
    
    int64_t count = 0;
    
    // Generate exact projection of TPC-C Customer table
    for (int32_t c_w_id = 1; c_w_id <= warehouseCount && count < targetCustomers; ++c_w_id) {
        for (int32_t c_d_id = 1; c_d_id <= kDistrictsPerWarehouse && count < targetCustomers; ++c_d_id) {
            for (int32_t c_id = 1; c_id <= kCustomerPerDistrict && count < targetCustomers; ++c_id) {
                W[count] = static_cast<uint32_t>(c_w_id);
                D[count] = static_cast<uint32_t>(c_d_id);
                C[count] = static_cast<uint32_t>(c_id);
                ++count;
            }
        }
    }
    
    // Compute min/max for each column
    minW = 1; maxW = static_cast<uint32_t>(warehouseCount);
    minD = 1; maxD = kDistrictsPerWarehouse;
    minC = 1; maxC = kCustomerPerDistrict;
    
    std::cerr << "[TPC-C] Generated " << count << " customers\n";
    std::cerr << "[TPC-C] W_ID range: [" << minW << ", " << maxW << "]\n";
    std::cerr << "[TPC-C] D_ID range: [" << minD << ", " << maxD << "]\n";
    std::cerr << "[TPC-C] C_ID range: [" << minC << ", " << maxC << "]\n";
}

// Save TPC-C data to binary file (column-major format)
static void saveTPCCData(const std::string& filename, const std::vector<uint32_t>& data, uint32_t npoints) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "[TPC-C] ERROR: Failed to open output file: " << filename << "\n";
        return;
    }
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint32_t));
    out.close();
    std::cerr << "[TPC-C] Saved to: " << filename << " (" 
              << (data.size() * sizeof(uint32_t) / (1024.0 * 1024.0)) << " MB)\n";
}

void testTPCCBenchmark(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    // dataId is repurposed as scale factor indicator:
    // 0 = 100K customers (~3 warehouses)
    // 1 = 1M customers (~34 warehouses)
    // 2 = 10M customers (~334 warehouses)
    // 3 = 50M customers (~1667 warehouses)
    // 4 = 100M customers (~3334 warehouses)
    
    static const std::vector<int64_t> scaleFactors = {
        100000,      // 100K
        1000000,     // 1M
        10000000,    // 10M
        50000000,    // 50M
        100000000    // 100M
    };
    
    static const std::vector<std::string> scaleNames = {
        "100K", "1M", "10M", "50M", "100M"
    };
    
    int scaleIdx = std::min(dataId, (int)scaleFactors.size() - 1);
    int64_t targetCustomers = scaleFactors[scaleIdx];
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 24: TPC-C Customer Table Benchmark\n";
    std::cerr << "Scale: " << scaleNames[scaleIdx] << " (" << targetCustomers << " customers)\n";
    std::cerr << "========================================\n";
    
    // =========================================================
    // Generate TPC-C Data
    // =========================================================
    std::vector<uint32_t> data;
    uint32_t minW, maxW, minD, maxD, minC, maxC;
    
    CPUTimer genTimer;
    genTimer.start();
    generateTPCCData(targetCustomers, data, minW, maxW, minD, maxD, minC, maxC);
    double genTime = double(genTimer.stop()) / 1000000.0;
    std::cerr << "[TPC-C] Data generation time: " << (genTime * 1000.0) << " ms\n";
    
    uint32_t npoints = static_cast<uint32_t>(targetCustomers);
    int ncols = 3;
    
    // Optionally save to file
    // saveTPCCData("tpcc_" + scaleNames[scaleIdx] + ".bin", data, npoints);
    
    // =========================================================
    // Upload to GPU
    // =========================================================
    GPUMemoryTool::printGPUMemoryStatus(vd, "Before TPC-C data upload");
    
    vkcore::PBuffer pointsBuffer(new Buffer(vd));
    size_t dataSize = data.size() * sizeof(uint32_t);
    pointsBuffer->create(dataSize, 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, 
        MemoryType::Internal);
    
    loadUsingStagingBuf((char*)data.data(), dataSize, pointsBuffer, staging, vd, 0);
    
    GPUMemoryTool::printGPUMemoryStatus(vd, "After TPC-C data upload");
    std::cerr << "[TPC-C] Dataset: " << npoints << " points, " << ncols << " columns\n";
    
    // Min/Max arrays for buildIndex
    std::vector<uint32_t> minval = {minW, minD, minC};
    std::vector<uint32_t> maxval = {maxW, maxD, maxC};
    
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
        std::cerr << "[Mode24] Using INDEXED delete (O(1) per delete)\n";
    } else {
        std::cerr << "[Mode24] Using LINEAR SCAN delete (O(bin_size) per delete)\n";
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
    const uint32_t NUM_BATCHES = 10000000;
    const int RUNS = 10000;
    const bool USE_RANDOM_BATCHES = true;
    const bool CPU_CHECK = false;
    uint32_t batchSize = npoints / NUM_BATCHES;
    
    std::cerr << "\n--- Batch Delete/Insert Cycles ---\n";
    std::cerr << "Total points: " << npoints << "\n";
    std::cerr << "Number of batches: " << NUM_BATCHES << "\n";
    std::cerr << "Batch size: " << batchSize << " points\n";
    std::cerr << "Runs: " << RUNS << "\n";
    std::cerr << "Batch order: " << (USE_RANDOM_BATCHES ? "RANDOM" : "SEQUENTIAL") << "\n";
    std::cerr << "CPU verification: " << (CPU_CHECK ? "ON" : "OFF") << "\n";
    
    // Create indices for batch assignment
    std::vector<uint32_t> indices(npoints);
    std::iota(indices.begin(), indices.end(), 0);
    if (USE_RANDOM_BATCHES) {
        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    // CPU reference (only if CPU_CHECK)
    std::vector<bool> cpuValidMask;
    if (CPU_CHECK) {
        cpuValidMask.resize(npoints, true);
    }
    
    // Allocate batch buffer (row-major: [x0,y0,z0, x1,y1,z1, ...])
    vkcore::PBuffer batchBuffer(new Buffer(vd));
    batchBuffer->create(batchSize * 3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    std::vector<uint32_t> batchData(batchSize * 3);
    
    // For indexed delete
    vkcore::PBuffer indexBuffer;
    std::vector<uint32_t> indexData;
    if (g_useSkewedPipeline) {
        indexBuffer = std::make_shared<Buffer>(vd);
        indexBuffer->create(batchSize * sizeof(uint32_t), 
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
            MemoryType::Internal);
        indexData.resize(batchSize);
    }
    
    // Statistics
    uint32_t totalPassed = 0;
    double totalDelTime = 0, totalInsTime = 0;
    
    std::cerr << "\n";
    std::cerr << std::setw(6) << "Run" 
              << std::setw(10) << "Size"
              << std::setw(12) << "Del(ms)"
              << std::setw(12) << "Ins(ms)"
              << std::setw(12) << "Del(us/pt)"
              << std::setw(12) << "Ins(us/pt)" << "\n";
    std::cerr << std::string(64, '-') << "\n";
    
    for (int r = 0; r < RUNS; r++) {
        // Use batch r % NUM_BATCHES
        uint32_t b = r % NUM_BATCHES;
        uint32_t startIdx = b * batchSize;
        uint32_t endIdx = (b == NUM_BATCHES - 1) ? npoints : (b + 1) * batchSize;
        uint32_t currentBatchSize = endIdx - startIdx;
        
        // Prepare batch data (row-major format)
        for (uint32_t i = 0; i < currentBatchSize; i++) {
            uint32_t pointIdx = indices[startIdx + i];
            batchData[i * 3 + 0] = data[pointIdx];                  // W_ID
            batchData[i * 3 + 1] = data[npoints + pointIdx];        // D_ID
            batchData[i * 3 + 2] = data[2 * npoints + pointIdx];    // C_ID
        }
        
        // Upload batch to GPU
        loadUsingStagingBuf((char*)batchData.data(), currentBatchSize * 3 * sizeof(uint32_t), batchBuffer, staging, vd, 0);
        
        // --- DELETE ---
        CPUTimer delTimer;
        delTimer.start();
        if (g_useSkewedPipeline) {
            for (uint32_t i = 0; i < currentBatchSize; i++) {
                indexData[i] = indices[startIdx + i];
            }
            loadUsingStagingBuf((char*)indexData.data(), currentBatchSize * sizeof(uint32_t), indexBuffer, staging, vd, 0);
            compactIndex->deletePointsIndexed(indexBuffer, currentBatchSize);
        } else {
            compactIndex->deletePoints(batchBuffer, currentBatchSize);
        }
        double delTime = double(delTimer.stop()) / 1000000.0;
        totalDelTime += delTime;
        
        // Update CPU reference if needed
        if (CPU_CHECK) {
            for (uint32_t i = 0; i < currentBatchSize; i++) {
                cpuValidMask[indices[startIdx + i]] = false;
            }
        }
        
        // --- INSERT ---
        CPUTimer insTimer;
        insTimer.start();
        compactIndex->insertPoints(batchBuffer, currentBatchSize);
        double insTime = double(insTimer.stop()) / 1000000.0;
        totalInsTime += insTime;
        
        // Update CPU reference if needed
        if (CPU_CHECK) {
            for (uint32_t i = 0; i < currentBatchSize; i++) {
                cpuValidMask[indices[startIdx + i]] = true;
            }
        }
        
        // Print timing
        double delUsPerPt = (delTime * 1000000.0) / currentBatchSize;
        double insUsPerPt = (insTime * 1000000.0) / currentBatchSize;
        
        std::cerr << std::setw(6) << (r + 1)
                  << std::setw(10) << currentBatchSize
                  << std::setw(12) << std::fixed << std::setprecision(3) << (delTime * 1000.0)
                  << std::setw(12) << std::fixed << std::setprecision(3) << (insTime * 1000.0)
                  << std::setw(12) << std::fixed << std::setprecision(3) << delUsPerPt
                  << std::setw(12) << std::fixed << std::setprecision(3) << insUsPerPt << "\n";
        
        totalPassed++;
    }
    
    // Summary
    std::cerr << std::string(64, '-') << "\n";
    std::cerr << "\n--- Summary ---\n";
    std::cerr << "Scale: " << scaleNames[scaleIdx] << " (" << npoints << " customers)\n";
    std::cerr << "Runs completed: " << totalPassed << "/" << RUNS << "\n";
    std::cerr << "Total delete time: " << std::fixed << std::setprecision(3) << (totalDelTime * 1000.0) << " ms\n";
    std::cerr << "Total insert time: " << std::fixed << std::setprecision(3) << (totalInsTime * 1000.0) << " ms\n";
    std::cerr << "Avg delete time per run: " << std::fixed << std::setprecision(3) << (totalDelTime * 1000.0 / RUNS) << " ms\n";
    std::cerr << "Avg insert time per run: " << std::fixed << std::setprecision(3) << (totalInsTime * 1000.0 / RUNS) << " ms\n";
    std::cerr << "Avg delete throughput: " << std::fixed << std::setprecision(3) << (totalDelTime * 1000000.0 / (RUNS * batchSize)) << " us/point\n";
    std::cerr << "Avg insert throughput: " << std::fixed << std::setprecision(3) << (totalInsTime * 1000000.0 / (RUNS * batchSize)) << " us/point\n";
    
    // Cleanup
    batchBuffer->destroy();
    if (indexBuffer) indexBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 24 (TPC-C) Complete.\n";
}
