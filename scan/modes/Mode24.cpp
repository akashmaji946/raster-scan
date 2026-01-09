#include "RunModes.hpp"
#include "../CompactScanIndex.hpp"
#include "../RasterScan2D.hpp"
#include "../BufferPool.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>

// Set USE_RASTER to 1 to run RasterScan2D, 0 to run CompactScanIndex
#ifndef USE_RASTER
#define USE_RASTER 1

#endif

// TPC-C Constants
static constexpr int32_t kDistrictsPerWarehouse = 100;
static constexpr int32_t kCustomerPerDistrict   = 30000;
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
    (void)npoints;  // unused
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

// Generate queries with selectivities 10%, 20%, ..., 100%
// For TPC-C data, we use CENTERED strategy since data is structured
static void generateTPCCQueries(
    const std::vector<uint32_t>& minval, 
    const std::vector<uint32_t>& maxval,
    int ncols,
    std::vector<uint32_t>& targets,
    int numQueries = 10
) {
    targets.resize(numQueries * 6);  // 6 values per query (x1,x2,y1,y2,z1,z2)
    
    for (int q = 0; q < numQueries; q++) {
        double overallSelectivity = (q + 1) * 0.1;  // 10%, 20%, ..., 100%
        double perDimSelectivity = std::pow(overallSelectivity, 1.0 / ncols);
        
        for (int c = 0; c < ncols; c++) {
            uint64_t range = (uint64_t)maxval[c] - (uint64_t)minval[c];
            uint64_t queryRange = (uint64_t)(range * perDimSelectivity);
            
            // Center the query in the data range
            uint64_t margin = (range - queryRange) / 2;
            uint32_t lo = minval[c] + (uint32_t)margin;
            uint32_t hi = minval[c] + (uint32_t)(margin + queryRange);
            
            // Store as x1,x2,y1,y2,z1,z2 format
            targets[q * 6 + c * 2] = lo;
            targets[q * 6 + c * 2 + 1] = hi;
        }
        // Fill remaining dimensions if ncols < 3
        for (int c = ncols; c < 3; c++) {
            targets[q * 6 + c * 2] = 0;
            targets[q * 6 + c * 2 + 1] = 0xFFFFFFFF;
        }
    }
    
    std::cerr << "[TPC-C] Generated " << numQueries << " queries (selectivity 10%-100%)\n";
}

void testTPCCBenchmark(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    // dataId is repurposed as scale factor indicator:
    // 0 = 100K customers (~3 warehouses)
    // 1 = 1M customers (~34 warehouses)
    // 2 = 10M customers (~334 warehouses)
    // 3 = 25M customers (~834 warehouses)
    // 4 = 50M customers (~1667 warehouses)
    // 5 = 75M customers (~2500 warehouses)
    // 6 = 100M customers (~3334 warehouses)
    // 7 = 250M customers (~6667 warehouses)
    // 8 = 500M customers (~16666 warehouses)
    // 9 = 750M customers
    // 10 = 1B customers (~33333 warehouses)
    
    static const std::vector<int64_t> scaleFactors = {
        100000,      // 100K
        1000000,     // 1M
        10000000,    // 10M
        25000000,    // 25M
        50000000,    // 50M
        75000000,    // 75M
        100000000,   // 100M
        250000000,   // 250M
        500000000,   // 500M
        750000000,   // 750M
        1000000000   // 1B
    };
    
    static const std::vector<std::string> scaleNames = {
        "100K", "1M", "10M", "25M", "50M", "75M", "100M", "250M", "500M", "750M", "1B"
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

#if USE_RASTER == 1
    // =========================================================
    // Build RasterScan2D Index
    // =========================================================
    std::cerr << "\n--- [RasterScan2D] ---\n";
    
    vkcore::ReduceMax *reduce = (vkcore::ReduceMax *) op.getFunction(vkcore::FunctionType::ReduceMax);
    PBufferCache bufs(new CommonBufferPool(vd));
    
    RasterScan2D rs(vd, bufs, scan, reduce, ncols);
    rs.initalize();
    GPUMemoryTool::printGPUMemoryStatus(vd, "Before RasterScan2D build");
    
    std::cerr << "\nBuilding RasterScan2D Index (taking median of 5 runs)...\n";
    std::vector<double> rsBuildTimes;
    rsBuildTimes.reserve(5);
    PRasterIndex rsIndex;
    for(int k=0; k<5; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";
        CPUTimer buildTimer;
        buildTimer.start();
        rsIndex = rs.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
        double bt = double(buildTimer.stop()) / 1000000.0;
        rsBuildTimes.push_back(bt);
    }
    std::sort(rsBuildTimes.begin(), rsBuildTimes.end());
    double buildTime = rsBuildTimes[rsBuildTimes.size() / 2];
    std::cerr << "\n\n==> RasterScan2D Index Build time: " << (buildTime * 1000.0) << " ms\n";
    GPUMemoryTool::printGPUMemoryStatus(vd, "After RasterScan2D build");
    
#else
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
    
    std::cerr << "\nBuilding Compact Index (taking median of 5 runs)...\n";
    std::vector<double> compactBuildTimes;
    compactBuildTimes.reserve(5);
    for(int k=0; k<5; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";
        CPUTimer buildTimer;
        buildTimer.start();
        compactIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
        double bt = double(buildTimer.stop()) / 1000000.0;
        compactBuildTimes.push_back(bt);
    }
    std::sort(compactBuildTimes.begin(), compactBuildTimes.end());
    double buildTime = compactBuildTimes[compactBuildTimes.size() / 2];
    std::cerr << "\n\n ==> Compact Index Build time: " << (buildTime * 1000.0) << " ms\n";
    GPUMemoryTool::printGPUMemoryStatus(vd, "After CompactScanIndex build");
#endif
    
    // =========================================================
    // Query Execution
    // =========================================================
    const int numQueries = 10;
    std::vector<uint32_t> targets;
    generateTPCCQueries(minval, maxval, ncols, targets, numQueries);
    
    // Result buffer size
    uint32_t resultSizeUints = (npoints + 31) / 32;
    
    // Query buffer
    vkcore::PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);

#if USE_RASTER == 1
    // =========================================================
    // RasterScan2D Query Performance
    // =========================================================
    std::cerr << "\n--- RasterScan2D Query Performance ---\n";
    
    double rsTotTime = 0;
    for (int i = 0; i < numQueries; i++) {
        int in = i * 6;
        // RasterScan2D format: x1, y1, x2, y2, z1, z2
        std::vector<uint32_t> queries = {targets[in], targets[in+2], targets[in+1], targets[in+3], targets[in+4], targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
        
        CPUTimer qTimer;
        qTimer.start();
        rs.runRangeQueries(rsIndex, queryBuffer, 1);
        double t = double(qTimer.stop()) / 1000000.0;
        rsTotTime += t;
        
        // Read back and count
        std::vector<uint32_t> result(resultSizeUints);
        readUsingStagingBuf((char *)result.data(), resultSizeUints * sizeof(uint32_t), bufs->resBuffer, staging, vd);
        
        uint32_t count = 0;
        for(uint32_t val : result) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << " (" << ((i+1)*10) << "%): " << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Result Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << (rsTotTime * 1000.0 / numQueries) << " ms\n";
    
    std::cerr << "\n[RasterScan2D] Note: Delete/Insert not supported.\n";
    
    // Cleanup
    queryBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 24 (TPC-C with RasterScan2D) Complete.\n";
    return;
#else
    // =========================================================
    // CompactScanIndex Query Performance
    // =========================================================
    vkcore::PBuffer resultBuffer(new Buffer(vd));
    resultBuffer->create(resultSizeUints * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    std::cerr << "\n--- Compact Index Query Performance ---\n";
    double compactTotTime = 0;
    for (int i = 0; i < numQueries; i++) {
        int in = i * 6;
        // CompactScanIndex format: x1, x2, y1, y2, z1, z2
        std::vector<uint32_t> queries = {targets[in], targets[in+1], targets[in+2], targets[in+3], targets[in+4], targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
        
        CPUTimer qTimer;
        qTimer.start();
        compactIndex->runRangeQueries(queryBuffer, 1, resultBuffer);
        double t = double(qTimer.stop()) / 1000000.0;
        compactTotTime += t;
        
        // Read back and count
        std::vector<uint32_t> result(resultSizeUints);
        readUsingStagingBuf((char *)result.data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
        
        uint32_t count = 0;
        for(uint32_t val : result) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << " (" << ((i+1)*10) << "%): " << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Result Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << (compactTotTime * 1000.0 / numQueries) << " ms\n";

    // =========================================================
    // Batch Delete/Insert Cycles (CompactScanIndex only)
    // =========================================================
    
    // Configuration
    const uint32_t NUM_BATCHES = 10000000;
    const int RUNS = 2;
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
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
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
    queryBuffer->destroy();
    resultBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 24 (TPC-C) Complete.\n";
#endif  // USE_RASTER
}
