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

// Mode 44: TPC-C Benchmark with Query Performance After Each Update Cycle
// Based on Mode 24, but runs queries after each batch delete/insert


#ifndef BUILD_COUNT
#define BUILD_COUNT 1
#endif

#ifndef QUERY_COUNT
#define QUERY_COUNT 1
#endif

// TPC-C Constants
static constexpr int32_t kDistrictsPerWarehouse = 100;
static constexpr int32_t kCustomerPerDistrict   = 30000;
static constexpr int64_t kCustomersPerWarehouse = kDistrictsPerWarehouse * kCustomerPerDistrict;

// Generate TPC-C Customer table data with 3 columns: W_ID, D_ID, C_ID
static void generateTPCCData44(
    int64_t targetCustomers,
    std::vector<uint32_t>& data,
    uint32_t& minW, uint32_t& maxW,
    uint32_t& minD, uint32_t& maxD,
    uint32_t& minC, uint32_t& maxC
) {
    const int64_t warehouseCount = (targetCustomers + kCustomersPerWarehouse - 1) / kCustomersPerWarehouse;
    
    std::cerr << "[TPC-C] Target customers: " << targetCustomers << "\n";
    std::cerr << "[TPC-C] Warehouses needed: " << warehouseCount << "\n";
    
    data.resize(targetCustomers * 3);
    uint32_t* W = data.data();
    uint32_t* D = data.data() + targetCustomers;
    uint32_t* C = data.data() + 2 * targetCustomers;
    
    int64_t count = 0;
    
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
    
    minW = 1; maxW = static_cast<uint32_t>(warehouseCount);
    minD = 1; maxD = kDistrictsPerWarehouse;
    minC = 1; maxC = kCustomerPerDistrict;
    
    std::cerr << "[TPC-C] Generated " << count << " customers\n";
}

// Generate queries with selectivities 10%, 20%, ..., 100%
static void generateTPCCQueries44(
    const std::vector<uint32_t>& minval, 
    const std::vector<uint32_t>& maxval,
    int ncols,
    std::vector<uint32_t>& targets,
    int numQueries = 10
) {
    targets.clear();
    targets.reserve(numQueries * 6);
    
    for (int q = 0; q < numQueries; q++) {
        double overallSelectivity = (q + 1) * 0.1;
        double perDimSelectivity = std::pow(overallSelectivity, 1.0 / ncols);
        
        bool isFullRange = (q == numQueries - 1);
        
        for (int c = 0; c < ncols; c++) {
            uint32_t lo, hi;
            
            if (isFullRange) {
                lo = minval[c];
                hi = maxval[c];
            } else {
                uint64_t range = (uint64_t)maxval[c] - (uint64_t)minval[c];
                uint64_t queryRange = (uint64_t)(range * perDimSelectivity);
                uint64_t margin = (range - queryRange) / 2;
                lo = minval[c] + (uint32_t)margin;
                hi = minval[c] + (uint32_t)(margin + queryRange);
            }
            
            targets.push_back(lo);
            targets.push_back(hi);
        }
        
        for (int c = ncols; c < 3; c++) {
            targets.push_back(0);
            targets.push_back(UINT32_MAX);
        }
    }
}

// Run all queries and return average and median time (in ms)
static std::pair<double, double> runQueriesAndGetStats(
    PCompactScanIndex& compactIndex,
    vkcore::PBuffer& queryBuffer,
    vkcore::PBuffer& resultBuffer,
    vkcore::PBuffer& staging,
    vkcore::PVkDevice& vd,
    const std::vector<uint32_t>& targets,
    uint32_t resultSizeUints,
    int numQueries,
    bool printDetails = false
) {
    std::vector<double> queryTimes;
    queryTimes.reserve(numQueries);
    
    for (int i = 0; i < numQueries; i++) {
        int in = i * 6;
        // CompactScanIndex format: x1, x2, y1, y2, z1, z2
        std::vector<uint32_t> queries = {targets[in], targets[in+1], targets[in+2], targets[in+3], targets[in+4], targets[in+5]};

        std::vector<double> qt;
        qt.reserve(QUERY_COUNT);
        for(int r = 0; r < QUERY_COUNT; r++) {
            loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            CPUTimer qTimer;
            qTimer.start();
            compactIndex->runRangeQueries(queryBuffer, 1, resultBuffer);
            double t = double(qTimer.stop()) / 1000000.0;
            qt.push_back(t);
        }
        std::sort(qt.begin(), qt.end());
        double t = qt[qt.size() / 2];
        queryTimes.push_back(t * 1000.0);  // Store in ms
        
        std::cout << "MAX QUERY TIME: " << std::fixed << std::setprecision(3) << qt.back() * 1000 << " ms\n";

        if (printDetails) {
            std::vector<uint32_t> result(resultSizeUints);
            readUsingStagingBuf((char *)result.data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
            uint32_t count = 0;
            for(uint32_t val : result) count += __builtin_popcount(val);
            std::cerr << "  Q" << (i+1) << ": " << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Count: " << count << "\n";
        }
    }
    
    // Compute average
    double totalTime = 0;
    for (double t : queryTimes) totalTime += t;
    double avgTime = totalTime / numQueries;
    
    // Compute median
    std::sort(queryTimes.begin(), queryTimes.end());
    double medianTime = queryTimes[queryTimes.size() / 2];
    
    return {avgTime, medianTime};
}

void testTPCCWithQueryAfterUpdate(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging) {
    using namespace vkcore;
    
    // dataId is repurposed as scale factor indicator (same as Mode24):
    // 0 = 100K, 1 = 1M, 2 = 10M, 3 = 25M, 4 = 50M, 5 = 75M,
    // 6 = 100M, 7 = 250M, 8 = 500M, 9 = 750M, 10 = 1B
    
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
    std::cerr << "MODE 44: TPC-C with Query After Each Update\n";
    std::cerr << "Scale: " << scaleNames[scaleIdx] << " (" << targetCustomers << " customers)\n";
    std::cerr << "========================================\n";
    
    // Generate TPC-C data
    std::vector<uint32_t> data;
    uint32_t minW, maxW, minD, maxD, minC, maxC;
    generateTPCCData44(targetCustomers, data, minW, maxW, minD, maxD, minC, maxC);
    
    uint32_t npoints = static_cast<uint32_t>(targetCustomers);
    int32_t ncols = 3;
    
    std::vector<uint32_t> minval = {minW, minD, minC};
    std::vector<uint32_t> maxval = {maxW, maxD, maxC};
    
    // Upload data to GPU
    size_t dataSize = data.size() * sizeof(uint32_t);
    vkcore::PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(dataSize, 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    loadUsingStagingBuf((char*)data.data(), dataSize, pointsBuffer, staging, vd, 0);
    
    // Initialize operators
    SinglePassScan* scan = new SinglePassScan(vd);
    scan->initialize();
    
    // Build CompactScanIndex
    std::cerr << "\nBuilding Compact Index...\n";
    PCompactScanIndex compactIndex = std::make_shared<CompactScanIndex>(vd, ncols, scan);
    compactIndex->useIndexedDelete = g_useSkewedPipeline;
    compactIndex->initialize();
    
    CPUTimer buildTimer;
    buildTimer.start();
    compactIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    double buildTime = double(buildTimer.stop()) / 1000000.0;
    std::cerr << "Build time: " << std::fixed << std::setprecision(3) << (buildTime * 1000.0) << " ms\n";
    
    // Setup query infrastructure
    const int numQueries = 10;
    std::vector<uint32_t> targets;
    generateTPCCQueries44(minval, maxval, ncols, targets, numQueries);
    
    uint32_t resultSizeUints = (npoints + 31) / 32;
    
    vkcore::PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    vkcore::PBuffer resultBuffer(new Buffer(vd));
    resultBuffer->create(resultSizeUints * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    // Helper lambda to compute total extent
    auto computeTotalExtent = [&]() -> uint64_t {
        uint32_t numBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
        std::vector<uint32_t> extents(numBins);
        readUsingStagingBuf((char*)extents.data(), numBins * sizeof(uint32_t), compactIndex->extentBuffer, staging, vd);
        uint64_t total = 0;
        for (uint32_t e : extents) total += e;
        return total;
    };
    
    uint64_t initialTotalExtent = computeTotalExtent();
    std::cerr << "\nInitial total extent (sum of bin sizes): " << initialTotalExtent << "\n";
    
    // Initial query performance
    std::cerr << "\n--- Initial Query Performance ---\n";
    auto [initialAvgTime, initialMedianTime] = runQueriesAndGetStats(compactIndex, queryBuffer, resultBuffer, staging, vd, 
                                                     targets, resultSizeUints, numQueries, false);
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << initialAvgTime << " ms\n";
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << initialMedianTime << " ms\n";
    // std::cerr << "Max Query Time: " << std::fixed << std::setprecision(3) << initialMaxTime << " ms\n";
    
    // =========================================================
    // Batch Delete/Insert Cycles with Query After Each
    // =========================================================
    const uint32_t NUM_BATCHES = 2;
    const int RUNS = 2;
    uint32_t batchSize = npoints / NUM_BATCHES;
    
    std::cerr << "\n--- Batch Delete/Insert with Query Performance ---\n";
    std::cerr << "Total points: " << npoints << "\n";
    std::cerr << "Number of batches: " << NUM_BATCHES << "\n";
    std::cerr << "Batch size: " << batchSize << " points\n";
    std::cerr << "Runs: " << RUNS << "\n";
    
    // Create indices for batch assignment
    std::vector<uint32_t> indices(npoints);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    // Allocate batch buffer
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
    
    // Print header
    std::cerr << "\n";
    std::cerr << std::setw(6) << "Run" 
              << std::setw(10) << "Size"
              << std::setw(12) << "Del(ms)"
              << std::setw(12) << "Ins(ms)"
              << std::setw(14) << "AvgQuery(ms)" << "\n";
    std::cerr << std::string(54, '-') << "\n";
    
    double totalDelTime = 0, totalInsTime = 0, totalQueryTime = 0;
    std::vector<double> allQueryTimes;  // Store all avg query times for median
    allQueryTimes.reserve(RUNS);
    
    for (int r = 0; r < RUNS; r++) {
        
        uint32_t b = r % NUM_BATCHES;
        uint32_t startIdx = b * batchSize;
        uint32_t endIdx = (b == NUM_BATCHES - 1) ? npoints : (b + 1) * batchSize;
        uint32_t currentBatchSize = endIdx - startIdx;
        
        // Prepare batch data (row-major format)
        for (uint32_t i = 0; i < currentBatchSize; i++) {
            uint32_t pointIdx = indices[startIdx + i];
            batchData[i * 3 + 0] = data[pointIdx];
            batchData[i * 3 + 1] = data[npoints + pointIdx];
            batchData[i * 3 + 2] = data[2 * npoints + pointIdx];
        }
        
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
        
        // --- INSERT ---
        CPUTimer insTimer;
        insTimer.start();
        compactIndex->insertPoints(batchBuffer, currentBatchSize);
        double insTime = double(insTimer.stop()) / 1000000.0;
        totalInsTime += insTime;
        
        // --- RUN QUERIES ---
        auto [avgQueryTime, medQueryTime] = runQueriesAndGetStats(compactIndex, queryBuffer, resultBuffer, staging, vd,
                                                       targets, resultSizeUints, numQueries, false);
        totalQueryTime += avgQueryTime;
        allQueryTimes.push_back(avgQueryTime);
        
        // Print row
        std::cerr << std::setw(6) << (r + 1)
                  << std::setw(10) << currentBatchSize
                  << std::setw(12) << std::fixed << std::setprecision(3) << (delTime * 1000.0)
                  << std::setw(12) << std::fixed << std::setprecision(3) << (insTime * 1000.0)
                  << std::setw(14) << std::fixed << std::setprecision(3) << avgQueryTime << "\n";
    }
    
    // Summary
    std::cerr << std::string(54, '-') << "\n";
    std::cerr << "\n--- Summary ---\n";
    std::cerr << "Scale: " << scaleNames[scaleIdx] << " (" << npoints << " customers)\n";
    std::cerr << "Runs completed: " << RUNS << "\n";
    std::cerr << "Avg delete time per run: " << std::fixed << std::setprecision(3) << (totalDelTime * 1000.0 / RUNS) << " ms\n";
    std::cerr << "Avg insert time per run: " << std::fixed << std::setprecision(3) << (totalInsTime * 1000.0 / RUNS) << " ms\n";
    std::cerr << "Avg query time (10 queries): " << std::fixed << std::setprecision(3) << (totalQueryTime / RUNS) << " ms\n";
    std::cerr << "Initial avg query time: " << std::fixed << std::setprecision(3) << initialAvgTime << " ms\n";
    std::cerr << "Initial median query time: " << std::fixed << std::setprecision(3) << initialMedianTime << " ms\n";
    
    // Compute final median query time
    std::sort(allQueryTimes.begin(), allQueryTimes.end());
    double finalMedianQueryTime = allQueryTimes[allQueryTimes.size() / 2];
    std::cerr << "Final avg query time: " << std::fixed << std::setprecision(3) << (totalQueryTime / RUNS) << " ms\n";
    std::cerr << "Final median query time: " << std::fixed << std::setprecision(3) << finalMedianQueryTime << " ms\n";
    
    uint64_t finalTotalExtent = computeTotalExtent();
    std::cerr << "\nInitial total extent: " << initialTotalExtent << "\n";
    std::cerr << "Final total extent: " << finalTotalExtent << "\n";
    std::cerr << "Extent growth: " << (finalTotalExtent - initialTotalExtent) << " (" 
              << std::fixed << std::setprecision(2) << (100.0 * (finalTotalExtent - initialTotalExtent) / initialTotalExtent) << "%)\n";
    
    // Cleanup
    batchBuffer->destroy();
    if (indexBuffer) indexBuffer->destroy();
    queryBuffer->destroy();
    resultBuffer->destroy();
    pointsBuffer->destroy();
    delete scan;
    
    std::cerr << "\nMode 44 Complete.\n";
}
