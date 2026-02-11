#include "RunModes.hpp"
#include "../BruteForceIndexUpdatable.hpp"
#include "../BufferPool.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

// Mode 70: BruteForce Scan with Update Support (TPC-C)
// Allocates extra space in dataBuffer for inserts/updates/deletes
// BRUTE_SCALE_FACTOR controls the buffer size multiplier (1.2x, 1.5x, 2.0x, etc.)

#ifndef BUILD_COUNT
#define BUILD_COUNT 3
#endif

#ifndef QUERY_COUNT
#define QUERY_COUNT 3
#endif

// Q controls number of queries: Q=10 means 10%, 20%, ..., 100%
#ifndef Q
#define Q 10
#endif

// TPC-C Constants
static constexpr int32_t kDistrictsPerWarehouse70 = 100;
static constexpr int32_t kCustomerPerDistrict70   = 30000;
static constexpr int64_t kCustomersPerWarehouse70 = kDistrictsPerWarehouse70 * kCustomerPerDistrict70;

// Generate TPC-C Customer table data with 3 columns: W_ID, D_ID, C_ID
static void generateTPCCDataMode70(
    int64_t targetCustomers,
    std::vector<uint32_t>& data,
    uint32_t& minW, uint32_t& maxW,
    uint32_t& minD, uint32_t& maxD,
    uint32_t& minC, uint32_t& maxC
) {
    const int64_t warehouseCount = (targetCustomers + kCustomersPerWarehouse70 - 1) / kCustomersPerWarehouse70;
    
    std::cerr << "[TPC-C] Target customers: " << targetCustomers << "\n";
    std::cerr << "[TPC-C] Warehouses needed: " << warehouseCount << "\n";
    
    data.resize(targetCustomers * 3);
    uint32_t* W = data.data();
    uint32_t* D = data.data() + targetCustomers;
    uint32_t* C = data.data() + 2 * targetCustomers;
    
    int64_t count = 0;
    
    for (int32_t c_w_id = 1; c_w_id <= warehouseCount && count < targetCustomers; ++c_w_id) {
        for (int32_t c_d_id = 1; c_d_id <= kDistrictsPerWarehouse70 && count < targetCustomers; ++c_d_id) {
            for (int32_t c_id = 1; c_id <= kCustomerPerDistrict70 && count < targetCustomers; ++c_id) {
                W[count] = static_cast<uint32_t>(c_w_id);
                D[count] = static_cast<uint32_t>(c_d_id);
                C[count] = static_cast<uint32_t>(c_id);
                ++count;
            }
        }
    }
    
    minW = 1; maxW = static_cast<uint32_t>(warehouseCount);
    minD = 1; maxD = kDistrictsPerWarehouse70;
    minC = 1; maxC = kCustomerPerDistrict70;
    
    std::cerr << "[TPC-C] Generated " << count << " customers\n";
}

// Generate queries with selectivities based on Q
static void generateTPCCQueriesMode70(
    const std::vector<uint32_t>& minval, 
    const std::vector<uint32_t>& maxval,
    int ncols,
    std::vector<uint32_t>& targets,
    int numQueries
) {
    targets.clear();
    targets.reserve(numQueries * 6);
    
    double selectivityStep = 1.0 / numQueries;
    
    for (int q = 0; q < numQueries; q++) {
        double overallSelectivity = (q + 1) * selectivityStep;
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

// Run queries and return stats
static std::pair<double, double> runQueriesMode70(
    PBruteForceIndexUpdatable& index,
    vkcore::PBuffer& queryBuffer,
    vkcore::PBuffer& resultBuffer,
    vkcore::PBuffer& staging,
    vkcore::PVkDevice& vd,
    const std::vector<uint32_t>& targets,
    uint32_t resultSizeUints,
    int numQueries,
    std::vector<uint32_t>* outCounts = nullptr
) {
    std::vector<double> queryTimes;
    queryTimes.reserve(numQueries);
    
    for (int i = 0; i < numQueries; i++) {
        int in = i * 6;
        std::vector<uint32_t> queries = {targets[in], targets[in+1], targets[in+2], targets[in+3], targets[in+4], targets[in+5]};
        
        std::vector<double> qt;
        qt.reserve(QUERY_COUNT);
        for(int r = 0; r < QUERY_COUNT; r++) {
            loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            CPUTimer qTimer;
            qTimer.start();
            index->runRangeQueries(queryBuffer, 1, resultBuffer);
            double t = double(qTimer.stop()) / 1000000.0;
            qt.push_back(t);
        }
        std::sort(qt.begin(), qt.end());
        double t = qt[qt.size() / 2];
        queryTimes.push_back(t * 1000.0);
        
        if (outCounts) {
            std::vector<uint32_t> result(resultSizeUints);
            readUsingStagingBuf((char *)result.data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
            uint32_t count = 0;
            for(uint32_t val : result) count += __builtin_popcount(val);
            outCounts->push_back(count);
        }
    }
    
    double totalTime = 0;
    for (double t : queryTimes) totalTime += t;
    double avgTime = totalTime / numQueries;
    
    std::sort(queryTimes.begin(), queryTimes.end());
    double medianTime = queryTimes[queryTimes.size() / 2];
    
    return {avgTime, medianTime};
}

void testBruteForceUpdateTPCC(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    using namespace vkcore;
    
    static const std::vector<int64_t> scaleFactors = {
        100000, 1000000, 10000000, 25000000, 50000000, 75000000,
        100000000, 250000000, 500000000, 750000000, 1000000000
    };
    static const std::vector<std::string> scaleNames = {
        "100K", "1M", "10M", "25M", "50M", "75M", "100M", "250M", "500M", "750M", "1B"
    };
    
    int scaleIdx = std::min(dataId, (int)scaleFactors.size() - 1);
    int64_t targetCustomers = scaleFactors[scaleIdx];
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 70: BruteForce Scan with Update (TPC-C)\n";
    std::cerr << "Scale: " << scaleNames[scaleIdx] << " (" << targetCustomers << " customers)\n";
    std::cerr << "Scale Factor: " << BRUTE_SCALE_FACTOR << "x\n";
    std::cerr << "========================================\n";
    
    // Generate TPC-C data
    std::vector<uint32_t> data;
    uint32_t minW, maxW, minD, maxD, minC, maxC;
    generateTPCCDataMode70(targetCustomers, data, minW, maxW, minD, maxD, minC, maxC);
    
    uint32_t npoints = static_cast<uint32_t>(targetCustomers);
    int32_t ncols = 3;
    
    std::vector<uint32_t> minval = {minW, minD, minC};
    std::vector<uint32_t> maxval = {maxW, maxD, maxC};
    
    // Shuffle data for realistic workload
    std::vector<uint32_t> indices(npoints);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    // Create shuffled column-major data
    std::vector<uint32_t> shuffledData(npoints * 3);
    for (uint32_t i = 0; i < npoints; i++) {
        uint32_t srcIdx = indices[i];
        shuffledData[i] = data[srcIdx];
        shuffledData[npoints + i] = data[npoints + srcIdx];
        shuffledData[2 * npoints + i] = data[2 * npoints + srcIdx];
    }
    
    // Upload data to GPU
    size_t dataSize = shuffledData.size() * sizeof(uint32_t);
    vkcore::PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(dataSize, 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eShaderDeviceAddress, 
        MemoryType::Internal);
    loadUsingStagingBuf((char*)shuffledData.data(), dataSize, pointsBuffer, staging, vd, 0);
    
    // Build BruteForceIndexUpdatable
    std::cerr << "\nBuilding BruteForce Updatable Index...\n";
    PBruteForceIndexUpdatable bruteForceIndex = std::make_shared<BruteForceIndexUpdatable>(vd, ncols, BRUTE_SCALE_FACTOR);
    bruteForceIndex->initialize();
    
    CPUTimer buildTimer;
    buildTimer.start();
    bruteForceIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    double buildTime = double(buildTimer.stop()) / 1000000.0;
    std::cerr << "Build time: " << std::fixed << std::setprecision(3) << (buildTime * 1000.0) << " ms\n";
    
    // Setup query infrastructure
    const int numQueries = Q;
    std::vector<uint32_t> targets;
    generateTPCCQueriesMode70(minval, maxval, ncols, targets, numQueries);
    
    uint32_t capacity = bruteForceIndex->capacity;
    uint32_t resultSizeUints = (capacity + 31) / 32;
    
    vkcore::PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    vkcore::PBuffer resultBuffer(new Buffer(vd));
    resultBuffer->create(resultSizeUints * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    // Initial query performance
    std::cerr << "\n--- Initial Query Performance ---\n";
    std::vector<uint32_t> initialCounts;
    auto [initialAvgTime, initialMedianTime] = runQueriesMode70(bruteForceIndex, queryBuffer, resultBuffer, staging, vd,
                                                                 targets, resultSizeUints, numQueries, &initialCounts);
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << initialAvgTime << " ms\n";
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << initialMedianTime << " ms\n";
    std::cerr << "Active entries: " << bruteForceIndex->activeCount << "\n";
    
    // =========================================================
    // Batch Delete/Insert Cycles with Verification
    // =========================================================
    const uint32_t NUM_BATCHES = 2;
    const int RUNS = 2;
    uint32_t batchSize = 5000;
    
    std::cerr << "\n--- Batch Delete/Insert with Query Performance ---\n";
    std::cerr << "Total points: " << npoints << "\n";
    std::cerr << "Number of batches: " << NUM_BATCHES << "\n";
    std::cerr << "Batch size: " << batchSize << " points\n";
    std::cerr << "Runs: " << RUNS << "\n";
    
    // Allocate buffers for GPU-accelerated delete/insert
    // rowIdBuffer: stores row indices for delete
    vkcore::PBuffer rowIdBuffer(new Buffer(vd));
    rowIdBuffer->create(batchSize * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    // slotBuffer: stores slot indices for insert (same as rowIds for re-insert)
    vkcore::PBuffer slotBuffer(new Buffer(vd));
    slotBuffer->create(batchSize * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    // pointsBuffer: stores (x, y, z) per point for insert
    vkcore::PBuffer batchBuffer(new Buffer(vd));
    batchBuffer->create(batchSize * 3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    std::vector<uint32_t> rowIdData(batchSize);
    std::vector<uint32_t> batchData(batchSize * 3);
    
    // Print header
    std::cerr << "\n";
    std::cerr << std::setw(6) << "Run" 
              << std::setw(10) << "Size"
              << std::setw(12) << "Del(ms)"
              << std::setw(12) << "Ins(ms)"
              << std::setw(14) << "AvgQuery(ms)"
              << std::setw(10) << "Active" << "\n";
    std::cerr << std::string(64, '-') << "\n";
    
    double totalDelTime = 0, totalInsTime = 0, totalQueryTime = 0;
    
    for (int r = 0; r < RUNS; r++) {
        uint32_t b = r % NUM_BATCHES;
        uint32_t startIdx = b * batchSize;
        uint32_t currentBatchSize = std::min(batchSize, npoints - startIdx);
        
        // Prepare row IDs for delete and slot IDs for append
        std::vector<uint32_t> slotData(currentBatchSize);
        for (uint32_t i = 0; i < currentBatchSize; i++) {
            uint32_t pointIdx = indices[startIdx + i];
            rowIdData[i] = pointIdx;  // Row ID for delete
            slotData[i] = bruteForceIndex->nextFreeSlot + i;  // Append at end
            batchData[i * 3 + 0] = data[pointIdx];
            batchData[i * 3 + 1] = data[npoints + pointIdx];
            batchData[i * 3 + 2] = data[2 * npoints + pointIdx];
        }
        
        // Upload row IDs for delete
        loadUsingStagingBuf((char*)rowIdData.data(), currentBatchSize * sizeof(uint32_t), rowIdBuffer, staging, vd, 0);
        // Upload slot IDs for insert (append at end)
        loadUsingStagingBuf((char*)slotData.data(), currentBatchSize * sizeof(uint32_t), slotBuffer, staging, vd, 0);
        // Upload point data for insert
        loadUsingStagingBuf((char*)batchData.data(), currentBatchSize * 3 * sizeof(uint32_t), batchBuffer, staging, vd, 0);
        
        // --- DELETE (GPU-accelerated) ---
        CPUTimer delTimer;
        delTimer.start();
        bruteForceIndex->deleteByRowId(rowIdBuffer, currentBatchSize);
        double delTime = double(delTimer.stop()) / 1000000.0;
        totalDelTime += delTime;
        
        // --- INSERT (GPU-accelerated) ---
        CPUTimer insTimer;
        insTimer.start();
        bruteForceIndex->insertPoints(batchBuffer, slotBuffer, rowIdBuffer, currentBatchSize);
        double insTime = double(insTimer.stop()) / 1000000.0;
        totalInsTime += insTime;
        
        // --- RUN QUERIES ---
        auto [avgQueryTime, medQueryTime] = runQueriesMode70(bruteForceIndex, queryBuffer, resultBuffer, staging, vd,
                                                              targets, resultSizeUints, numQueries, nullptr);
        totalQueryTime += avgQueryTime;
        
        // Print row
        std::cerr << std::setw(6) << (r + 1)
                  << std::setw(10) << currentBatchSize
                  << std::setw(12) << std::fixed << std::setprecision(3) << (delTime * 1000.0)
                  << std::setw(12) << std::fixed << std::setprecision(3) << (insTime * 1000.0)
                  << std::setw(14) << std::fixed << std::setprecision(3) << avgQueryTime
                  << std::setw(10) << bruteForceIndex->activeCount
                  << " (nfs=" << bruteForceIndex->nextFreeSlot << ")\n";
    }
    
    // Final verification
    std::cerr << "\n--- Final Verification ---\n";
    std::vector<uint32_t> finalCounts;
    auto [finalAvgTime, finalMedianTime] = runQueriesMode70(bruteForceIndex, queryBuffer, resultBuffer, staging, vd,
                                                             targets, resultSizeUints, numQueries, &finalCounts);
    
    bool allMatch = true;
    double selectivityStep = 100.0 / numQueries;
    for (int q = 0; q < numQueries; q++) {
        bool match = (initialCounts[q] == finalCounts[q]);
        if (!match) {
            allMatch = false;
            std::cerr << "MISMATCH at query " << (q + 1) << " (" << std::fixed << std::setprecision(1) 
                      << ((q + 1) * selectivityStep) << "%): Initial=" << initialCounts[q] 
                      << ", Final=" << finalCounts[q] << "\n";
        }
    }
    
    if (allMatch) {
        std::cerr << "All " << numQueries << " queries match between initial and final state!\n";
    }
    
    // Summary
    std::cerr << std::string(64, '-') << "\n";
    std::cerr << "\n--- Summary ---\n";
    std::cerr << "Scale: " << scaleNames[scaleIdx] << " (" << npoints << " customers)\n";
    std::cerr << "Scale Factor: " << BRUTE_SCALE_FACTOR << "x (capacity: " << capacity << ")\n";
    std::cerr << "Runs completed: " << RUNS << "\n";
    std::cerr << "Avg delete time per run: " << std::fixed << std::setprecision(3) << (totalDelTime * 1000.0 / RUNS) << " ms\n";
    std::cerr << "Avg insert time per run: " << std::fixed << std::setprecision(3) << (totalInsTime * 1000.0 / RUNS) << " ms\n";
    std::cerr << "Initial avg query time: " << std::fixed << std::setprecision(3) << initialAvgTime << " ms\n";
    std::cerr << "Final avg query time: " << std::fixed << std::setprecision(3) << finalAvgTime << " ms\n";
    std::cerr << "Final active entries: " << bruteForceIndex->activeCount << "\n";
    
    // Cleanup
    rowIdBuffer->destroy();
    slotBuffer->destroy();
    batchBuffer->destroy();
    queryBuffer->destroy();
    resultBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 70 Complete.\n";
}
