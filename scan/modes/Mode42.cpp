#include "RunModes.hpp"
#include "../CompactScanIndex.hpp"
#include "../RasterScan2D.hpp"
#include "../BufferPool.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sys/stat.h>
#include <algorithm>
#include <random>

// Mode 42: Mode 22 + Query Performance After Each Update Cycle
// Based on Mode 22, but runs queries after each batch delete/insert

#ifndef BUILD_COUNT
#define BUILD_COUNT 1
#endif

#ifndef QUERY_COUNT
#define QUERY_COUNT 1
#endif

// Distribution names for dataId 0-4
static const std::vector<std::string> distributionFiles42 = {
    "uniform.bin",
    "normal.bin",
    "zipf1.1.bin",
    "zipf1.3.bin",
    "zipf1.5.bin",
    "tpcc.bin"
};

static const std::vector<std::string> distributionNames42 = {
    "uniform",
    "normal",
    "zipf1.1",
    "zipf1.3",
    "zipf1.5",
    "tpcc"
};

// Query generation strategy based on distribution type
enum class QueryStrategy42 {
    CENTERED,
    FROM_MIN
};

static QueryStrategy42 getQueryStrategy42(int dataId) {
    switch (dataId) {
        case 0: return QueryStrategy42::CENTERED;
        case 1: return QueryStrategy42::CENTERED;
        case 2: return QueryStrategy42::FROM_MIN;
        case 3: return QueryStrategy42::FROM_MIN;
        case 4: return QueryStrategy42::FROM_MIN;
        case 5: return QueryStrategy42::CENTERED;  // TPC-C
        default: return QueryStrategy42::CENTERED;
    }
}

// Generate queries with specific selectivities
static void generateQueries42(
    const std::vector<uint32_t>& minval, 
    const std::vector<uint32_t>& maxval,
    int ncols,
    int dataId,
    std::vector<uint32_t>& targets,
    int numQueries = 10
) {
    targets.clear();
    targets.reserve(numQueries * 6);
    
    QueryStrategy42 strategy = getQueryStrategy42(dataId);
    
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
                
                if (strategy == QueryStrategy42::CENTERED) {
                    uint64_t margin = (range - queryRange) / 2;
                    lo = minval[c] + (uint32_t)margin;
                    hi = minval[c] + (uint32_t)(margin + queryRange);
                } else {
                    lo = minval[c];
                    hi = minval[c] + (uint32_t)queryRange;
                }
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
static std::pair<double, double> runQueriesAndGetStats42(
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
        
        std::cerr << "MAX QUERY TIME: " << std::fixed << std::setprecision(3) << t * 1000.0 << " ms\n";

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

void testCompactIndexWithQueryAfterUpdate(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    using namespace vkcore;
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 42: CompactIndex with Query After Each Update\n";
    std::cerr << "Distribution: " << distributionNames42[dataId] << " (dataId=" << dataId << ")\n";
    std::cerr << "========================================\n";
    
    vkcore::SinglePassScan *scan = (vkcore::SinglePassScan *) op.getFunction(vkcore::FunctionType::SinglePassScan);
    
    // Construct plain data folder path
    uint32_t millions = g_npoints / 1000000;
    std::string plainDataFolder = PROJECT_DIR + "data/data_" + std::to_string(millions) + "m_" + std::to_string(g_dim) + "c";
    std::string dataFile = plainDataFolder + "/" + distributionFiles42[dataId];
    
    std::cerr << "Data file: " << dataFile << "\n";
    std::cerr << "Columns: " << g_dim << "\n";

    // Read Plain Dataset
    int32_t ncols = g_dim;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<uint32_t> points;
    vkcore::PBuffer pointsBuffer = readPlainData(dataFile, vd, staging, npoints, minval, maxval, points, ncols);
    
    if (!pointsBuffer) {
        std::cerr << "ERROR: Failed to load data from " << dataFile << "\n";
        return;
    }
    
    std::cerr << "Dataset: " << npoints << " points\n";
    std::cerr << "MIN: " << minval[0] << " " << minval[1] << " " << minval[2] << "\n";
    std::cerr << "MAX: " << maxval[0] << " " << maxval[1] << " " << maxval[2] << "\n";
    
    // Generate queries
    const int numQueries = 10;
    std::vector<uint32_t> targets;
    generateQueries42(minval, maxval, ncols, dataId, targets, numQueries);
    
    uint32_t resultSizeUints = (npoints + 31) / 32;
    
    // Query buffer
    vkcore::PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    // Build CompactScanIndex
    std::cerr << "\nBuilding Compact Index...\n";
    PCompactScanIndex compactIndex = std::make_shared<CompactScanIndex>(vd, ncols, scan);
    compactIndex->useIndexedDelete = g_useSkewedPipeline;
    // Note: initialize() is called internally by buildIndex()
    
    CPUTimer buildTimer;
    buildTimer.start();
    compactIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    double buildTime = double(buildTimer.stop()) / 1000000.0;
    std::cerr << "Build time: " << std::fixed << std::setprecision(3) << (buildTime * 1000.0) << " ms\n";
    
    // Result buffer
    vkcore::PBuffer resultBuffer(new Buffer(vd));
    resultBuffer->create(resultSizeUints * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    // Initial query performance
    std::cerr << "\n--- Initial Query Performance ---\n";
    auto [initialAvgTime, initialMedianTime] = runQueriesAndGetStats42(compactIndex, queryBuffer, resultBuffer, staging, vd, 
                                                       targets, resultSizeUints, numQueries, true);
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << initialAvgTime << " ms\n";
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << initialMedianTime << " ms\n";
    
    // =========================================================
    // Batch Delete/Insert Cycles with Query After Each
    // =========================================================
    const int K = 10;  // Number of batches (increased to avoid GPU timeout with large batches)
    const uint32_t batchSize = std::max<uint32_t>(1u, npoints / (uint32_t)K);
    
    std::cerr << "\n--- Batch Delete/Insert with Query Performance ---\n";
    std::cerr << "Total points: " << npoints << "\n";
    std::cerr << "Number of batches (K): " << K << "\n";
    std::cerr << "Batch size: " << batchSize << " points\n";
    
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
    allQueryTimes.reserve(K);
    
    for (int k = 0; k < K; k++) {
        uint32_t startIdx = (uint32_t)k * batchSize;
        uint32_t endIdx = (k == K - 1) ? npoints : std::min(npoints, (uint32_t)(k + 1) * batchSize);
        uint32_t currentBatchSize = endIdx - startIdx;
        if (currentBatchSize == 0) continue;
        
        // Prepare batch data (row-major format)
        for (uint32_t i = 0; i < currentBatchSize; i++) {
            uint32_t pointIdx = indices[startIdx + i];
            batchData[i * 3 + 0] = points[pointIdx];
            batchData[i * 3 + 1] = points[npoints + pointIdx];
            batchData[i * 3 + 2] = points[2 * npoints + pointIdx];
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
        auto [avgQueryTime, medQueryTime] = runQueriesAndGetStats42(compactIndex, queryBuffer, resultBuffer, staging, vd,
                                                         targets, resultSizeUints, numQueries, false);
        totalQueryTime += avgQueryTime;
        allQueryTimes.push_back(avgQueryTime);
        
        // Print row
        std::cerr << std::setw(6) << (k + 1)
                  << std::setw(10) << currentBatchSize
                  << std::setw(12) << std::fixed << std::setprecision(3) << (delTime * 1000.0)
                  << std::setw(12) << std::fixed << std::setprecision(3) << (insTime * 1000.0)
                  << std::setw(14) << std::fixed << std::setprecision(3) << avgQueryTime << "\n";
    }
    
    // Summary
    std::cerr << std::string(54, '-') << "\n";
    std::cerr << "\n--- Summary ---\n";
    std::cerr << "Distribution: " << distributionNames42[dataId] << "\n";
    std::cerr << "Dataset size: " << npoints << " points\n";
    std::cerr << "Batches (K): " << K << "\n";
    std::cerr << "Avg delete time per batch: " << std::fixed << std::setprecision(3) << (totalDelTime * 1000.0 / K) << " ms\n";
    std::cerr << "Avg insert time per batch: " << std::fixed << std::setprecision(3) << (totalInsTime * 1000.0 / K) << " ms\n";
    std::cerr << "Avg query time (10 queries): " << std::fixed << std::setprecision(3) << (totalQueryTime / K) << " ms\n";
    std::cerr << "Initial avg query time: " << std::fixed << std::setprecision(3) << initialAvgTime << " ms\n";
    std::cerr << "Initial median query time: " << std::fixed << std::setprecision(3) << initialMedianTime << " ms\n";
    
    // Compute final median query time
    std::sort(allQueryTimes.begin(), allQueryTimes.end());
    double finalMedianQueryTime = allQueryTimes[allQueryTimes.size() / 2];
    std::cerr << "Final avg query time: " << std::fixed << std::setprecision(3) << (totalQueryTime / K) << " ms\n";
    std::cerr << "Final median query time: " << std::fixed << std::setprecision(3) << finalMedianQueryTime << " ms\n";
    
    // Cleanup
    batchBuffer->destroy();
    if (indexBuffer) indexBuffer->destroy();
    resultBuffer->destroy();
    queryBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 42 Complete.\n";
}
