#include "RunModes.hpp"
#include "../EquiDepthIndex.hpp"
#include "../CompactScanIndex.hpp"
#include "../BufferPool.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sys/stat.h>
#include <algorithm>

#ifndef BUILD_COUNT
#define BUILD_COUNT 1
#endif

#ifndef QUERY_COUNT
#define QUERY_COUNT 1
#endif

// Distribution names for dataId 0-4
static const std::vector<std::string> distributionFilesMode51 = {
    "uniform.bin",
    "normal.bin",
    "zipf1.1.bin",
    "zipf1.3.bin",
    "zipf1.5.bin"
};

static const std::vector<std::string> distributionNamesMode51 = {
    "uniform",
    "normal",
    "zipf1.1",
    "zipf1.3",
    "zipf1.5"
};

// Query strategy based on distribution type (same as Mode22)
enum class QueryStrategyMode51 { CENTERED, FROM_MIN };

static QueryStrategyMode51 getQueryStrategyMode51(int dataId) {
    switch (dataId) {
        case 0: return QueryStrategyMode51::CENTERED;  // Uniform
        case 1: return QueryStrategyMode51::CENTERED;  // Normal
        case 2: return QueryStrategyMode51::FROM_MIN;  // Zipf 1.1
        case 3: return QueryStrategyMode51::FROM_MIN;  // Zipf 1.3
        case 4: return QueryStrategyMode51::FROM_MIN;  // Zipf 1.5
        default: return QueryStrategyMode51::CENTERED;
    }
}

// Generate queries with selectivities 10%, 20%, ..., 100% (same as Mode22)
static std::vector<std::array<uint32_t, 6>> generateQueriesMode51(
    const std::vector<uint32_t>& minval,
    const std::vector<uint32_t>& maxval,
    int ncols,
    int dataId,
    int numQueries = 10
) {
    std::vector<std::array<uint32_t, 6>> queries(numQueries);
    QueryStrategyMode51 strategy = getQueryStrategyMode51(dataId);
    
    std::cerr << "Query strategy: " << (strategy == QueryStrategyMode51::CENTERED ? "CENTERED" : "FROM_MIN") << "\n";
    
    for (int q = 0; q < numQueries; q++) {
        double overallSelectivity = (q + 1) * 0.1;  // 10%, 20%, ..., 100%
        double perDimSelectivity = std::pow(overallSelectivity, 1.0 / ncols);
        bool isFullRange = (q == numQueries - 1);
        
        for (int c = 0; c < 3; c++) {
            uint32_t lo, hi;
            
            if (c >= ncols) {
                lo = 0;
                hi = UINT32_MAX;
            } else if (isFullRange) {
                lo = minval[c];
                hi = maxval[c];
            } else {
                uint64_t range = (uint64_t)maxval[c] - (uint64_t)minval[c];
                uint64_t queryRange = (uint64_t)(range * perDimSelectivity);
                
                if (strategy == QueryStrategyMode51::CENTERED) {
                    uint64_t margin = (range - queryRange) / 2;
                    lo = minval[c] + (uint32_t)margin;
                    hi = minval[c] + (uint32_t)(margin + queryRange);
                } else {
                    lo = minval[c];
                    hi = minval[c] + (uint32_t)queryRange;
                }
            }
            
            queries[q][c * 2] = lo;      // min for dimension c
            queries[q][c * 2 + 1] = hi;  // max for dimension c
        }
    }
    
    return queries;
}

void testEquiDepthIndex(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    
    // Validate dataId (0-4 for the 5 distributions)
    if (dataId < 0 || dataId >= (int)distributionFilesMode51.size()) {
        std::cerr << "ERROR: Invalid dataId " << dataId << ". Must be 0-4.\n";
        std::cerr << "  0: uniform, 1: normal, 2: zipf1.1, 3: zipf1.3, 4: zipf1.5\n";
        return;
    }
    
    // Construct plain data folder path
    uint32_t millions = g_npoints / 1000000;
    std::string plainDataFolder = PROJECT_DIR + "data/data_" + std::to_string(millions) + "m_" + std::to_string(g_dim) + "c";
    std::string dataFile = plainDataFolder + "/" + distributionFilesMode51[dataId];
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 51: Equi-Depth Index Test\n";
    std::cerr << "Distribution: " << distributionNamesMode51[dataId] << " (dataId=" << dataId << ")\n";
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

    // Result Buffer
    uint32_t resultSizeUints = (npoints + 31) / 32;

    // =========================================================
    // Build and test EquiDepthIndex
    // =========================================================
    std::cerr << "\n--- [EquiDepthIndex] ---\n";

    GPUMemoryTool::printGPUMemoryStatus(vd, "Before EquiDepthIndex build");

    std::cerr << "\nBuilding Equi-Depth Index (taking median of " << BUILD_COUNT << " runs)...\n";
    std::vector<double> equiDepthBuildTimes;
    equiDepthBuildTimes.reserve(BUILD_COUNT);
    
    for (int k = 0; k < BUILD_COUNT; k++) {
        if (k > 0) std::cerr << "  Run " << k + 1 << "...\n";

        CPUTimer buildTimer;
        buildTimer.start();

        PEquiDepthIndex equiDepthRun = std::make_shared<EquiDepthIndex>(vd, ncols, scan);
        equiDepthRun->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());

        double bt = double(buildTimer.stop()) / 1000000.0;

        std::cout << k << ": ============>  Build time: " << bt * 1000.0 << " ms\n";

        equiDepthBuildTimes.push_back(bt);

        equiDepthRun.reset();
        vd->device->waitIdle();
    }
    std::sort(equiDepthBuildTimes.begin(), equiDepthBuildTimes.end());
    double buildTime = equiDepthBuildTimes[equiDepthBuildTimes.size() / 2];
    std::cerr << "\n\n>>> Equi-Depth Index build time: " << (buildTime * 1000.0) << " ms\n\n";

    // Build final index for queries
    PEquiDepthIndex equiDepthIndex = std::make_shared<EquiDepthIndex>(vd, ncols, scan);
    equiDepthIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    GPUMemoryTool::printGPUMemoryStatus(vd, "After EquiDepthIndex build");

    std::cout << "[EquiDepthIndex] Total DataBuffer Size: " << equiDepthIndex->getSizeMB() << " MB\n";

    // Print bin statistics
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    std::vector<uint32_t> extents(totalBins);
    readUsingStagingBuf((char *)extents.data(), totalBins * sizeof(uint32_t), equiDepthIndex->extentBuffer, staging, vd);
    
    uint32_t cmin = UINT32_MAX, cmax = 0;
    uint64_t csum = 0;
    uint32_t nonEmptyBins = 0;
    for (uint32_t c : extents) {
        if (c < cmin) cmin = c;
        if (c > cmax) cmax = c;
        csum += c;
        if (c > 0) nonEmptyBins++;
    }
    double avg = (double)csum / totalBins;
    double avgNonEmpty = nonEmptyBins > 0 ? (double)csum / nonEmptyBins : 0;
    
    std::cerr << "\n[Statistics] Equi-Depth Bin Counts:\n";
    std::cerr << "  Min=" << cmin << ", Max=" << cmax << ", Avg=" << std::fixed << std::setprecision(2) << avg << "\n";
    std::cerr << "  Non-empty bins: " << nonEmptyBins << " / " << totalBins << "\n";
    std::cerr << "  Avg (non-empty): " << std::fixed << std::setprecision(2) << avgNonEmpty << "\n";
    std::cerr << "  Total Points: " << csum << " / " << npoints << "\n";
    
    // Calculate variance to show equi-depth improvement
    double variance = 0;
    for (uint32_t c : extents) {
        variance += (c - avg) * (c - avg);
    }
    variance /= totalBins;
    double stddev = sqrt(variance);
    std::cerr << "  StdDev: " << std::fixed << std::setprecision(2) << stddev << "\n";
    std::cerr << "  Coefficient of Variation: " << std::fixed << std::setprecision(4) << (avg > 0 ? stddev / avg : 0) << "\n";

    // =========================================================
    // Compare with CompactScanIndex (non-equi-depth)
    // =========================================================
    std::cerr << "\n--- [CompactScanIndex (baseline)] ---\n";

    // find time
    CPUTimer compactTimer;
    compactTimer.start();
    PCompactScanIndex compactIndex = std::make_shared<CompactScanIndex>(vd, ncols, scan);
    compactIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    double compactBuildTime = double(compactTimer.stop()) / 1000000.0;
    std::cerr << "CompactScanIndex build time: " << (compactBuildTime * 1000.0) << " ms\n";
    
    std::vector<uint32_t> compactExtents(totalBins);
    readUsingStagingBuf((char *)compactExtents.data(), totalBins * sizeof(uint32_t), compactIndex->extentBuffer, staging, vd);
    
    uint32_t compactMin = UINT32_MAX, compactMax = 0;
    uint64_t compactSum = 0;
    uint32_t compactNonEmpty = 0;
    for (uint32_t c : compactExtents) {
        if (c < compactMin) compactMin = c;
        if (c > compactMax) compactMax = c;
        compactSum += c;
        if (c > 0) compactNonEmpty++;
    }
    double compactAvg = (double)compactSum / totalBins;
    
    double compactVariance = 0;
    for (uint32_t c : compactExtents) {
        compactVariance += (c - compactAvg) * (c - compactAvg);
    }
    compactVariance /= totalBins;
    double compactStddev = sqrt(compactVariance);
    
    std::cerr << "\n[Statistics] Compact (non-equi-depth) Bin Counts:\n";
    std::cerr << "  Min=" << compactMin << ", Max=" << compactMax << ", Avg=" << std::fixed << std::setprecision(2) << compactAvg << "\n";
    std::cerr << "  Non-empty bins: " << compactNonEmpty << " / " << totalBins << "\n";
    std::cerr << "  StdDev: " << std::fixed << std::setprecision(2) << compactStddev << "\n";
    std::cerr << "  Coefficient of Variation: " << std::fixed << std::setprecision(4) << (compactAvg > 0 ? compactStddev / compactAvg : 0) << "\n";

    // =========================================================
    // Run 10 queries with varying selectivity (10%, 20%, ..., 100%)
    // =========================================================
    std::cerr << "\n--- [Query Execution] ---\n";
    
    const int NUM_QUERIES = 10;
    std::vector<std::array<uint32_t, 6>> queries = generateQueriesMode51(minval, maxval, ncols, dataId, NUM_QUERIES);
    
    // Create query buffer
    vkcore::PBuffer queryBuffer(new vkcore::Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);
    
    // Create result buffers
    vkcore::PBuffer equiDepthResultBuffer(new vkcore::Buffer(vd));
    equiDepthResultBuffer->create(resultSizeUints * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vkcore::MemoryType::Internal);
    
    vkcore::PBuffer compactResultBuffer(new vkcore::Buffer(vd));
    compactResultBuffer->create(resultSizeUints * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vkcore::MemoryType::Internal);
    
    // ========== Run queries on CompactScanIndex ==========
    std::cerr << "\n--- CompactScanIndex Query Performance ---\n";
    double compactTotTime = 0;
    std::vector<double> compactQueryTimes;
    compactQueryTimes.reserve(NUM_QUERIES);
    std::vector<uint32_t> compactCounts(NUM_QUERIES);
    
    for (int q = 0; q < NUM_QUERIES; q++) {
        // Format: x1, x2, y1, y2, z1, z2
        std::vector<uint32_t> queryData = {queries[q][0], queries[q][1], queries[q][2], queries[q][3], queries[q][4], queries[q][5]};
        
        std::vector<double> qt;
        qt.reserve(QUERY_COUNT);
        for (int r = 0; r < QUERY_COUNT; r++) {
            loadUsingStagingBuf((char *)queryData.data(), queryData.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            CPUTimer qTimer;
            qTimer.start();
            compactIndex->runRangeQueries(queryBuffer, 1, compactResultBuffer);
            double t = double(qTimer.stop()) / 1000000.0;
            qt.push_back(t);
        }
        std::sort(qt.begin(), qt.end());
        double t = qt[qt.size() / 2];
        compactTotTime += t;
        compactQueryTimes.push_back(t * 1000.0);
        
        // Read back and count
        std::vector<uint32_t> resultData(resultSizeUints);
        readUsingStagingBuf((char *)resultData.data(), resultSizeUints * sizeof(uint32_t), compactResultBuffer, staging, vd);
        uint32_t count = 0;
        for (uint32_t val : resultData) count += __builtin_popcount(val);
        compactCounts[q] = count;
        
        double selectivity = (q + 1) * 10.0;
        std::cerr << "Query " << (q + 1) << " (" << std::fixed << std::setprecision(0) << selectivity << "%): " 
                  << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((compactTotTime * 1000.0) / NUM_QUERIES) << " ms\n";
    std::sort(compactQueryTimes.begin(), compactQueryTimes.end());
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << compactQueryTimes[compactQueryTimes.size() / 2] << " ms\n";
    
    // ========== Run queries on EquiDepthIndex ==========
    std::cerr << "\n--- EquiDepthIndex Query Performance ---\n";
    double equiDepthTotTime = 0;
    std::vector<double> equiDepthQueryTimes;
    equiDepthQueryTimes.reserve(NUM_QUERIES);
    std::vector<uint32_t> equiDepthCounts(NUM_QUERIES);
    
    for (int q = 0; q < NUM_QUERIES; q++) {
        std::vector<uint32_t> queryData = {queries[q][0], queries[q][1], queries[q][2], queries[q][3], queries[q][4], queries[q][5]};
        
        std::vector<double> qt;
        qt.reserve(QUERY_COUNT);
        for (int r = 0; r < QUERY_COUNT; r++) {
            loadUsingStagingBuf((char *)queryData.data(), queryData.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            CPUTimer qTimer;
            qTimer.start();
            equiDepthIndex->runRangeQueries(queryBuffer, 1, equiDepthResultBuffer);
            double t = double(qTimer.stop()) / 1000000.0;
            qt.push_back(t);
        }
        std::sort(qt.begin(), qt.end());
        double t = qt[qt.size() / 2];
        equiDepthTotTime += t;
        equiDepthQueryTimes.push_back(t * 1000.0);
        
        std::vector<uint32_t> resultData(resultSizeUints);
        readUsingStagingBuf((char *)resultData.data(), resultSizeUints * sizeof(uint32_t), equiDepthResultBuffer, staging, vd);
        uint32_t count = 0;
        for (uint32_t val : resultData) count += __builtin_popcount(val);
        equiDepthCounts[q] = count;
        
        double selectivity = (q + 1) * 10.0;
        std::cerr << "Query " << (q + 1) << " (" << std::fixed << std::setprecision(0) << selectivity << "%): " 
                  << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((equiDepthTotTime * 1000.0) / NUM_QUERIES) << " ms\n";
    std::sort(equiDepthQueryTimes.begin(), equiDepthQueryTimes.end());
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << equiDepthQueryTimes[equiDepthQueryTimes.size() / 2] << " ms\n";
    
    // ========== Verify counts match between CompactScan and EquiDepth ==========
    std::cerr << "\n--- Query Count Verification (CompactScan vs EquiDepth) ---\n";
    std::cerr << "Query | Selectivity | Compact    | EquiDepth  | Match\n";
    std::cerr << "------+-------------+------------+------------+------\n";
    bool allMatch = true;
    for (int q = 0; q < NUM_QUERIES; q++) {
        double selectivity = (q + 1) * 10.0;
        bool match = (compactCounts[q] == equiDepthCounts[q]);
        if (!match) allMatch = false;
        std::cerr << std::setw(5) << (q + 1) << " | " 
                  << std::setw(10) << std::fixed << std::setprecision(0) << selectivity << "% | "
                  << std::setw(10) << compactCounts[q] << " | "
                  << std::setw(10) << equiDepthCounts[q] << " | "
                  << (match ? "YES" : "NO") << "\n";
    }
    std::cerr << "\nAll counts match: " << (allMatch ? "YES" : "NO") << "\n";
    
    // Cleanup query buffers
    queryBuffer->destroy();
    equiDepthResultBuffer->destroy();
    compactResultBuffer->destroy();
    
    // =========================================================
    // Summary comparison
    // =========================================================
    std::cerr << "\n========================================\n";
    std::cerr << "BUILD TIME COMPARISON\n";
    std::cerr << "========================================\n";
    std::cerr << "Equi-Depth Index build time: " << std::fixed << std::setprecision(2) << (buildTime * 1000.0) << " ms\n";
    std::cerr << "CompactScan Index build time: " << std::fixed << std::setprecision(2) << (compactBuildTime * 1000.0) << " ms\n";
    
    std::cerr << "\n========================================\n";
    std::cerr << "BIN DISTRIBUTION COMPARISON\n";
    std::cerr << "========================================\n";
    std::cerr << "                    Equi-Depth    Compact\n";
    std::cerr << "Max bin count:      " << std::setw(10) << cmax << "  " << std::setw(10) << compactMax << "\n";
    std::cerr << "StdDev:             " << std::setw(10) << std::fixed << std::setprecision(2) << stddev 
              << "  " << std::setw(10) << compactStddev << "\n";
    std::cerr << "CoV:                " << std::setw(10) << std::fixed << std::setprecision(4) << (avg > 0 ? stddev / avg : 0)
              << "  " << std::setw(10) << (compactAvg > 0 ? compactStddev / compactAvg : 0) << "\n";
    
    double improvement = compactMax > 0 ? (1.0 - (double)cmax / compactMax) * 100.0 : 0;
    std::cerr << "\nMax bin reduction: " << std::fixed << std::setprecision(1) << improvement << "%\n";
    
    double covImprovement = (compactAvg > 0 && avg > 0) ? 
        (1.0 - (stddev / avg) / (compactStddev / compactAvg)) * 100.0 : 0;
    std::cerr << "CoV reduction: " << std::fixed << std::setprecision(1) << covImprovement << "%\n";

    // Cleanup
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 51 Complete.\n";
}
