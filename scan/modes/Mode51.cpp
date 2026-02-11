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
#define BUILD_COUNT 11
#endif

#ifndef QUERY_COUNT
#define QUERY_COUNT 11
#endif

// Distribution names for dataId 0-4
static const std::vector<std::string> distributionFilesMode51 = {
    "uniform.bin",
    "normal.bin",
    "zipf1.1.bin",
    "zipf1.3.bin",
    "zipf1.5.bin",
    "tpcc.bin"
};

static const std::vector<std::string> distributionNamesMode51 = {
    "uniform",
    "normal",
    "zipf1.1",
    "zipf1.3",
    "zipf1.5",
    "tpcc"
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
        case 5: return QueryStrategyMode51::CENTERED;  // TPC-C
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
    
    // Run mode: 0=both, 1=EquiDepth only, 2=CompactScan only
    bool runEquiDepth = (g_runmode == 0 || g_runmode == 1);
    bool runCompactScan = (g_runmode == 0 || g_runmode == 2);
    
    // Construct plain data folder path
    uint32_t millions = g_npoints / 1000000;
    std::string plainDataFolder = PROJECT_DIR + "data/data_" + std::to_string(millions) + "m_" + std::to_string(g_dim) + "c";
    std::string dataFile = plainDataFolder + "/" + distributionFilesMode51[dataId];
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 51: Equi-Depth vs CompactScan Comparison\n";
    std::cerr << "Distribution: " << distributionNamesMode51[dataId] << " (dataId=" << dataId << ")\n";
    std::cerr << "Data file: " << dataFile << "\n";
    std::cerr << "Columns: " << g_dim << "\n";
    std::cerr << "Run mode: " << g_runmode << " (0=both, 1=EquiDepth, 2=CompactScan)\n";
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
    const int NUM_QUERIES = 10;
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    
    // Generate queries
    std::vector<std::array<uint32_t, 6>> queries = generateQueriesMode51(minval, maxval, ncols, dataId, NUM_QUERIES);
    
    // Create query buffer
    vkcore::PBuffer queryBuffer(new vkcore::Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);

    // Variables for comparison
    double equiDepthBuildTime = 0;
    double compactBuildTime = 0;
    double equiDepthTotTime = 0;
    double compactTotTime = 0;
    std::vector<double> equiDepthQueryTimes;
    std::vector<double> compactQueryTimes;
    std::vector<uint32_t> equiDepthCounts(NUM_QUERIES, 0);
    std::vector<uint32_t> compactCounts(NUM_QUERIES, 0);
    
    // Bin statistics
    uint32_t cmax = 0, compactMax = 0;
    double stddev = 0, compactStddev = 0;
    double avg = 0, compactAvg = 0;

    // =========================================================
    // PART 1: EquiDepth (Build + Query)
    // =========================================================
    if (runEquiDepth) {
        std::cerr << "\n###############################################\n";
        std::cerr << "# PART 1: EquiDepthIndex (Build + Query)\n";
        std::cerr << "###############################################\n";

        GPUMemoryTool::printGPUMemoryStatus(vd, "Before EquiDepthIndex");

        // Build timing
        std::cerr << "\nBuilding Equi-Depth Index (warmup + mean of " << BUILD_COUNT << " runs)...\n";
        std::vector<double> buildTimes;
        buildTimes.reserve(BUILD_COUNT);
        
        // Warmup run
        {
            std::cerr << "  Warmup run (shader compilation)...\n";
            PEquiDepthIndex warmup = std::make_shared<EquiDepthIndex>(vd, ncols, scan);
            warmup->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
            vd->device->waitIdle();
            warmup.reset();
            vd->device->waitIdle();
        }
        
        for (int k = 0; k < BUILD_COUNT; k++) {
            std::cerr << "  Run " << k + 1 << "/" << BUILD_COUNT << "...\n";

            PEquiDepthIndex idx = std::make_shared<EquiDepthIndex>(vd, ncols, scan);
            
            CPUTimer buildTimer;
            buildTimer.start();
            idx->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
            vd->device->waitIdle();
            double bt = double(buildTimer.stop()) / 1000000.0;

            std::cout << k << ": ============> Equi-Depth Build time: " << bt * 1000.0 << " ms\n";
            buildTimes.push_back(bt);

            idx.reset();
            vd->device->waitIdle();
        }
        
        for (double t : buildTimes) equiDepthBuildTime += t;
        equiDepthBuildTime /= buildTimes.size();
        std::cerr << "\n>>> Equi-Depth Index build time: " << (equiDepthBuildTime * 1000.0) << " ms\n";

        // Build final index for queries
        PEquiDepthIndex equiDepthIndex = std::make_shared<EquiDepthIndex>(vd, ncols, scan);
        equiDepthIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
        GPUMemoryTool::printGPUMemoryStatus(vd, "After EquiDepthIndex build");
        std::cout << "[EquiDepthIndex] Total DataBuffer Size: " << equiDepthIndex->getSizeMB() << " MB\n";

        // Print bin statistics
        std::vector<uint32_t> extents(totalBins);
        readUsingStagingBuf((char *)extents.data(), totalBins * sizeof(uint32_t), equiDepthIndex->extentBuffer, staging, vd);
        
        uint32_t cmin = UINT32_MAX;
        uint64_t csum = 0;
        uint32_t nonEmptyBins = 0;
        for (uint32_t c : extents) {
            if (c < cmin) cmin = c;
            if (c > cmax) cmax = c;
            csum += c;
            if (c > 0) nonEmptyBins++;
        }
        avg = (double)csum / totalBins;
        
        double variance = 0;
        for (uint32_t c : extents) variance += (c - avg) * (c - avg);
        variance /= totalBins;
        stddev = sqrt(variance);
        
        std::cerr << "\n[Statistics] Equi-Depth Bin Counts:\n";
        std::cerr << "  Min=" << cmin << ", Max=" << cmax << ", Avg=" << std::fixed << std::setprecision(2) << avg << "\n";
        std::cerr << "  Non-empty bins: " << nonEmptyBins << " / " << totalBins << "\n";
        std::cerr << "  StdDev: " << std::fixed << std::setprecision(2) << stddev << "\n";
        std::cerr << "  CoV: " << std::fixed << std::setprecision(4) << (avg > 0 ? stddev / avg : 0) << "\n";

        // Create result buffer
        vkcore::PBuffer resultBuffer(new vkcore::Buffer(vd));
        resultBuffer->create(resultSizeUints * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            vkcore::MemoryType::Internal);

        // Run queries
        std::cerr << "\n--- EquiDepthIndex Query Performance (warmup + mean of " << QUERY_COUNT << " runs) ---\n";
        equiDepthQueryTimes.reserve(NUM_QUERIES);
        
        for (int q = 0; q < NUM_QUERIES; q++) {
            std::vector<uint32_t> queryData = {queries[q][0], queries[q][1], queries[q][2], queries[q][3], queries[q][4], queries[q][5]};
            
            // Warmup
            loadUsingStagingBuf((char *)queryData.data(), queryData.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            equiDepthIndex->runRangeQueries(queryBuffer, 1, resultBuffer);
            vd->device->waitIdle();
            
            std::vector<double> qt;
            qt.reserve(QUERY_COUNT);
            for (int r = 0; r < QUERY_COUNT; r++) {
                loadUsingStagingBuf((char *)queryData.data(), queryData.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
                CPUTimer qTimer;
                qTimer.start();
                equiDepthIndex->runRangeQueries(queryBuffer, 1, resultBuffer);
                vd->device->waitIdle();
                qt.push_back(double(qTimer.stop()) / 1000000.0);
            }
            
            double t = 0;
            for (double v : qt) t += v;
            t /= qt.size();
            equiDepthTotTime += t;
            equiDepthQueryTimes.push_back(t * 1000.0);
            
            std::vector<uint32_t> resultData(resultSizeUints);
            readUsingStagingBuf((char *)resultData.data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
            uint32_t count = 0;
            for (uint32_t val : resultData) count += __builtin_popcount(val);
            equiDepthCounts[q] = count;
            
            std::cerr << "Query " << (q + 1) << " (" << std::fixed << std::setprecision(0) << ((q + 1) * 10.0) << "%): " 
                      << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Count: " << count << "\n";
        }
        std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((equiDepthTotTime * 1000.0) / NUM_QUERIES) << " ms\n";

        // Cleanup EquiDepth
        resultBuffer->destroy();
        equiDepthIndex.reset();
        vd->device->waitIdle();
        GPUMemoryTool::printGPUMemoryStatus(vd, "After EquiDepthIndex cleanup");
    }

    // =========================================================
    // PART 2: CompactScan (Build + Query)
    // =========================================================
    if (runCompactScan) {
        std::cerr << "\n###############################################\n";
        std::cerr << "# PART 2: CompactScanIndex (Build + Query)\n";
        std::cerr << "###############################################\n";

        GPUMemoryTool::printGPUMemoryStatus(vd, "Before CompactScanIndex");

        // Build timing
        std::cerr << "\nBuilding CompactScan Index (warmup + mean of " << BUILD_COUNT << " runs)...\n";
        std::vector<double> buildTimes;
        buildTimes.reserve(BUILD_COUNT);
        
        // Warmup run
        {
            std::cerr << "  Warmup run (shader compilation)...\n";
            PCompactScanIndex warmup = std::make_shared<CompactScanIndex>(vd, ncols, scan);
            warmup->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
            vd->device->waitIdle();
            warmup.reset();
            vd->device->waitIdle();
        }
        
        for (int k = 0; k < BUILD_COUNT; k++) {
            std::cerr << "  Run " << k + 1 << "/" << BUILD_COUNT << "...\n";

            PCompactScanIndex idx = std::make_shared<CompactScanIndex>(vd, ncols, scan);
            
            CPUTimer buildTimer;
            buildTimer.start();
            idx->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
            vd->device->waitIdle();
            double bt = double(buildTimer.stop()) / 1000000.0;

            std::cout << k << ": ============> CompactScan Build time: " << bt * 1000.0 << " ms\n";
            buildTimes.push_back(bt);

            idx.reset();
            vd->device->waitIdle();
        }
        
        for (double t : buildTimes) compactBuildTime += t;
        compactBuildTime /= buildTimes.size();
        std::cerr << "\n>>> CompactScan Index build time: " << (compactBuildTime * 1000.0) << " ms\n";

        // Build final index for queries
        PCompactScanIndex compactIndex = std::make_shared<CompactScanIndex>(vd, ncols, scan);
        compactIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
        GPUMemoryTool::printGPUMemoryStatus(vd, "After CompactScanIndex build");

        // Print bin statistics
        std::vector<uint32_t> extents(totalBins);
        readUsingStagingBuf((char *)extents.data(), totalBins * sizeof(uint32_t), compactIndex->extentBuffer, staging, vd);
        
        uint32_t compactMin = UINT32_MAX;
        uint64_t compactSum = 0;
        uint32_t compactNonEmpty = 0;
        for (uint32_t c : extents) {
            if (c < compactMin) compactMin = c;
            if (c > compactMax) compactMax = c;
            compactSum += c;
            if (c > 0) compactNonEmpty++;
        }
        compactAvg = (double)compactSum / totalBins;
        
        double compactVariance = 0;
        for (uint32_t c : extents) compactVariance += (c - compactAvg) * (c - compactAvg);
        compactVariance /= totalBins;
        compactStddev = sqrt(compactVariance);
        
        std::cerr << "\n[Statistics] CompactScan Bin Counts:\n";
        std::cerr << "  Min=" << compactMin << ", Max=" << compactMax << ", Avg=" << std::fixed << std::setprecision(2) << compactAvg << "\n";
        std::cerr << "  Non-empty bins: " << compactNonEmpty << " / " << totalBins << "\n";
        std::cerr << "  StdDev: " << std::fixed << std::setprecision(2) << compactStddev << "\n";
        std::cerr << "  CoV: " << std::fixed << std::setprecision(4) << (compactAvg > 0 ? compactStddev / compactAvg : 0) << "\n";

        // Create result buffer
        vkcore::PBuffer resultBuffer(new vkcore::Buffer(vd));
        resultBuffer->create(resultSizeUints * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            vkcore::MemoryType::Internal);

        // Run queries
        std::cerr << "\n--- CompactScanIndex Query Performance (warmup + mean of " << QUERY_COUNT << " runs) ---\n";
        compactQueryTimes.reserve(NUM_QUERIES);
        
        for (int q = 0; q < NUM_QUERIES; q++) {
            std::vector<uint32_t> queryData = {queries[q][0], queries[q][1], queries[q][2], queries[q][3], queries[q][4], queries[q][5]};
            
            // Warmup
            loadUsingStagingBuf((char *)queryData.data(), queryData.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            compactIndex->runRangeQueries(queryBuffer, 1, resultBuffer);
            vd->device->waitIdle();
            
            std::vector<double> qt;
            qt.reserve(QUERY_COUNT);
            for (int r = 0; r < QUERY_COUNT; r++) {
                loadUsingStagingBuf((char *)queryData.data(), queryData.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
                CPUTimer qTimer;
                qTimer.start();
                compactIndex->runRangeQueries(queryBuffer, 1, resultBuffer);
                vd->device->waitIdle();
                qt.push_back(double(qTimer.stop()) / 1000000.0);
            }
            
            double t = 0;
            for (double v : qt) t += v;
            t /= qt.size();
            compactTotTime += t;
            compactQueryTimes.push_back(t * 1000.0);
            
            std::vector<uint32_t> resultData(resultSizeUints);
            readUsingStagingBuf((char *)resultData.data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
            uint32_t count = 0;
            for (uint32_t val : resultData) count += __builtin_popcount(val);
            compactCounts[q] = count;
            
            std::cerr << "Query " << (q + 1) << " (" << std::fixed << std::setprecision(0) << ((q + 1) * 10.0) << "%): " 
                      << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Count: " << count << "\n";
        }
        std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((compactTotTime * 1000.0) / NUM_QUERIES) << " ms\n";

        // Cleanup CompactScan
        resultBuffer->destroy();
        compactIndex.reset();
        vd->device->waitIdle();
    }

    // =========================================================
    // Comparison (only if both were run)
    // =========================================================
    if (runEquiDepth && runCompactScan) {
        std::cerr << "\n###############################################\n";
        std::cerr << "# COMPARISON\n";
        std::cerr << "###############################################\n";

        std::cerr << "\n--- Query Count Verification (EquiDepth vs CompactScan) ---\n";
        std::cerr << "Query | Selectivity | EquiDepth  | CompactScan | Match | EquiDepth (ms) | CompactScan (ms) | Speedup\n";
        std::cerr << "------+-------------+------------+-------------+-------+----------------+------------------+--------\n";
        bool allMatch = true;
        for (int q = 0; q < NUM_QUERIES; q++) {
            double selectivity = (q + 1) * 10.0;
            bool match = (equiDepthCounts[q] == compactCounts[q]);
            if (!match) allMatch = false;

            double eTime = equiDepthQueryTimes[q];
            double cTime = compactQueryTimes[q];
            double speedup = eTime > 0.0 ? cTime / eTime : 0.0;

            std::cerr << std::setw(5) << (q + 1) << " | " 
                      << std::setw(10) << std::fixed << std::setprecision(0) << selectivity << "% | "
                      << std::setw(10) << equiDepthCounts[q] << " | "
                      << std::setw(11) << compactCounts[q] << " | "
                      << std::setw(5) << (match ? "YES" : "NO") << " | "
                      << std::setw(14) << std::fixed << std::setprecision(3) << eTime << " | "
                      << std::setw(16) << std::fixed << std::setprecision(3) << cTime << " | "
                      << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
        std::cerr << "\nAll counts match: " << (allMatch ? "YES" : "NO") << "\n";
        
        std::cerr << "\n========================================\n";
        std::cerr << "BUILD TIME COMPARISON\n";
        std::cerr << "========================================\n";
        std::cerr << "Equi-Depth Index build time: " << std::fixed << std::setprecision(2) << (equiDepthBuildTime * 1000.0) << " ms\n";
        std::cerr << "CompactScan Index build time: " << std::fixed << std::setprecision(2) << (compactBuildTime * 1000.0) << " ms\n";
        double buildSpeedup = equiDepthBuildTime > 0 ? compactBuildTime / equiDepthBuildTime : 0;
        std::cerr << "Build speedup (CompactScan/EquiDepth): " << std::fixed << std::setprecision(2) << buildSpeedup << "x\n";
        
        std::cerr << "\n========================================\n";
        std::cerr << "QUERY TIME COMPARISON\n";
        std::cerr << "========================================\n";
        std::cerr << "Equi-Depth avg query time: " << std::fixed << std::setprecision(3) << ((equiDepthTotTime * 1000.0) / NUM_QUERIES) << " ms\n";
        std::cerr << "CompactScan avg query time: " << std::fixed << std::setprecision(3) << ((compactTotTime * 1000.0) / NUM_QUERIES) << " ms\n";
        double querySpeedup = equiDepthTotTime > 0 ? compactTotTime / equiDepthTotTime : 0;
        std::cerr << "Query speedup (CompactScan/EquiDepth): " << std::fixed << std::setprecision(2) << querySpeedup << "x\n";
        
        std::cerr << "\n========================================\n";
        std::cerr << "BIN DISTRIBUTION COMPARISON\n";
        std::cerr << "========================================\n";
        std::cerr << "                    Equi-Depth    CompactScan\n";
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
    }

    // Cleanup
    queryBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 51 Complete.\n";
}
