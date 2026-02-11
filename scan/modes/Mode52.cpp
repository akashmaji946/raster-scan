#include "RunModes.hpp"
#include "../EquiDepthIndex.hpp"
#include "../RasterScan2D.hpp"
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
static const std::vector<std::string> distributionFilesMode52 = {
    "uniform.bin",
    "normal.bin",
    "zipf1.1.bin",
    "zipf1.3.bin",
    "zipf1.5.bin",
    "tpcc.bin"
};

static const std::vector<std::string> distributionNamesMode52 = {
    "uniform",
    "normal",
    "zipf1.1",
    "zipf1.3",
    "zipf1.5",
    "tpcc"
};

// Query strategy based on distribution type
enum class QueryStrategyMode52 { CENTERED, FROM_MIN };

static QueryStrategyMode52 getQueryStrategyMode52(int dataId) {
    switch (dataId) {
        case 0: return QueryStrategyMode52::CENTERED;  // Uniform
        case 1: return QueryStrategyMode52::CENTERED;  // Normal
        case 2: return QueryStrategyMode52::FROM_MIN;  // Zipf 1.1
        case 3: return QueryStrategyMode52::FROM_MIN;  // Zipf 1.3
        case 4: return QueryStrategyMode52::FROM_MIN;  // Zipf 1.5
        case 5: return QueryStrategyMode52::CENTERED;  // TPC-C
        default: return QueryStrategyMode52::CENTERED;
    }
}

// Generate queries with selectivities 10%, 20%, ..., 100%
static std::vector<std::array<uint32_t, 6>> generateQueriesMode52(
    const std::vector<uint32_t>& minval,
    const std::vector<uint32_t>& maxval,
    int ncols,
    int dataId,
    int numQueries = 10
) {
    std::vector<std::array<uint32_t, 6>> queries(numQueries);
    QueryStrategyMode52 strategy = getQueryStrategyMode52(dataId);
    
    std::cerr << "Query strategy: " << (strategy == QueryStrategyMode52::CENTERED ? "CENTERED" : "FROM_MIN") << "\n";
    
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
                
                if (strategy == QueryStrategyMode52::CENTERED) {
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

void testEquiDepthVsRasterScan(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    
    // Validate dataId (0-4 for the 5 distributions)
    if (dataId < 0 || dataId >= (int)distributionFilesMode52.size()) {
        std::cerr << "ERROR: Invalid dataId " << dataId << ". Must be 0-4.\n";
        std::cerr << "  0: uniform, 1: normal, 2: zipf1.1, 3: zipf1.3, 4: zipf1.5\n";
        return;
    }
    
    // Run mode: 0=both, 1=EquiDepth only, 2=RasterScan only
    bool runEquiDepth = (g_runmode == 0 || g_runmode == 1);
    bool runRasterScan = (g_runmode == 0 || g_runmode == 2);
    
    // Construct plain data folder path
    uint32_t millions = g_npoints / 1000000;
    std::string plainDataFolder = PROJECT_DIR + "data/data_" + std::to_string(millions) + "m_" + std::to_string(g_dim) + "c";
    std::string dataFile = plainDataFolder + "/" + distributionFilesMode52[dataId];
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 52: EquiDepth vs RasterScan Comparison\n";
    std::cerr << "Distribution: " << distributionNamesMode52[dataId] << " (dataId=" << dataId << ")\n";
    std::cerr << "Data file: " << dataFile << "\n";
    std::cerr << "Columns: " << g_dim << "\n";
    std::cerr << "Run mode: " << g_runmode << " (0=both, 1=EquiDepth, 2=RasterScan)\n";
    std::cerr << "========================================\n";

    // 1. Read Plain Dataset
    int32_t ncols = g_dim;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<uint32_t> points;
    vkcore::PBuffer pointsBuffer = readPlainData(dataFile, vd, staging, npoints, minval, maxval, points, ncols);
    std::cerr << "Dataset: " << npoints << " points\n";

    // Get operators
    vkcore::SinglePassScan *scan = (vkcore::SinglePassScan *) op.getFunction(vkcore::FunctionType::SinglePassScan);
    vkcore::ReduceMax *reduce = (vkcore::ReduceMax *) op.getFunction(vkcore::FunctionType::ReduceMax);

    // Result Buffer
    uint32_t resultSizeUints = (npoints + 31) / 32;
    const int NUM_QUERIES = 10;
    
    // Generate queries
    std::vector<std::array<uint32_t, 6>> queries = generateQueriesMode52(minval, maxval, ncols, dataId, NUM_QUERIES);
    
    // Create query buffer
    vkcore::PBuffer queryBuffer(new vkcore::Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);

    // Variables for comparison
    double equiDepthBuildTime = 0;
    double rasterBuildTime = 0;
    double equiDepthTotTime = 0;
    double rasterTotTime = 0;
    std::vector<double> equiDepthQueryTimes;
    std::vector<double> rasterQueryTimes;
    std::vector<uint32_t> equiDepthCounts(NUM_QUERIES, 0);
    std::vector<uint32_t> rasterCounts(NUM_QUERIES, 0);

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
    // PART 2: RasterScan2D (Build + Query)
    // =========================================================
    if (runRasterScan) {
        std::cerr << "\n###############################################\n";
        std::cerr << "# PART 2: RasterScan2D (Build + Query)\n";
        std::cerr << "###############################################\n";

        GPUMemoryTool::printGPUMemoryStatus(vd, "Before RasterScan2D");

        // Build timing
        std::cerr << "\nBuilding RasterScan2D Index (warmup + mean of " << BUILD_COUNT << " runs)...\n";
        std::vector<double> buildTimes;
        buildTimes.reserve(BUILD_COUNT);
        
        // Warmup run
        {
            std::cerr << "  Warmup run (shader compilation)...\n";
            PBufferCache bufsWarmup(new CommonBufferPool(vd));
            RasterScan2D rsWarmup(vd, bufsWarmup, scan, reduce, ncols);
            PRasterIndex tmpIndex = rsWarmup.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
            vd->device->waitIdle();
            tmpIndex.reset();
            bufsWarmup->destroy();
            bufsWarmup.reset();
            vd->device->waitIdle();
        }
        
        for (int k = 0; k < BUILD_COUNT; k++) {
            std::cerr << "  Run " << k + 1 << "/" << BUILD_COUNT << "...\n";

            PBufferCache bufsRun(new CommonBufferPool(vd));
            RasterScan2D rsRun(vd, bufsRun, scan, reduce, ncols);

            CPUTimer rsBuildTimer;
            rsBuildTimer.start();
            PRasterIndex tmpIndex = rsRun.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
            vd->device->waitIdle();
            double bt = double(rsBuildTimer.stop()) / 1000000.0;
            buildTimes.push_back(bt);

            std::cout << k << ": ============> RasterScan Build time: " << bt * 1000.0 << " ms\n";

            tmpIndex.reset();
            bufsRun->destroy();
            bufsRun.reset();
            vd->device->waitIdle();
        }
        
        for (double t : buildTimes) rasterBuildTime += t;
        rasterBuildTime /= buildTimes.size();
        std::cerr << "\n>>> RasterScan2D Index build time: " << (rasterBuildTime * 1000.0) << " ms\n";

        // Build final index for queries
        PBufferCache bufs(new CommonBufferPool(vd));
        RasterScan2D rs(vd, bufs, scan, reduce, ncols);
        PRasterIndex rsIndex = rs.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
        GPUMemoryTool::printGPUMemoryStatus(vd, "After RasterScan2D build");

        // Run queries
        std::cerr << "\n--- RasterScan2D Query Performance (warmup + mean of " << QUERY_COUNT << " runs) ---\n";
        rasterQueryTimes.reserve(NUM_QUERIES);
        
        for (int q = 0; q < NUM_QUERIES; q++) {
            std::vector<uint32_t> queryData = {queries[q][0], queries[q][2], queries[q][1], queries[q][3], queries[q][4], queries[q][5]};
            
            // Warmup
            loadUsingStagingBuf((char *)queryData.data(), queryData.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            rs.runRangeQueries(rsIndex, queryBuffer, 1);
            vd->device->waitIdle();
            
            std::vector<double> qt;
            qt.reserve(QUERY_COUNT);
            for (int r = 0; r < QUERY_COUNT; r++) {
                loadUsingStagingBuf((char *)queryData.data(), queryData.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
                CPUTimer qTimer;
                qTimer.start();
                rs.runRangeQueries(rsIndex, queryBuffer, 1);
                vd->device->waitIdle();
                qt.push_back(double(qTimer.stop()) / 1000000.0);
            }
            
            double t = 0;
            for (double v : qt) t += v;
            t /= qt.size();
            rasterTotTime += t;
            rasterQueryTimes.push_back(t * 1000.0);
            
            std::vector<uint32_t> resultData(resultSizeUints);
            readUsingStagingBuf((char *)resultData.data(), resultSizeUints * sizeof(uint32_t), bufs->resBuffer, staging, vd);
            uint32_t count = 0;
            for (uint32_t val : resultData) count += __builtin_popcount(val);
            rasterCounts[q] = count;
            
            std::cerr << "Query " << (q + 1) << " (" << std::fixed << std::setprecision(0) << ((q + 1) * 10.0) << "%): " 
                      << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Count: " << count << "\n";
        }
        std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((rasterTotTime * 1000.0) / NUM_QUERIES) << " ms\n";

        // Cleanup RasterScan
        bufs->destroy();
        vd->device->waitIdle();
    }

    // =========================================================
    // Comparison (only if both were run)
    // =========================================================
    if (runEquiDepth && runRasterScan) {
        std::cerr << "\n###############################################\n";
        std::cerr << "# COMPARISON\n";
        std::cerr << "###############################################\n";

        std::cerr << "\n--- Query Count Verification (EquiDepth vs RasterScan) ---\n";
        std::cerr << "Query | Selectivity | EquiDepth  | RasterScan | Match | EquiDepth (ms) | RasterScan (ms) | Speedup\n";
        std::cerr << "------+-------------+------------+------------+-------+----------------+-----------------+--------\n";
        bool allMatch = true;
        for (int q = 0; q < NUM_QUERIES; q++) {
            double selectivity = (q + 1) * 10.0;
            bool match = (equiDepthCounts[q] == rasterCounts[q]);
            if (!match) allMatch = false;

            double eTime = equiDepthQueryTimes[q];
            double rTime = rasterQueryTimes[q];
            double speedup = eTime > 0.0 ? rTime / eTime : 0.0;

            std::cerr << std::setw(5) << (q + 1) << " | " 
                      << std::setw(10) << std::fixed << std::setprecision(0) << selectivity << "% | "
                      << std::setw(10) << equiDepthCounts[q] << " | "
                      << std::setw(10) << rasterCounts[q] << " | "
                      << std::setw(5) << (match ? "YES" : "NO") << " | "
                      << std::setw(14) << std::fixed << std::setprecision(3) << eTime << " | "
                      << std::setw(15) << std::fixed << std::setprecision(3) << rTime << " | "
                      << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
        std::cerr << "\nAll counts match: " << (allMatch ? "YES" : "NO") << "\n";
        
        std::cerr << "\n========================================\n";
        std::cerr << "BUILD TIME COMPARISON\n";
        std::cerr << "========================================\n";
        std::cerr << "Equi-Depth Index build time: " << std::fixed << std::setprecision(2) << (equiDepthBuildTime * 1000.0) << " ms\n";
        std::cerr << "RasterScan2D build time: " << std::fixed << std::setprecision(2) << (rasterBuildTime * 1000.0) << " ms\n";
        double buildSpeedup = equiDepthBuildTime > 0 ? rasterBuildTime / equiDepthBuildTime : 0;
        std::cerr << "Build speedup (RasterScan/EquiDepth): " << std::fixed << std::setprecision(2) << buildSpeedup << "x\n";
        
        std::cerr << "\n========================================\n";
        std::cerr << "QUERY TIME COMPARISON\n";
        std::cerr << "========================================\n";
        std::cerr << "Equi-Depth avg query time: " << std::fixed << std::setprecision(3) << ((equiDepthTotTime * 1000.0) / NUM_QUERIES) << " ms\n";
        std::cerr << "RasterScan2D avg query time: " << std::fixed << std::setprecision(3) << ((rasterTotTime * 1000.0) / NUM_QUERIES) << " ms\n";
        double querySpeedup = equiDepthTotTime > 0 ? rasterTotTime / equiDepthTotTime : 0;
        std::cerr << "Query speedup (RasterScan/EquiDepth): " << std::fixed << std::setprecision(2) << querySpeedup << "x\n";
    }

    // Cleanup
    queryBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 52 Complete.\n";
}
