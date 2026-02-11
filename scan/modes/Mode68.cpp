#include "RunModes.hpp"
#include "../BruteForceIndex.hpp"
#include "../RasterScan2D.hpp"
#include "../BufferPool.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>

#ifndef BUILD_COUNT
#define BUILD_COUNT 11
#endif

#ifndef QUERY_COUNT
#define QUERY_COUNT 11
#endif

// TPC-C Constants
static constexpr int32_t kDistrictsPerWarehouse = 100;
static constexpr int32_t kCustomerPerDistrict   = 3000;
static constexpr int64_t kCustomersPerWarehouse = kDistrictsPerWarehouse * kCustomerPerDistrict;

// Generate TPC-C Customer table data
static void generateTPCCDataMode68(
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

// Generate 100 random point queries using actual data points from the TPC-C dataset
static std::vector<std::array<uint32_t, 6>> generatePointQueriesMode68(
    const std::vector<uint32_t>& data,  // actual data points (column-major)
    uint32_t npoints,
    int ncols,
    int numQueries = 100
) {
    std::vector<std::array<uint32_t, 6>> queries(numQueries);
    
    // Use a fixed seed for reproducibility
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> pointDist(0, npoints - 1);
    
    for (int q = 0; q < numQueries; q++) {
        // Pick a random point index from the actual dataset
        uint32_t pointIdx = pointDist(rng);
        
        for (int c = 0; c < 3; c++) {
            uint32_t lo, hi;
            
            if (c >= ncols) {
                // For dimensions beyond ncols, use full range
                lo = 0;
                hi = UINT32_MAX;
            } else {
                // Use the actual coordinate from the selected data point
                // Data is column-major: data[c * npoints + pointIdx]
                uint32_t point = data[c * npoints + pointIdx];
                lo = point;
                hi = point;  // Point query: min == max
            }
            
            queries[q][c * 2] = lo;      // min for dimension c
            queries[q][c * 2 + 1] = hi;  // max for dimension c
        }
    }
    
    return queries;
}

void testBruteForceVsRasterScanTPCCPointQueries(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    // Run mode: 0=both, 1=BruteForce only, 2=RasterScan only
    bool runBruteForce = (g_runmode == 0 || g_runmode == 1);
    bool runRasterScan = (g_runmode == 0 || g_runmode == 2);
    
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
    std::cerr << "MODE 68: BruteForce vs RasterScan (TPC-C, 100 point queries)\n";
    std::cerr << "Scale: " << scaleNames[scaleIdx] << " (" << targetCustomers << " customers)\n";
    std::cerr << "Run mode: " << g_runmode << " (0=both, 1=BruteForce, 2=RasterScan)\n";
    std::cerr << "========================================\n";
    
    // Generate TPC-C Data
    std::vector<uint32_t> data;
    uint32_t minW, maxW, minD, maxD, minC, maxC;
    
    CPUTimer genTimer;
    genTimer.start();
    generateTPCCDataMode68(targetCustomers, data, minW, maxW, minD, maxD, minC, maxC);
    double genTime = double(genTimer.stop()) / 1000000.0;
    std::cerr << "[TPC-C] Data generation time: " << (genTime * 1000.0) << " ms\n";
    
    uint32_t npoints = static_cast<uint32_t>(targetCustomers);
    int ncols = 3;
    
    // Upload to GPU
    GPUMemoryTool::printGPUMemoryStatus(vd, "Before TPC-C data upload");
    
    vkcore::PBuffer pointsBuffer(new Buffer(vd));
    size_t dataSize = data.size() * sizeof(uint32_t);
    pointsBuffer->create(dataSize, 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, 
        MemoryType::Internal);
    
    loadUsingStagingBuf((char*)data.data(), dataSize, pointsBuffer, staging, vd, 0);
    
    GPUMemoryTool::printGPUMemoryStatus(vd, "After TPC-C data upload");
    std::cerr << "[TPC-C] Dataset: " << npoints << " points, " << ncols << " columns\n";
    
    std::vector<uint32_t> minval = {minW, minD, minC};
    std::vector<uint32_t> maxval = {maxW, maxD, maxC};
    
    vkcore::SinglePassScan *scan = (vkcore::SinglePassScan *) op.getFunction(vkcore::FunctionType::SinglePassScan);
    vkcore::ReduceMax *reduce = (vkcore::ReduceMax *) op.getFunction(vkcore::FunctionType::ReduceMax);

    // Query setup - 100 point queries from actual data
    const int NUM_QUERIES = 100;
    std::vector<std::array<uint32_t, 6>> queries = generatePointQueriesMode68(data, npoints, ncols, NUM_QUERIES);
    
    uint32_t resultSizeUints = (npoints + 31) / 32;
    
    vkcore::PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);

    // Variables for comparison
    double bruteForceBuildTime = 0;
    double rasterBuildTime = 0;
    double bruteForceTotTime = 0;
    double rasterTotTime = 0;
    std::vector<double> bruteForceQueryTimes;
    std::vector<double> rasterQueryTimes;
    std::vector<uint32_t> bruteForceCounts(NUM_QUERIES, 0);
    std::vector<uint32_t> rasterCounts(NUM_QUERIES, 0);

    // =========================================================
    // PART 1: BruteForce (Build + Query)
    // =========================================================
    if (runBruteForce) {
        std::cerr << "\n###############################################\n";
        std::cerr << "# PART 1: BruteForceIndex (Build + Query)\n";
        std::cerr << "###############################################\n";

        GPUMemoryTool::printGPUMemoryStatus(vd, "Before BruteForceIndex");

        // Build timing
        std::cerr << "\nBuilding BruteForce Index (warmup + mean of " << BUILD_COUNT << " runs)...\n";
        std::vector<double> buildTimes;
        buildTimes.reserve(BUILD_COUNT);
        
        // Warmup
        {
            std::cerr << "  Warmup run (shader compilation)...\n";
            PBruteForceIndex warmup = std::make_shared<BruteForceIndex>(vd, ncols);
            warmup->initialize();
            warmup->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
            vd->device->waitIdle();
            warmup.reset();
            vd->device->waitIdle();
        }
        
        for (int k = 0; k < BUILD_COUNT; k++) {
            std::cerr << "  Run " << k + 1 << "/" << BUILD_COUNT << "...\n";

            PBruteForceIndex idx = std::make_shared<BruteForceIndex>(vd, ncols);
            idx->initialize();
            
            CPUTimer buildTimer;
            buildTimer.start();
            idx->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
            vd->device->waitIdle();
            double bt = double(buildTimer.stop()) / 1000000.0;

            std::cout << k << ": ============> BruteForce Build time: " << bt * 1000.0 << " ms\n";
            buildTimes.push_back(bt);

            idx.reset();
            vd->device->waitIdle();
        }
        
        for (double t : buildTimes) bruteForceBuildTime += t;
        bruteForceBuildTime /= buildTimes.size();
        std::cerr << "\n>>> BruteForce Index build time: " << (bruteForceBuildTime * 1000.0) << " ms\n";

        // Build final index for queries
        PBruteForceIndex bruteForceIndex = std::make_shared<BruteForceIndex>(vd, ncols);
        bruteForceIndex->initialize();
        bruteForceIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
        GPUMemoryTool::printGPUMemoryStatus(vd, "After BruteForceIndex build");
        std::cout << "[BruteForceIndex] Total DataBuffer Size: " << bruteForceIndex->getSizeMB() << " MB\n";

        // Create result buffer
        vkcore::PBuffer resultBuffer(new Buffer(vd));
        resultBuffer->create(resultSizeUints * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            MemoryType::Internal);

        // Run queries
        std::cerr << "\n--- BruteForceIndex Query Performance (warmup + mean of " << QUERY_COUNT << " runs) ---\n";
        bruteForceQueryTimes.reserve(NUM_QUERIES);
        
        for (int q = 0; q < NUM_QUERIES; q++) {
            // BruteForce format: x1, x2, y1, y2, z1, z2
            std::vector<uint32_t> queryData = {queries[q][0], queries[q][1], queries[q][2], queries[q][3], queries[q][4], queries[q][5]};
            
            // Warmup
            loadUsingStagingBuf((char *)queryData.data(), queryData.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            bruteForceIndex->runRangeQueries(queryBuffer, 1, resultBuffer);
            vd->device->waitIdle();
            
            std::vector<double> qt;
            qt.reserve(QUERY_COUNT);
            for (int r = 0; r < QUERY_COUNT; r++) {
                loadUsingStagingBuf((char *)queryData.data(), queryData.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
                CPUTimer qTimer;
                qTimer.start();
                bruteForceIndex->runRangeQueries(queryBuffer, 1, resultBuffer);
                vd->device->waitIdle();
                qt.push_back(double(qTimer.stop()) / 1000000.0);
            }
            
            double t = 0;
            for (double v : qt) t += v;
            t /= qt.size();
            bruteForceTotTime += t;
            bruteForceQueryTimes.push_back(t * 1000.0);
            
            std::vector<uint32_t> resultData(resultSizeUints);
            readUsingStagingBuf((char *)resultData.data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
            uint32_t count = 0;
            for (uint32_t val : resultData) count += __builtin_popcount(val);
            bruteForceCounts[q] = count;
            
            std::cerr << "Query " << (q + 1) << " (point): " 
                      << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Count: " << count << "\n";
        }
        std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((bruteForceTotTime * 1000.0) / NUM_QUERIES) << " ms\n";

        // Cleanup BruteForce
        resultBuffer->destroy();
        bruteForceIndex.reset();
        vd->device->waitIdle();
        GPUMemoryTool::printGPUMemoryStatus(vd, "After BruteForceIndex cleanup");
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
            // RasterScan format: x1, y1, x2, y2, z1, z2
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
            
            std::cerr << "Query " << (q + 1) << " (point): " 
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
    if (runBruteForce && runRasterScan) {
        std::cerr << "\n###############################################\n";
        std::cerr << "# COMPARISON (100 point queries)\n";
        std::cerr << "###############################################\n";

        std::cerr << "\n--- Query Count Verification (BruteForce vs RasterScan) ---\n";
        std::cerr << "Query | BruteForce | RasterScan | Match | BruteForce (ms) | RasterScan (ms) | Speedup\n";
        std::cerr << "------+------------+------------+-------+-----------------+-----------------+--------\n";
        bool allMatch = true;
        int mismatchCount = 0;
        for (int q = 0; q < NUM_QUERIES; q++) {
            bool match = (bruteForceCounts[q] == rasterCounts[q]);
            if (!match) {
                allMatch = false;
                mismatchCount++;
            }

            double bTime = bruteForceQueryTimes[q];
            double rTime = rasterQueryTimes[q];
            double speedup = bTime > 0.0 ? rTime / bTime : 0.0;

            std::cerr << std::setw(5) << (q + 1) << " | "
                      << std::setw(10) << bruteForceCounts[q] << " | "
                      << std::setw(10) << rasterCounts[q] << " | "
                      << std::setw(5) << (match ? "YES" : "NO") << " | "
                      << std::setw(15) << std::fixed << std::setprecision(3) << bTime << " | "
                      << std::setw(15) << std::fixed << std::setprecision(3) << rTime << " | "
                      << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
        std::cerr << "\nAll counts match: " << (allMatch ? "YES" : "NO");
        if (!allMatch) std::cerr << " (" << mismatchCount << " mismatches)";
        std::cerr << "\n";
        
        std::cerr << "\n========================================\n";
        std::cerr << "BUILD TIME COMPARISON\n";
        std::cerr << "========================================\n";
        std::cerr << "BruteForce Index build time: " << std::fixed << std::setprecision(2) << (bruteForceBuildTime * 1000.0) << " ms\n";
        std::cerr << "RasterScan2D build time: " << std::fixed << std::setprecision(2) << (rasterBuildTime * 1000.0) << " ms\n";
        double buildSpeedup = bruteForceBuildTime > 0 ? rasterBuildTime / bruteForceBuildTime : 0;
        std::cerr << "Build speedup (RasterScan/BruteForce): " << std::fixed << std::setprecision(2) << buildSpeedup << "x\n";
        
        std::cerr << "\n========================================\n";
        std::cerr << "QUERY TIME COMPARISON\n";
        std::cerr << "========================================\n";
        std::cerr << "BruteForce avg query time: " << std::fixed << std::setprecision(3) << ((bruteForceTotTime * 1000.0) / NUM_QUERIES) << " ms\n";
        std::cerr << "RasterScan2D avg query time: " << std::fixed << std::setprecision(3) << ((rasterTotTime * 1000.0) / NUM_QUERIES) << " ms\n";
        double querySpeedup = bruteForceTotTime > 0 ? rasterTotTime / bruteForceTotTime : 0;
        std::cerr << "Query speedup (RasterScan/BruteForce): " << std::fixed << std::setprecision(2) << querySpeedup << "x\n";
    }

    // Cleanup
    queryBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 68 Complete.\n";
}
