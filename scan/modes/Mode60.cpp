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

// TPC-C Constants (same as Mode53)
static constexpr int32_t kDistrictsPerWarehouse = 100;
static constexpr int32_t kCustomerPerDistrict   = 3000;
static constexpr int64_t kCustomersPerWarehouse = kDistrictsPerWarehouse * kCustomerPerDistrict;

// Generate TPC-C Customer table data
static void generateTPCCDataMode60(
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
    
    // Shuffle the rows randomly
    std::cerr << "[TPC-C] Shuffling data rows...\n";
    std::mt19937 rng(42);
    for (int64_t i = count - 1; i > 0; --i) {
        std::uniform_int_distribution<int64_t> dist(0, i);
        int64_t j = dist(rng);
        std::swap(W[i], W[j]);
        std::swap(D[i], D[j]);
        std::swap(C[i], C[j]);
    }
    
    minW = 1; maxW = static_cast<uint32_t>(warehouseCount);
    minD = 1; maxD = kDistrictsPerWarehouse;
    minC = 1; maxC = kCustomerPerDistrict;
    
    std::cerr << "[TPC-C] Generated " << count << " customers\n";
}

// Generate queries with selectivities 10%, 20%, ..., 100%
static void generateTPCCQueriesMode60(
    const std::vector<uint32_t>& minval, 
    const std::vector<uint32_t>& maxval,
    int ncols,
    std::vector<uint32_t>& targets,
    int numQueries = 10
) {
    targets.resize(numQueries * 6);
    
    for (int q = 0; q < numQueries; q++) {
        double overallSelectivity = (q + 1) * 0.1;
        double perDimSelectivity = std::pow(overallSelectivity, 1.0 / ncols);
        
        for (int c = 0; c < ncols; c++) {
            uint64_t range = (uint64_t)maxval[c] - (uint64_t)minval[c];
            uint64_t queryRange = (uint64_t)(range * perDimSelectivity);
            
            uint64_t margin = (range - queryRange) / 2;
            uint32_t lo = minval[c] + (uint32_t)margin;
            uint32_t hi = minval[c] + (uint32_t)(margin + queryRange);
            
            targets[q * 6 + c * 2] = lo;
            targets[q * 6 + c * 2 + 1] = hi;
        }
        for (int c = ncols; c < 3; c++) {
            targets[q * 6 + c * 2] = 0;
            targets[q * 6 + c * 2 + 1] = 0xFFFFFFFF;
        }
    }
    
    std::cerr << "[TPC-C] Generated " << numQueries << " queries (selectivity 10%-100%)\n";
}

void testBruteForceVsRasterScan(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
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
    std::cerr << "MODE 60: BruteForce vs RasterScan\n";
    std::cerr << "Scale: " << scaleNames[scaleIdx] << " (" << targetCustomers << " customers)\n";
    std::cerr << "Run mode: " << g_runmode << " (0=both, 1=BruteForce, 2=RasterScan)\n";
    std::cerr << "========================================\n";
    
    // Generate TPC-C Data
    std::vector<uint32_t> data;
    uint32_t minW, maxW, minD, maxD, minC, maxC;
    
    CPUTimer genTimer;
    genTimer.start();
    generateTPCCDataMode60(targetCustomers, data, minW, maxW, minD, maxD, minC, maxC);
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

    // Query setup
    const int numQueries = 10;
    std::vector<uint32_t> targets;
    generateTPCCQueriesMode60(minval, maxval, ncols, targets, numQueries);
    
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
    std::vector<uint32_t> bruteForceCounts(numQueries, 0);
    std::vector<uint32_t> rasterCounts(numQueries, 0);

    // =========================================================
    // PART 1: BruteForce (Build + Query)
    // =========================================================
    if (runBruteForce) {
        std::cerr << "\n###############################################\n";
        std::cerr << "# PART 1: BruteForceIndex (Build + Query)\n";
        std::cerr << "###############################################\n";

        GPUMemoryTool::printGPUMemoryStatus(vd, "Before BruteForceIndex");

        // Build timing
        std::cerr << "\nBuilding BruteForce Index (median of " << BUILD_COUNT << " runs)...\n";
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

            buildTimes.push_back(bt);

            idx.reset();
            vd->device->waitIdle();
        }
        
        std::sort(buildTimes.begin(), buildTimes.end());
        bruteForceBuildTime = buildTimes[buildTimes.size() / 2];
        std::cerr << "\n>>> BruteForce Index build time (median): " << (bruteForceBuildTime * 1000.0) << " ms\n";

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
        std::cerr << "\n--- BruteForceIndex Query Performance (median of " << QUERY_COUNT << " runs) ---\n";
        bruteForceQueryTimes.reserve(numQueries);
        
        for (int i = 0; i < numQueries; i++) {
            int in = i * 6;
            // BruteForce format: x1, x2, y1, y2, z1, z2
            std::vector<uint32_t> queryData = {targets[in], targets[in+1], targets[in+2], targets[in+3], targets[in+4], targets[in+5]};
            
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
            
            std::sort(qt.begin(), qt.end());
            double t = qt[qt.size() / 2];
            bruteForceTotTime += t;
            bruteForceQueryTimes.push_back(t * 1000.0);
            
            std::vector<uint32_t> result(resultSizeUints);
            readUsingStagingBuf((char *)result.data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
            uint32_t count = 0;
            for (uint32_t val : result) count += __builtin_popcount(val);
            bruteForceCounts[i] = count;
            
            std::cerr << "Query " << (i + 1) << " (" << ((i + 1) * 10) << "%): " 
                      << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Count: " << count << "\n";
        }
        std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((bruteForceTotTime * 1000.0) / numQueries) << " ms\n";
        std::sort(bruteForceQueryTimes.begin(), bruteForceQueryTimes.end());
        std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << bruteForceQueryTimes[bruteForceQueryTimes.size() / 2] << " ms\n";

        // Cleanup BruteForce
        resultBuffer->destroy();
        bruteForceIndex.reset();
        vd->device->waitIdle();
        GPUMemoryTool::printGPUMemoryStatus(vd, "After BruteForceIndex cleanup");
    }

    // =========================================================
    // PART 2: RasterScan (Build + Query)
    // =========================================================
    if (runRasterScan) {
        std::cerr << "\n###############################################\n";
        std::cerr << "# PART 2: RasterScan2D (Build + Query)\n";
        std::cerr << "###############################################\n";

        GPUMemoryTool::printGPUMemoryStatus(vd, "Before RasterScan2D");

        // Build timing
        std::cerr << "\nBuilding RasterScan2D Index (median of " << BUILD_COUNT << " runs)...\n";
        std::vector<double> buildTimes;
        buildTimes.reserve(BUILD_COUNT);
        
        for (int k = 0; k < BUILD_COUNT; k++) {
            std::cerr << "  Run " << k + 1 << "/" << BUILD_COUNT << "...\n";

            PBufferCache bufsRun(new CommonBufferPool(vd));
            RasterScan2D rsRun(vd, bufsRun, scan, reduce, ncols);
            rsRun.initalize();

            CPUTimer buildTimer;
            buildTimer.start();
            PRasterIndex tmpIndex = rsRun.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
            vd->device->waitIdle();
            double bt = double(buildTimer.stop()) / 1000000.0;
            buildTimes.push_back(bt);

            tmpIndex.reset();
            bufsRun->destroy();
            bufsRun.reset();
            vd->device->waitIdle();
        }

        std::sort(buildTimes.begin(), buildTimes.end());
        rasterBuildTime = buildTimes[buildTimes.size() / 2];
        std::cerr << "\n>>> RasterScan2D Index build time (median): " << (rasterBuildTime * 1000.0) << " ms\n";

        // Build final index for queries
        PBufferCache bufs(new CommonBufferPool(vd));
        RasterScan2D rs(vd, bufs, scan, reduce, ncols);
        rs.initalize();
        PRasterIndex rsIndex = rs.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
        GPUMemoryTool::printGPUMemoryStatus(vd, "After RasterScan2D build");

        // Run queries
        std::cerr << "\n--- RasterScan2D Query Performance (median of " << QUERY_COUNT << " runs) ---\n";
        rasterQueryTimes.reserve(numQueries);
        
        for (int i = 0; i < numQueries; i++) {
            int in = i * 6;
            // RasterScan2D format: x1, y1, x2, y2, z1, z2
            std::vector<uint32_t> queryData = {targets[in], targets[in+2], targets[in+1], targets[in+3], targets[in+4], targets[in+5]};
            
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
            
            std::sort(qt.begin(), qt.end());
            double t = qt[qt.size() / 2];
            rasterTotTime += t;
            rasterQueryTimes.push_back(t * 1000.0);
            
            std::vector<uint32_t> result(resultSizeUints);
            readUsingStagingBuf((char *)result.data(), resultSizeUints * sizeof(uint32_t), bufs->resBuffer, staging, vd);
            uint32_t count = 0;
            for (uint32_t val : result) count += __builtin_popcount(val);
            rasterCounts[i] = count;
            
            std::cerr << "Query " << (i + 1) << " (" << ((i + 1) * 10) << "%): " 
                      << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Count: " << count << "\n";
        }
        std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((rasterTotTime * 1000.0) / numQueries) << " ms\n";
        std::sort(rasterQueryTimes.begin(), rasterQueryTimes.end());
        std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << rasterQueryTimes[rasterQueryTimes.size() / 2] << " ms\n";

        // Cleanup RasterScan
        rsIndex.reset();
        bufs->destroy();
        bufs.reset();
        vd->device->waitIdle();
        GPUMemoryTool::printGPUMemoryStatus(vd, "After RasterScan2D cleanup");
    }

    // =========================================================
    // Comparison (only if both were run)
    // =========================================================
    if (runBruteForce && runRasterScan) {
        std::cerr << "\n###############################################\n";
        std::cerr << "# COMPARISON\n";
        std::cerr << "###############################################\n";

        std::cerr << "\n--- Query Count Verification (BruteForce vs RasterScan) ---\n";
        std::cerr << "Query | Selectivity | BruteForce | RasterScan | Match | BruteForce (ms) | RasterScan (ms) | Speedup\n";
        std::cerr << "------+-------------+------------+------------+-------+-----------------+-----------------+--------\n";
        bool allMatch = true;
        for (int q = 0; q < numQueries; q++) {
            double selectivity = (q + 1) * 10.0;
            bool match = (bruteForceCounts[q] == rasterCounts[q]);
            if (!match) allMatch = false;

            double bTime = bruteForceQueryTimes[q];
            double rTime = rasterQueryTimes[q];
            double speedup = bTime > 0.0 ? rTime / bTime : 0.0;

            std::cerr << std::setw(5) << (q + 1) << " | " 
                      << std::setw(10) << std::fixed << std::setprecision(0) << selectivity << "% | "
                      << std::setw(10) << bruteForceCounts[q] << " | "
                      << std::setw(10) << rasterCounts[q] << " | "
                      << std::setw(5) << (match ? "YES" : "NO") << " | "
                      << std::setw(15) << std::fixed << std::setprecision(3) << bTime << " | "
                      << std::setw(15) << std::fixed << std::setprecision(3) << rTime << " | "
                      << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
        std::cerr << "\nAll counts match: " << (allMatch ? "YES" : "NO") << "\n";
        
        std::cerr << "\n========================================\n";
        std::cerr << "BUILD TIME COMPARISON\n";
        std::cerr << "========================================\n";
        std::cerr << "BruteForce Index build time: " << std::fixed << std::setprecision(2) << (bruteForceBuildTime * 1000.0) << " ms\n";
        std::cerr << "RasterScan2D Index build time: " << std::fixed << std::setprecision(2) << (rasterBuildTime * 1000.0) << " ms\n";
        double buildSpeedup = bruteForceBuildTime > 0 ? rasterBuildTime / bruteForceBuildTime : 0;
        std::cerr << "Build speedup (RasterScan/BruteForce): " << std::fixed << std::setprecision(2) << buildSpeedup << "x\n";
        
        std::cerr << "\n========================================\n";
        std::cerr << "QUERY TIME COMPARISON\n";
        std::cerr << "========================================\n";
        std::cerr << "BruteForce avg query time: " << std::fixed << std::setprecision(3) << ((bruteForceTotTime * 1000.0) / numQueries) << " ms\n";
        std::cerr << "RasterScan avg query time: " << std::fixed << std::setprecision(3) << ((rasterTotTime * 1000.0) / numQueries) << " ms\n";
        double querySpeedup = bruteForceTotTime > 0 ? rasterTotTime / bruteForceTotTime : 0;
        std::cerr << "Query speedup (RasterScan/BruteForce): " << std::fixed << std::setprecision(2) << querySpeedup << "x\n";
    }

    // Cleanup
    queryBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 60 (BruteForce vs RasterScan) Complete.\n";
}
