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

// Mode 25: Mode 22 structure with TPC-C data
// Combines Mode 22's test harness with Mode 24's TPC-C data generation

// Set RUNRASTER to 1 to run RasterScan2D, 0 to run CompactScanIndex
#ifndef RUNRASTER
#define RUNRASTER 0

#endif

#ifndef BUILD_COUNT
#define BUILD_COUNT 11
#endif

#ifndef QUERY_COUNT
#define QUERY_COUNT 11
#endif

// Verbose timing flags
#ifndef VERBOSE_RASTER
#define VERBOSE_RASTER 0
#endif

#ifndef VERBOSE_COMPACT
#define VERBOSE_COMPACT 0
#endif

// TPC-C Constants
static constexpr int32_t kDistrictsPerWarehouse = 100;
static constexpr int32_t kCustomerPerDistrict   = 30000;
static constexpr int64_t kCustomersPerWarehouse = kDistrictsPerWarehouse * kCustomerPerDistrict;

// Scale factors for TPC-C
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

// Generate TPC-C Customer table data with 3 columns: W_ID, D_ID, C_ID
// Returns data in column-major format: [W_ID...][D_ID...][C_ID...]
static void generateTPCCData25(
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

// Generate queries with selectivities 10%, 20%, ..., 100%
static void generateTPCCQueries25(
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
        
        // For 100% selectivity (last query), use exact min/max
        bool isFullRange = (q == numQueries - 1);
        
        for (int c = 0; c < ncols; c++) {
            uint32_t lo, hi;
            
            if (isFullRange) {
                lo = minval[c];
                hi = maxval[c];
            } else {
                uint64_t range = (uint64_t)maxval[c] - (uint64_t)minval[c];
                uint64_t queryRange = (uint64_t)(range * perDimSelectivity);
                
                // Center the query in the data range
                uint64_t margin = (range - queryRange) / 2;
                lo = minval[c] + (uint32_t)margin;
                hi = minval[c] + (uint32_t)(margin + queryRange);
            }
            
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

void testTPCCMode25(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    using namespace vkcore;
    
    // dataId is scale factor indicator (0-10)
    int scaleIdx = std::min(dataId, (int)scaleFactors.size() - 1);
    int64_t targetCustomers = scaleFactors[scaleIdx];
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 25: TPC-C with Mode 22 Structure\n";
    std::cerr << "Scale: " << scaleNames[scaleIdx] << " (" << targetCustomers << " customers)\n";
    std::cerr << "========================================\n";
    
    // =========================================================
    // Generate TPC-C Data
    // =========================================================
    int32_t ncols = 3;
    std::vector<uint32_t> data;
    uint32_t minW, maxW, minD, maxD, minC, maxC;
    generateTPCCData25(targetCustomers, data, minW, maxW, minD, maxD, minC, maxC);
    
    uint32_t npoints = static_cast<uint32_t>(targetCustomers);
    std::vector<uint32_t> minval = {minW, minD, minC};
    std::vector<uint32_t> maxval = {maxW, maxD, maxC};
    
    // Upload data to GPU (column-major format)
    vkcore::PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(data.size() * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)data.data(), data.size() * sizeof(uint32_t), pointsBuffer, staging, vd, 0);
    
    std::cerr << "Dataset: " << npoints << " points\n";
    
    // Generate queries
    int numQueries = 10;
    std::vector<uint32_t> targets;
    generateTPCCQueries25(minval, maxval, ncols, targets, numQueries);
    
    // Result Buffer
    uint32_t resultSizeUints = (npoints + 31) / 32;
    
    // Get SinglePassScan for GPU prefix sum
    vkcore::SinglePassScan *scan = (vkcore::SinglePassScan *) op.getFunction(vkcore::FunctionType::SinglePassScan);
    
    // Query Buffer (shared)
    vkcore::PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);

#if RUNRASTER == 1
    // =========================================================
    // PART A: Run RasterScan2D
    // =========================================================
    std::cerr << "\n--- [RasterScan2D] ---\n";
    
    vkcore::ReduceMax *reduce = (vkcore::ReduceMax *) op.getFunction(vkcore::FunctionType::ReduceMax);
    GPUMemoryTool::printGPUMemoryStatus(vd, "Before RasterScan2D build");
    std::cerr << "Building RasterScan2D Index (taking median of " << BUILD_COUNT << ")...\n";
    std::vector<double> rsBuildTimes;
    rsBuildTimes.reserve(BUILD_COUNT);
    for(int k=0; k<BUILD_COUNT; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";

        PBufferCache bufsRun(new CommonBufferPool(vd));
        RasterScan2D rsRun(vd, bufsRun, scan, reduce, ncols);

        CPUTimer rsBuildTimer;
        rsBuildTimer.start();

        PRasterIndex tmpIndex = rsRun.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
        
        double bt = double(rsBuildTimer.stop()) / 1000000.0;
        rsBuildTimes.push_back(bt);

        std::cout << k << ": ============>  Build time: " << bt * 1000.0 << " ms\n";

        tmpIndex.reset();
        bufsRun->destroy();
        bufsRun.reset();
        vd->device->waitIdle();
    }
    std::sort(rsBuildTimes.begin(), rsBuildTimes.end());
    double rsBuildTime = rsBuildTimes[rsBuildTimes.size() / 2];

    std::cerr << "\n\n>>> RasterScan2D Index build time: " << (rsBuildTime * 1000.0) << " ms\n\n";

    PBufferCache bufs(new CommonBufferPool(vd));
    RasterScan2D rs(vd, bufs, scan, reduce, ncols);
    PRasterIndex rsIndex = rs.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());

    GPUMemoryTool::printGPUMemoryStatus(vd, "After RasterScan2D build");

    std::vector<std::vector<uint32_t>> rasterResults(numQueries);
    
    std::cerr << "\n--- RasterScan2D Query Performance ---\n";
    double rsTotTime = 0;
    std::vector<double> rsQueryTimes;
    rsQueryTimes.reserve(numQueries);

    for (int i = 0; i < numQueries; i++) {
        int in = i * 6;
        // RasterScan2D format: x1, y1, x2, y2, z1, z2
        std::vector<uint32_t> queries = {targets[in], targets[in+2], targets[in+1], targets[in+3], targets[in+4], targets[in+5]};

        std::vector<double> qt;
        qt.reserve(QUERY_COUNT);
        for(int r = 0; r < QUERY_COUNT; r++) {
            loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            CPUTimer rsQTimer;
            rsQTimer.start();
            rs.runRangeQueries(rsIndex, queryBuffer, 1);
            double t = double(rsQTimer.stop()) / 1000000.0;
            qt.push_back(t);
        }
        std::sort(qt.begin(), qt.end());
        double t = qt[qt.size() / 2];
        rsTotTime += t;
        rsQueryTimes.push_back(t * 1000.0);
        
        // Read back
        rasterResults[i].resize(resultSizeUints);
        readUsingStagingBuf((char *)rasterResults[i].data(), resultSizeUints * sizeof(uint32_t), bufs->resBuffer, staging, vd);
        
        uint32_t count = 0;
        for(uint32_t val : rasterResults[i]) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << ": " << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Result Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((rsTotTime * 1000.0) / numQueries) << " ms\n";
    std::sort(rsQueryTimes.begin(), rsQueryTimes.end());
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << rsQueryTimes[rsQueryTimes.size() / 2] << " ms\n";
    
    // Clean up RasterScan2D resources
    bufs->destroy();
    queryBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nRasterScan2D Mode Complete.\n";

#else // RUNRASTER == 0
    // =========================================================
    // PART B: Run CompactScanIndex
    // =========================================================
    std::cerr << "\n--- [CompactScanIndex] ---\n";

    GPUMemoryTool::printGPUMemoryStatus(vd, "Before CompactScanIndex build");

    std::cerr << "\nBuilding Compact Index (taking median of " << BUILD_COUNT << " runs)...\n";
    std::vector<double> compactBuildTimes;
    compactBuildTimes.reserve(BUILD_COUNT);
    for(int k=0; k<BUILD_COUNT; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";

        CPUTimer buildTimer;
        buildTimer.start();

        PCompactScanIndex compactRun = std::make_shared<CompactScanIndex>(vd, ncols, scan);
        compactRun->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());

        double bt = double(buildTimer.stop()) / 1000000.0;

        std::cout << k << ": ============>  Build time: " << bt * 1000.0 << " ms\n";

        compactBuildTimes.push_back(bt);

        compactRun.reset();
        vd->device->waitIdle();
    }
    std::sort(compactBuildTimes.begin(), compactBuildTimes.end());
    double buildTime = compactBuildTimes[compactBuildTimes.size() / 2];
    std::cerr << "\n\n>>>Compact Index build time: " << (buildTime * 1000.0) << " ms\n\n";

    PCompactScanIndex compactIndex = std::make_shared<CompactScanIndex>(vd, ncols, scan);
    // Note: initialize() is called internally by buildIndex(), no need to call it separately
    compactIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    GPUMemoryTool::printGPUMemoryStatus(vd, "After CompactScanIndex build");

    vkcore::PBuffer compactResultBuffer(new Buffer(vd));
    compactResultBuffer->create(resultSizeUints * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    // Store Compact Results
    std::vector<std::vector<uint32_t>> compactResults(numQueries); 
    
    std::cerr << "\n--- Compact Index Query Performance ---\n";
    double compactTotTime = 0;
    std::vector<double> compactQueryTimes;
    compactQueryTimes.reserve(numQueries);
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
            compactIndex->runRangeQueries(queryBuffer, 1, compactResultBuffer);
            double t = double(qTimer.stop()) / 1000000.0;
            qt.push_back(t);
        }
        std::sort(qt.begin(), qt.end());
        double t = qt[qt.size() / 2];
        compactTotTime += t;
        compactQueryTimes.push_back(t * 1000.0);

        // Read back
        compactResults[i].resize(resultSizeUints);
        readUsingStagingBuf((char *)compactResults[i].data(), resultSizeUints * sizeof(uint32_t), compactResultBuffer, staging, vd);
        
        // Log count
        uint32_t count = 0;
        for(uint32_t val : compactResults[i]) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << ": " << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Result Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((compactTotTime * 1000.0) / numQueries) << " ms\n";
    std::sort(compactQueryTimes.begin(), compactQueryTimes.end());
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << compactQueryTimes[compactQueryTimes.size() / 2] << " ms\n";
    
    // =========================================================
    // Delete/Insert Performance (CompactScanIndex)
    // =========================================================
    
    // Verify initial count after build
    {
        uint64_t capacity = compactIndex->totalAllocatedCapacity;
        std::vector<CompactEntry> hostData(capacity);
        readUsingStagingBuf((char*)hostData.data(), capacity * sizeof(CompactEntry), compactIndex->dataBuffer, staging, vd);
        uint32_t validCount = 0;
        uint32_t invalidCount = 0;
        for(const auto& entry : hostData) {
            if(entry.rowId & 0x80000000) validCount++;
            else if(entry.x != 0 || entry.y != 0 || entry.z != 0) invalidCount++;
        }
        std::cerr << "\n[DEBUG] After build: Expected=" << npoints << ", Valid=" << validCount << ", InvalidNonZero=" << invalidCount << "\n";
        
        // Check extent buffer
        uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
        std::vector<uint32_t> extentData(totalBins);
        readUsingStagingBuf((char*)extentData.data(), totalBins * sizeof(uint32_t), compactIndex->extentBuffer, staging, vd);
        uint32_t maxExtent = 0, sumExtent = 0;
        for(uint32_t e : extentData) {
            if(e > maxExtent) maxExtent = e;
            sumExtent += e;
        }
        std::cerr << "[DEBUG] Extent: max=" << maxExtent << ", sum=" << sumExtent << "\n";
    }
    
    // Split dataset into K batches after shuffling indices, then delete/insert each batch once.
    const int K = 100000000;

    std::vector<uint32_t> indices(npoints);
    std::iota(indices.begin(), indices.end(), 0);
    {
        std::mt19937 rng(100);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    const uint32_t batchSize = std::max<uint32_t>(1u, npoints / (uint32_t)K);

    std::cerr << "\n--- Delete/Insert Performance ---\n";
    std::cerr << "K: " << K << "\n";
    std::cerr << "Batch size: " << batchSize << " (last batch may be larger due to remainder)\n";

    vkcore::PBuffer deleteDataBuffer(new Buffer(vd));
    deleteDataBuffer->create(batchSize * 3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);

    std::vector<uint32_t> deleteData(batchSize * 3);
    
    // Lambda for count verification
    auto verifyCount = [&](uint32_t expected, const std::string& label) {
        uint64_t capacity = compactIndex->totalAllocatedCapacity;
        std::vector<CompactEntry> hostData(capacity);
        
        readUsingStagingBuf((char*)hostData.data(), capacity * sizeof(CompactEntry), compactIndex->dataBuffer, staging, vd);
        
        uint32_t validCount = 0;
        for(const auto& entry : hostData) {
            if(entry.rowId & 0x80000000) {
                validCount++;
            }
        }
        std::cerr << label << ": Expected=" << expected << ", Actual=" << validCount << " [" << (expected==validCount ? "PASS" : "FAIL") << "]\n";
    };

    double totalDelTime = 0.0;
    double totalInsTime = 0.0;
    uint64_t totalProcessedPoints = 0;

    // --- Delete/Insert Performance ---
    for (int k = 0; k < K; k++) {
        uint32_t startIdx = (uint32_t)k * batchSize;
        uint32_t endIdx = (k == K - 1) ? npoints : std::min(npoints, (uint32_t)(k + 1) * batchSize);
        uint32_t currentBatchSize = endIdx - startIdx;
        if (currentBatchSize == 0) continue;

        for (uint32_t i = 0; i < currentBatchSize; i++) {
            uint32_t pointIdx = indices[startIdx + i];
            deleteData[i * 3 + 0] = data[pointIdx];
            deleteData[i * 3 + 1] = data[npoints + pointIdx];
            deleteData[i * 3 + 2] = data[2 * npoints + pointIdx];
        }

        loadUsingStagingBuf((char*)deleteData.data(), currentBatchSize * 3 * sizeof(uint32_t), deleteDataBuffer, staging, vd, 0);

        CPUTimer delTimer;
        delTimer.start();
        compactIndex->deletePoints(deleteDataBuffer, currentBatchSize);
        double delTime = double(delTimer.stop()) / 1000000.0;
        totalDelTime += delTime;
        totalProcessedPoints += currentBatchSize;

        std::cerr << k << ">>> Delete Time: " << (delTime * 1000.0) << " ms (" << (delTime * 1000000.0 / currentBatchSize) << " us/point)\n";

        CPUTimer insTimer;
        insTimer.start();
        compactIndex->insertPoints(deleteDataBuffer, currentBatchSize);
        double insTime = double(insTimer.stop()) / 1000000.0;
        totalInsTime += insTime;

        std::cerr << k << ">>> Insert Time: " << (insTime * 1000.0) << " ms (" << (insTime * 1000000.0 / currentBatchSize) << " us/point)\n";

        std::cout << std::endl;
    }

    std::cerr << "\n--- Delete/Insert (K cycles) Summary ---\n";
    std::cerr << "Scale: " << scaleNames[scaleIdx] << " (" << npoints << " customers)\n";
    std::cerr << "K: " << K << " (nominal batch size " << batchSize << ")\n";
    std::cerr << "Total delete time: " << std::fixed << std::setprecision(3) << (totalDelTime * 1000.0) << " ms\n";
    std::cerr << "Total insert time: " << std::fixed << std::setprecision(3) << (totalInsTime * 1000.0) << " ms\n";
    std::cerr << "Avg delete time: " << std::fixed << std::setprecision(3) << ((totalDelTime * 1000.0) / K) << " ms (" << (totalProcessedPoints ? (totalDelTime * 1000000.0 / (double)totalProcessedPoints) : 0.0) << " us/point)\n";
    std::cerr << "Avg insert time: " << std::fixed << std::setprecision(3) << ((totalInsTime * 1000.0) / K) << " ms (" << (totalProcessedPoints ? (totalInsTime * 1000000.0 / (double)totalProcessedPoints) : 0.0) << " us/point)\n";
    
    // Final cleanup
    deleteDataBuffer->destroy();
    compactResultBuffer->destroy();
    queryBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nMode 25 (TPC-C with Mode 22 Structure) Complete.\n";
#endif // RUNRASTER
}
