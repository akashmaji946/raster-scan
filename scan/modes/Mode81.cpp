#include "RunModes.hpp"
#include "../CompactBruteScanIndex.hpp"
#include "../BufferPool.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sys/stat.h>
#include <algorithm>
#include <random>

#ifndef BUILD_COUNT
#define BUILD_COUNT 5
#endif

#ifndef QUERY_COUNT
#define QUERY_COUNT 11
#endif

#ifndef Q
#define Q 10
#endif

#define RUN_POINTWISE 1
#define RUN_BATCHWISE 0

// Set to 1 to enable count verification (expensive GPU readbacks)
#define RUN_TEST 0

// =========================================================
// CONTROLLABLE PARAMETERS FOR MODE 81
// =========================================================
#define MODE81_AUX_BUF_SIZE 10000       // Aux buffer capacity in slots
#define MODE81_TOTAL_UPDATES 50000    // Total number of point updates
#define MODE81_BATCH_SIZE 1000        // Points per batch (for logging)
#define MODE81_QUERY_COUNT 11         // Iterations per query (take median)
#define N 400000000
// =========================================================

// Override aux buffer size for Mode 81
// Formula: AUX_BUF_SIZE / npoints = fraction
// For 400M points with 100 slots: 100 / 400000000 = 0.00000025
#undef COMPACTBRUTE_AUX_FRACTION
#define COMPACTBRUTE_AUX_FRACTION (double(MODE81_AUX_BUF_SIZE) / N)

static const std::vector<std::string> distributionFiles = {
    "uniform.bin",
    "normal.bin",
    "zipf1.01.bin",
    "zipf1.05.bin",
    "zipf1.1.bin",
    "tpcc.bin"
};

static const std::vector<std::string> distributionNames = {
    "uniform",
    "normal",
    "zipf1.01",
    "zipf1.05",
    "zipf1.1",
    "tpcc"
};

enum class QueryStrategy {
    CENTERED,
    FROM_MIN
};

static QueryStrategy getQueryStrategy81(int dataId) {
    switch (dataId) {
        case 0: return QueryStrategy::CENTERED;  // uniform
        case 1: return QueryStrategy::CENTERED;  // normal
        case 2: return QueryStrategy::FROM_MIN;  // zipf1.01
        case 3: return QueryStrategy::FROM_MIN;  // zipf1.05
        case 4: return QueryStrategy::FROM_MIN;  // zipf1.1
        case 5: return QueryStrategy::CENTERED;  // tpcc
        default: return QueryStrategy::CENTERED;
    }
}

static void generateAndSaveQueries81(
    const std::vector<uint32_t>& minval, 
    const std::vector<uint32_t>& maxval,
    const std::vector<uint32_t>& points,
    uint32_t npoints,
    int ncols,
    const std::string& outputFile,
    int dataId,
    int numQueries = 10
) {
    std::string dir = outputFile.substr(0, outputFile.find_last_of('/'));
    mkdir(dir.c_str(), 0755);
    
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "ERROR: Cannot create query file: " << outputFile << "\n";
        return;
    }
    
    QueryStrategy strategy = getQueryStrategy81(dataId);

    const uint32_t sampleCount = std::min<uint32_t>(npoints, 1000000u);
    const uint32_t sampleStride = std::max<uint32_t>(1u, npoints / sampleCount);
    std::vector<std::vector<uint32_t>> samples(ncols);
    for (int c = 0; c < ncols; c++) {
        samples[c].reserve(sampleCount);
        const uint32_t* col = points.data() + static_cast<size_t>(c) * npoints;
        for (uint32_t i = 0; i < npoints && samples[c].size() < sampleCount; i += sampleStride) {
            samples[c].push_back(col[i]);
        }
        std::sort(samples[c].begin(), samples[c].end());
    }

    auto quantile = [&](int col, double q) -> uint32_t {
        if (samples[col].empty()) return minval[col];
        size_t idx = static_cast<size_t>(q * (samples[col].size() - 1));
        return samples[col][idx];
    };

    for (int q = 1; q <= numQueries; q++) {
        double selectivity = q / double(numQueries);
        double perDimSelectivity = std::pow(selectivity, 1.0 / ncols);
        
        for (int c = 0; c < ncols; c++) {
            uint32_t lo, hi;
            bool isFullRange = (q == numQueries);
            
            if (isFullRange) {
                lo = minval[c];
                hi = maxval[c];
            } else {
                double loQ, hiQ;
                if (strategy == QueryStrategy::CENTERED) {
                    loQ = 0.5 - perDimSelectivity / 2.0;
                    hiQ = 0.5 + perDimSelectivity / 2.0;
                } else {
                    loQ = 0.0;
                    hiQ = perDimSelectivity;
                }

                lo = quantile(c, loQ);
                hi = quantile(c, hiQ);
                if (hi < lo) {
                    uint32_t tmp = lo;
                    lo = hi;
                    hi = tmp;
                }
            }
            
            out << "lt " << lo << "\n";
            out << "lt " << hi << "\n";
        }
        
        for (int c = ncols; c < 3; c++) {
            out << "lt 0\n";
            out << "lt 4294967295\n";
        }
    }
    
    out.close();
    std::cerr << "Saved " << numQueries << " queries to: " << outputFile << "\n";
}

void testCompactBruteScanAuxOverflow(vkcore::PVkDevice vd, int dataId, vkcore::PBuffer staging, const std::string& testFolder) {
    
    if (dataId < 0 || dataId >= (int)distributionFiles.size()) {
        std::cerr << "ERROR: Invalid dataId " << dataId << ". Must be 0-5.\n";
        return;
    }
    
    uint32_t millions = g_npoints / 1000000;
    std::string plainDataFolder = PROJECT_DIR + "data/data_" + std::to_string(millions) + "m_" + std::to_string(g_dim) + "c";
    std::string dataFile = plainDataFolder + "/" + distributionFiles[dataId];
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 81: CompactBruteScan Aux Overflow Test\n";
    std::cerr << "Distribution: " << distributionNames[dataId] << " (dataId=" << dataId << ")\n";
    std::cerr << "Data file: " << dataFile << "\n";
    std::cerr << "Columns: " << g_dim << "\n";
    std::cerr << "Aux buffer size: " << MODE81_AUX_BUF_SIZE << " slots\n";
    std::cerr << "Total updates: " << MODE81_TOTAL_UPDATES << " points\n";
    std::cerr << "Batch size: " << MODE81_BATCH_SIZE << " points\n";
    std::cerr << "========================================\n";

    int32_t ncols = g_dim;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<uint32_t> points;
    vkcore::PBuffer pointsBuffer = readPlainData(dataFile, vd, staging, npoints, minval, maxval, points, ncols);
    std::cerr << "Dataset: " << npoints << " points\n";

    std::string queryFolder = PROJECT_DIR + "tests/test1";
    std::string queryFile = testFolder + "/" + distributionNames[dataId] + "_mode81.txt";
    int numQueries = Q;
    generateAndSaveQueries81(minval, maxval, points, npoints, ncols, queryFile, dataId, numQueries);
    
    std::vector<uint32_t> targets(numQueries * 6);
    {
        std::ifstream qf(queryFile);
        std::string cmd;
        uint32_t val;
        int idx = 0;
        while (qf >> cmd >> val && idx < numQueries * 6) {
            targets[idx++] = val;
        }
    }
    std::cerr << "Loaded " << numQueries << " queries (selectivity 10%-100%)\n";

    vkcore::PBuffer queryBuffer = std::make_shared<vkcore::Buffer>(vd);
    queryBuffer->create(targets.size() * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);
    loadUsingStagingBuf((char*)targets.data(), targets.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);

    vkcore::PBuffer resultBuffer = std::make_shared<vkcore::Buffer>(vd);
    // Allocate extra space for new rowIds from inserts (rowIds can go beyond npoints)
    size_t maxRowId = npoints + MODE81_TOTAL_UPDATES;
    size_t resultSize = ((maxRowId + 31) / 32) * sizeof(uint32_t);
    resultBuffer->create(resultSize, 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
        vkcore::MemoryType::Internal);

    vkcore::SinglePassScan* scan = new vkcore::SinglePassScan(vd);
    scan->initialize();

    std::cerr << "\n--- [CompactBruteScanIndex] ---\n";
    GPUMemoryTool::printGPUMemoryStatus(vd, "Before CompactBruteScan build");

    std::shared_ptr<CompactBruteScanIndex> index;
    
    std::cerr << "\nBuilding CompactBruteScan Index (taking median of " << BUILD_COUNT << " runs)...\n";
    std::vector<double> buildTimes;
    buildTimes.reserve(BUILD_COUNT);
    for (int run = 0; run < BUILD_COUNT; run++) {
        if (run > 0) std::cerr << "  Run " << (run + 1) << "...\n";
        
        CPUTimer buildTimer;
        buildTimer.start();
        
        std::shared_ptr<CompactBruteScanIndex> indexRun = std::make_shared<CompactBruteScanIndex>(vd, ncols, scan);
        indexRun->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
        
        double bt = double(buildTimer.stop()) / 1000000.0;
        
        std::cout << run << ": ============>  Build time: " << bt * 1000.0 << " ms\n";
        
        buildTimes.push_back(bt);
        
        indexRun.reset();
        vd->device->waitIdle();
    }
    
    std::sort(buildTimes.begin(), buildTimes.end());
    double medianBuildTime = buildTimes[buildTimes.size() / 2];
    std::cerr << "\n\n>>> CompactBruteScan Index build time: " << (medianBuildTime * 1000.0) << " ms\n\n";
    
    index.reset();
    index = std::make_shared<CompactBruteScanIndex>(vd, ncols, scan);
    index->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    GPUMemoryTool::printGPUMemoryStatus(vd, "After CompactBruteScan build");

    std::cerr << "\n--- CompactBruteScan Query Performance ---\n";
    std::vector<double> queryTimes;
    std::vector<uint32_t> queryCounts;
    uint32_t resultSizeUints = resultSize / sizeof(uint32_t);
    
    for (int q = 0; q < numQueries; q++) {
        // Load query q from targets array into queryBuffer
        int qOffset = q * 6;
        std::vector<uint32_t> queryData = {
            targets[qOffset], targets[qOffset+1], 
            targets[qOffset+2], targets[qOffset+3], 
            targets[qOffset+4], targets[qOffset+5]
        };
        loadUsingStagingBuf((char*)queryData.data(), 6 * sizeof(uint32_t), queryBuffer, staging, vd, 0);

        CPUTimer queryTimer;
        queryTimer.start();
        index->runRangeQueries(queryBuffer, 1, resultBuffer);
        double queryTime = double(queryTimer.stop()) / 1000000.0;
        queryTimes.push_back(queryTime * 1000.0);  // Convert to ms

        std::vector<uint32_t> result(resultSizeUints);
        readUsingStagingBuf((char*)result.data(), resultSize, resultBuffer, staging, vd);
        
        uint32_t count = 0;
        for (uint32_t val : result) count += __builtin_popcount(val);
        queryCounts.push_back(count);
        
        std::cerr << "Query " << (q + 1) << ": " << std::fixed << std::setprecision(3) << (queryTime * 1000.0) << " ms, Result Count: " << count << "\n";
    }

    double avgQueryTime = std::accumulate(queryTimes.begin(), queryTimes.end(), 0.0) / queryTimes.size();
    std::sort(queryTimes.begin(), queryTimes.end());
    double medianQueryTime = queryTimes[queryTimes.size() / 2];
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << avgQueryTime << " ms\n";
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << medianQueryTime << " ms\n";

#if RUN_TEST
    std::cerr << "\n[DEBUG] After build: Main valid=" << index->getMainValidCount() 
              << ", Aux count=" << index->getAuxValidCount() << "\n";
#endif

    // =========================================================
    // UPDATE TEST with controllable parameters
    // =========================================================
    
    const int TOTAL_UPDATES = MODE81_TOTAL_UPDATES;
    const int BATCH_SIZE = MODE81_BATCH_SIZE;
    const int AUX_BUF_SIZE = MODE81_AUX_BUF_SIZE;
    const int NUM_BATCHES = (TOTAL_UPDATES + BATCH_SIZE - 1) / BATCH_SIZE;
    const int EXPECTED_PUSHES = TOTAL_UPDATES / AUX_BUF_SIZE;
    
    std::cerr << "\n=========================================================\n";
    std::cerr << "Update Test: " << TOTAL_UPDATES << " points (" << NUM_BATCHES << " batches of " << BATCH_SIZE << " each)\n";
    std::cerr << "Aux buffer capacity: " << AUX_BUF_SIZE << " slots\n";
    std::cerr << "Expected pushes: " << EXPECTED_PUSHES << "\n";
    std::cerr << "=========================================================\n\n";
    
    const int S = 1;  // No scaling for this test

    std::vector<uint32_t> indices(npoints);
    std::iota(indices.begin(), indices.end(), 0);
    {
        std::random_device rd;
        std::mt19937 g(42);
        std::shuffle(indices.begin(), indices.end(), g);
    }

    vkcore::PBuffer singleDeleteBuffer = std::make_shared<vkcore::Buffer>(vd);
    singleDeleteBuffer->create(3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);

    vkcore::PBuffer singleInsertBuffer = std::make_shared<vkcore::Buffer>(vd);
    singleInsertBuffer->create(3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);

#if RUN_POINTWISE
    std::cerr << "--- APPROACH: Point-wise (per point: delete, insert, check overflow) ---\n";
    
#if RUN_TEST
    std::cerr << "[DEBUG] Point-wise initial: Main valid=" << index->getMainValidCount() 
              << ", Aux count=" << index->getAuxValidCount() << "\n";
#endif

    uint32_t totalPointsUpdated = 0;
    uint32_t auxPushCount = 0;
    double totalDeleteTime = 0.0;
    double totalInsertTime = 0.0;
    
    // Helper lambda to run all queries and report timing
    auto runAllQueriesWithTiming = [&](const std::string& label) {
        vd->device->waitIdle();  // Ensure clean GPU state before timing
        
        // Warmup query to avoid cold cache effects
        {
            std::vector<uint32_t> warmupQuery = {targets[0], targets[1], targets[2], targets[3], targets[4], targets[5]};
            loadUsingStagingBuf((char*)warmupQuery.data(), 6 * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            index->runRangeQueries(queryBuffer, 1, resultBuffer);
        }
        
        std::cerr << "\n--- Query Timing: " << label << " ---\n";
        std::vector<double> medianTimes;
        for (int q = 0; q < numQueries; q++) {
            int qOffset = q * 6;
            std::vector<uint32_t> queryData = {
                targets[qOffset], targets[qOffset+1], 
                targets[qOffset+2], targets[qOffset+3], 
                targets[qOffset+4], targets[qOffset+5]
            };
            
            // Run multiple iterations and take median (like Mode80)
            std::vector<double> iterTimes;
            iterTimes.reserve(MODE81_QUERY_COUNT);
            for (int r = 0; r < MODE81_QUERY_COUNT; r++) {
                loadUsingStagingBuf((char*)queryData.data(), 6 * sizeof(uint32_t), queryBuffer, staging, vd, 0);
                CPUTimer qt;
                qt.start();
                index->runRangeQueries(queryBuffer, 1, resultBuffer);
                double t = double(qt.stop()) / 1000000.0 * 1000.0;  // ms
                iterTimes.push_back(t);
            }
            std::sort(iterTimes.begin(), iterTimes.end());
            double medianT = iterTimes[iterTimes.size() / 2];
            medianTimes.push_back(medianT);
            
            std::vector<uint32_t> res(resultSizeUints);
            readUsingStagingBuf((char*)res.data(), resultSize, resultBuffer, staging, vd);
            uint32_t cnt = 0;
            for (uint32_t val : res) cnt += __builtin_popcount(val);
            std::cerr << "  Q" << (q+1) << ": " << std::fixed << std::setprecision(3) << medianT << " ms, Count: " << cnt << "\n";
        }
        double avg = std::accumulate(medianTimes.begin(), medianTimes.end(), 0.0) / medianTimes.size();
        std::sort(medianTimes.begin(), medianTimes.end());
        double med = medianTimes[medianTimes.size() / 2];
        std::cerr << "  Avg: " << std::fixed << std::setprecision(3) << avg << " ms, Median: " << med << " ms\n";
    };

    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        std::cerr << "\n--- Batch " << (batch + 1) << " (" << BATCH_SIZE << " points) ---\n";
        
        int batchStart = batch * BATCH_SIZE;
        int batchEnd = std::min(batchStart + BATCH_SIZE, TOTAL_UPDATES);
        double batchDeleteTime = 0.0;
        double batchInsertTime = 0.0;
        
        for (int i = batchStart; i < batchEnd; i++) {
            uint32_t pointIdx = indices[i];
            
            uint32_t singleDelete[3] = {
                points[pointIdx],
                points[npoints + pointIdx],
                points[2 * npoints + pointIdx]
            };
            
            uint32_t singleInsert[3] = {
                points[pointIdx] * S,
                points[npoints + pointIdx] * S,
                points[2 * npoints + pointIdx] * S
            };

            loadUsingStagingBuf((char*)singleDelete, 3 * sizeof(uint32_t), singleDeleteBuffer, staging, vd, 0);
            loadUsingStagingBuf((char*)singleInsert, 3 * sizeof(uint32_t), singleInsertBuffer, staging, vd, 0);

            CPUTimer deleteTimer;
            deleteTimer.start();
            index->deletePoints(singleDeleteBuffer, 1);
            double deleteTime = double(deleteTimer.stop()) / 1000000.0;
            batchDeleteTime += deleteTime;

            CPUTimer insertTimer;
            insertTimer.start();
            index->insertPoints(singleInsertBuffer, 1);
            double insertTime = double(insertTimer.stop()) / 1000000.0;
            batchInsertTime += insertTime;
            
            totalPointsUpdated++;
            
            // Check if aux buffer is full
            if (index->cachedAuxCount >= (uint32_t)AUX_BUF_SIZE) {
                auxPushCount++;
                std::cerr << "\n[AUX OVERFLOW] After " << totalPointsUpdated << " updates, aux count = " 
                          << index->cachedAuxCount << " >= " << AUX_BUF_SIZE << "\n";
                
                // Query timing BEFORE push
                runAllQueriesWithTiming("Before Push #" + std::to_string(auxPushCount));
                
                std::cerr << "[PUSH #" << auxPushCount << "] Pushing aux buffer to main buffer...\n";
                
#if RUN_TEST
                uint32_t mainBefore = index->getMainValidCount();
                uint32_t auxBefore = index->getAuxValidCount();
                std::cerr << "[DEBUG] Before push: Main valid=" << mainBefore << ", Aux count=" << auxBefore << "\n";
#endif
                
                index->pushAuxToMain();
                
#if RUN_TEST
                uint32_t mainAfter = index->getMainValidCount();
                uint32_t auxAfter = index->getAuxValidCount();
                uint32_t totalAfter = mainAfter + auxAfter;
                std::cerr << "[DEBUG] After push: Main valid=" << mainAfter << ", Aux count=" << auxAfter 
                          << ", Total=" << totalAfter << " (expected: " << npoints << ")\n";
                
                if (totalAfter != npoints) {
                    std::cerr << "[ERROR] Point count mismatch after push! Expected " << npoints 
                              << ", got " << totalAfter << "\n";
                }
#endif
                
                // Query timing AFTER push
                runAllQueriesWithTiming("After Push #" + std::to_string(auxPushCount));
            }
        }
        
        totalDeleteTime += batchDeleteTime;
        totalInsertTime += batchInsertTime;
        int batchSize = batchEnd - batchStart;
        std::cerr << "Batch " << (batch + 1) << " complete. Total updates: " << totalPointsUpdated
                  << ", Delete: " << std::fixed << std::setprecision(3) << (batchDeleteTime * 1000.0) << " ms"
                  << ", Insert: " << (batchInsertTime * 1000.0) << " ms"
                  << ", Avg/point: " << ((batchDeleteTime + batchInsertTime) * 1000.0 / batchSize) << " ms\n";
        
#if RUN_TEST
        uint32_t mainValid = index->getMainValidCount();
        uint32_t auxCount = index->getAuxValidCount();
        uint32_t total = mainValid + auxCount;
        std::cerr << "[DEBUG] After batch " << (batch + 1) << ": Main valid=" << mainValid 
                  << ", Aux count=" << auxCount << ", Total=" << total << " (expected: " << npoints << ")\n";
#endif
    }
    
    std::cerr << "\n--- Point-wise Summary ---\n";
    std::cerr << "Total points updated: " << totalPointsUpdated << "\n";
    std::cerr << "Total delete time: " << std::fixed << std::setprecision(3) << (totalDeleteTime * 1000.0) << " ms\n";
    std::cerr << "Total insert time: " << std::fixed << std::setprecision(3) << (totalInsertTime * 1000.0) << " ms\n";
    std::cerr << "Avg time per point: " << std::fixed << std::setprecision(3) 
              << ((totalDeleteTime + totalInsertTime) * 1000000.0 / totalPointsUpdated) << " us\n";
    std::cerr << "Aux buffer pushes: " << auxPushCount << " (expected: " << EXPECTED_PUSHES << ")\n";
    
#if RUN_TEST
    uint32_t finalMain = index->getMainValidCount();
    uint32_t finalAux = index->getAuxValidCount();
    uint32_t finalTotal = finalMain + finalAux;
    std::cerr << "[DEBUG] Final: Main valid=" << finalMain << ", Aux count=" << finalAux 
              << ", Total=" << finalTotal << " (expected: " << npoints << ")\n";
#endif
#endif

    std::cerr << "\nCompactBruteScan Mode 81 Complete.\n";
}
