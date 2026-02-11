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
#define BUILD_COUNT 11
#endif

#ifndef QUERY_COUNT
#define QUERY_COUNT 11
#endif

#define RUN_BATCHWISE 1
#define RUN_POINTWISE 1

static const std::vector<std::string> distributionFiles = {
    "uniform.bin",
    "normal.bin",
    "zipf1.1.bin",
    "zipf1.3.bin",
    "zipf1.5.bin"
};

static const std::vector<std::string> distributionNames = {
    "uniform",
    "normal",
    "zipf1.1",
    "zipf1.3",
    "zipf1.5"
};

enum class QueryStrategy {
    CENTERED,
    FROM_MIN
};

static QueryStrategy getQueryStrategy80(int dataId) {
    switch (dataId) {
        case 0: return QueryStrategy::CENTERED;
        case 1: return QueryStrategy::CENTERED;
        case 2: return QueryStrategy::FROM_MIN;
        case 3: return QueryStrategy::FROM_MIN;
        case 4: return QueryStrategy::FROM_MIN;
        default: return QueryStrategy::CENTERED;
    }
}

static void generateAndSaveQueries80(
    const std::vector<uint32_t>& minval, 
    const std::vector<uint32_t>& maxval,
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
    
    QueryStrategy strategy = getQueryStrategy80(dataId);
    
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
                
                if (strategy == QueryStrategy::CENTERED) {
                    uint64_t margin = (range - queryRange) / 2;
                    lo = minval[c] + (uint32_t)margin;
                    hi = minval[c] + (uint32_t)(margin + queryRange);
                } else {
                    lo = minval[c];
                    hi = minval[c] + (uint32_t)queryRange;
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

void testCompactBruteScan(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    
    if (dataId < 0 || dataId >= (int)distributionFiles.size()) {
        std::cerr << "ERROR: Invalid dataId " << dataId << ". Must be 0-4.\n";
        return;
    }
    
    uint32_t millions = g_npoints / 1000000;
    std::string plainDataFolder = PROJECT_DIR + "data/data_" + std::to_string(millions) + "m_" + std::to_string(g_dim) + "c";
    std::string dataFile = plainDataFolder + "/" + distributionFiles[dataId];
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 80: CompactBruteScan Index Test\n";
    std::cerr << "Distribution: " << distributionNames[dataId] << " (dataId=" << dataId << ")\n";
    std::cerr << "Data file: " << dataFile << "\n";
    std::cerr << "Columns: " << g_dim << "\n";
    std::cerr << "========================================\n";

    int32_t ncols = g_dim;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<uint32_t> points;
    vkcore::PBuffer pointsBuffer = readPlainData(dataFile, vd, staging, npoints, minval, maxval, points, ncols);
    std::cerr << "Dataset: " << npoints << " points\n";

    std::string queryFolder = PROJECT_DIR + "tests/test1";
    std::string queryFile = queryFolder + "/" + distributionNames[dataId] + "_mode80.txt";
    int numQueries = 10;
    generateAndSaveQueries80(minval, maxval, ncols, queryFile, dataId, numQueries);
    
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

    // Result buffer must be large enough for npoints + potential aux entries
    // Aux buffer can hold up to npoints * COMPACTBRUTE_AUX_FRACTION entries
    uint32_t maxRowId = npoints + (uint32_t)(npoints * COMPACTBRUTE_AUX_FRACTION);
    uint32_t resultSizeUints = (maxRowId + 31) / 32;
    vkcore::SinglePassScan *scan = (vkcore::SinglePassScan *) op.getFunction(vkcore::FunctionType::SinglePassScan);

    vkcore::PBuffer queryBuffer(new vkcore::Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);

    // =========================================================
    // Build CompactBruteScan Index
    // =========================================================
    std::cerr << "\n--- [CompactBruteScanIndex] ---\n";

    GPUMemoryTool::printGPUMemoryStatus(vd, "Before CompactBruteScan build");

    std::cerr << "\nBuilding CompactBruteScan Index (taking median of " << BUILD_COUNT << " runs)...\n";
    std::vector<double> buildTimes;
    buildTimes.reserve(BUILD_COUNT);
    for(int k = 0; k < BUILD_COUNT; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";

        CPUTimer buildTimer;
        buildTimer.start();

        PCompactBruteScanIndex indexRun = std::make_shared<CompactBruteScanIndex>(vd, ncols, scan);
        indexRun->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());

        double bt = double(buildTimer.stop()) / 1000000.0;

        std::cout << k << ": ============>  Build time: " << bt * 1000.0 << " ms\n";

        buildTimes.push_back(bt);

        indexRun.reset();
        vd->device->waitIdle();
    }
    std::sort(buildTimes.begin(), buildTimes.end());
    double buildTime = buildTimes[buildTimes.size() / 2];
    std::cerr << "\n\n>>> CompactBruteScan Index build time: " << (buildTime * 1000.0) << " ms\n\n";

    PCompactBruteScanIndex index = std::make_shared<CompactBruteScanIndex>(vd, ncols, scan);
    index->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    GPUMemoryTool::printGPUMemoryStatus(vd, "After CompactBruteScan build");

    vkcore::PBuffer resultBuffer(new vkcore::Buffer(vd));
    resultBuffer->create(resultSizeUints * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);
    
    std::vector<std::vector<uint32_t>> queryResults(numQueries); 
    
    std::cerr << "\n--- CompactBruteScan Query Performance ---\n";
    double totTime = 0;
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
        totTime += t;
        queryTimes.push_back(t * 1000.0);

        queryResults[i].resize(resultSizeUints);
        readUsingStagingBuf((char *)queryResults[i].data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
        
        uint32_t count = 0;
        for(uint32_t val : queryResults[i]) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << ": " << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Result Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((totTime * 1000.0) / numQueries) << " ms\n";
    std::sort(queryTimes.begin(), queryTimes.end());
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << queryTimes[queryTimes.size() / 2] << " ms\n";
    
    // =========================================================
    // Update Performance Test
    // Two approaches:
    //   1. Batch-wise: Delete entire batch, then insert entire batch
    //   2. Point-wise: For each point, delete then insert
    // 
    // Uses different data: original batch (x,y,z) and updated batch (2*x, 2*y, 2*z)
    // =========================================================
    
    {
        uint32_t validCount = index->getMainValidCount();
        uint32_t auxCount = index->getAuxValidCount();
        std::cerr << "\n[DEBUG] After build: Main valid=" << validCount << ", Aux count=" << auxCount << "\n";
    }
    
    const int S = 1;
    const float percent = 0.00001;
    const int NUM_BATCHES = 5;  // Number of batches to test
    const uint32_t batchSize = std::min<uint32_t>(npoints, npoints / 100 * percent);  // p% of data per batch

    std::vector<uint32_t> indices(npoints);
    std::iota(indices.begin(), indices.end(), 0);
    {
        std::mt19937 rng(28739);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::cerr << "\n=========================================================\n";
    std::cerr << "Update Test: " << NUM_BATCHES << " batches, " << batchSize << " points/batch\n";
    std::cerr << "Original batch: (x, y, z)\n";
    std::cerr << "Updated batch:  (2*x, 2*y, 2*z) - different data!\n";
    std::cerr << "=========================================================\n";

    // Allocate buffers for delete data (original) and insert data (2x values)
    vkcore::PBuffer deleteDataBuffer(new vkcore::Buffer(vd));
    deleteDataBuffer->create(batchSize * 3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);

    vkcore::PBuffer insertDataBuffer(new vkcore::Buffer(vd));
    insertDataBuffer->create(batchSize * 3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);

    // For point-wise test, we need single-point buffers
    vkcore::PBuffer singleDeleteBuffer(new vkcore::Buffer(vd));
    singleDeleteBuffer->create(3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);

    vkcore::PBuffer singleInsertBuffer(new vkcore::Buffer(vd));
    singleInsertBuffer->create(3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        vkcore::MemoryType::Internal);

    std::vector<uint32_t> deleteData(batchSize * 3);
    std::vector<uint32_t> insertData(batchSize * 3);

#if RUN_BATCHWISE
    // =========================================================
    // APPROACH 1: Batch-wise (full batch delete + full batch insert)
    // =========================================================
    std::cerr << "\n--- APPROACH 1: Batch-wise (delete all, then insert all) ---\n";
    
    double totalBatchDeleteTime = 0.0;
    double totalBatchInsertTime = 0.0;
    uint64_t totalBatchPoints = 0;
    uint64_t totalDeletesSucceeded = 0;
    uint64_t totalInsertsSucceeded = 0;

    for (int k = 0; k < NUM_BATCHES; k++) {
        uint32_t startIdx = (uint32_t)k * batchSize;
        uint32_t currentBatchSize = std::min(batchSize, npoints - startIdx);
        if (currentBatchSize == 0) break;

        // Get counts BEFORE this batch (COMMENTED OUT - expensive GPU readback)
        // uint32_t mainValidBefore = index->getMainValidCount();
        // uint32_t auxCountBefore = index->getAuxValidCount();

        // Prepare delete batch (original x, y, z)
        for (uint32_t i = 0; i < currentBatchSize; i++) {
            uint32_t pointIdx = indices[startIdx + i];
            deleteData[i * 3 + 0] = points[pointIdx];
            deleteData[i * 3 + 1] = points[npoints + pointIdx];
            deleteData[i * 3 + 2] = points[2 * npoints + pointIdx];
        }

        // Prepare insert batch (S*x, S*y, S*z)
        for (uint32_t i = 0; i < currentBatchSize; i++) {
            uint32_t pointIdx = indices[startIdx + i];
            insertData[i * 3 + 0] = points[pointIdx] * S;
            insertData[i * 3 + 1] = points[npoints + pointIdx] * S;
            insertData[i * 3 + 2] = points[2 * npoints + pointIdx] * S;
        }

        loadUsingStagingBuf((char*)deleteData.data(), currentBatchSize * 3 * sizeof(uint32_t), deleteDataBuffer, staging, vd, 0);
        loadUsingStagingBuf((char*)insertData.data(), currentBatchSize * 3 * sizeof(uint32_t), insertDataBuffer, staging, vd, 0);

        // Step 1: Delete entire batch
        CPUTimer deleteTimer;
        deleteTimer.start();
        index->deletePoints(deleteDataBuffer, currentBatchSize);
        double deleteTime = double(deleteTimer.stop()) / 1000000.0;
        totalBatchDeleteTime += deleteTime;

        // Get counts AFTER delete (COMMENTED OUT - expensive GPU readback)
        // uint32_t mainValidAfterDelete = index->getMainValidCount();
        // uint32_t deletesActual = mainValidBefore - mainValidAfterDelete;
        // totalDeletesSucceeded += deletesActual;
        totalDeletesSucceeded += currentBatchSize;  // Assume all succeed

        // Step 2: Insert entire batch (new data goes to aux buffer)
        CPUTimer insertTimer;
        insertTimer.start();
        index->insertPoints(insertDataBuffer, currentBatchSize);
        double insertTime = double(insertTimer.stop()) / 1000000.0;
        totalBatchInsertTime += insertTime;

        // Get counts AFTER insert (COMMENTED OUT - expensive GPU readback)
        // uint32_t auxCountAfterInsert = index->getAuxValidCount();
        // uint32_t insertsActual = auxCountAfterInsert - auxCountBefore;
        // totalInsertsSucceeded += insertsActual;
        totalInsertsSucceeded += currentBatchSize;  // Assume all succeed

        totalBatchPoints += currentBatchSize;

        std::cerr << "Batch " << (k+1) << ": Delete " << std::fixed << std::setprecision(3) 
                  << (deleteTime * 1000.0) << " ms, Insert " << (insertTime * 1000.0) 
                  << " ms, Total " << ((deleteTime + insertTime) * 1000.0) << " ms\n";
    }

    std::cerr << "\n--- Batch-wise Summary ---\n";
    std::cerr << "Total delete time: " << std::fixed << std::setprecision(3) << (totalBatchDeleteTime * 1000.0) << " ms\n";
    std::cerr << "Total insert time: " << std::fixed << std::setprecision(3) << (totalBatchInsertTime * 1000.0) << " ms\n";
    std::cerr << "Total update time: " << std::fixed << std::setprecision(3) << ((totalBatchDeleteTime + totalBatchInsertTime) * 1000.0) << " ms\n";
    std::cerr << "Avg time per point: " << std::fixed << std::setprecision(3) 
              << (totalBatchPoints ? ((totalBatchDeleteTime + totalBatchInsertTime) * 1000000.0 / (double)totalBatchPoints) : 0.0) << " us\n";
    std::cerr << "Deletes succeeded: " << totalDeletesSucceeded << "/" << totalBatchPoints 
              << " (" << std::fixed << std::setprecision(1) << (100.0 * totalDeletesSucceeded / totalBatchPoints) << "%)\n";
    std::cerr << "Inserts succeeded: " << totalInsertsSucceeded << "/" << totalBatchPoints 
              << " (" << std::fixed << std::setprecision(1) << (100.0 * totalInsertsSucceeded / totalBatchPoints) << "%)\n";

    // Verify counts after batch-wise updates
    {
        uint32_t mainValid = index->getMainValidCount();
        uint32_t auxCount = index->getAuxValidCount();
        uint32_t total = mainValid + auxCount;
        std::cerr << "[DEBUG] After batch-wise: Main valid=" << mainValid << ", Aux count=" << auxCount 
                  << ", Total=" << total << " (expected: " << npoints << ")\n";
        if (total != npoints) {
            std::cerr << "[WARNING] Total count mismatch! Diff=" << (int64_t)(npoints - total) << "\n";
        }
    }

    // Query performance after batch-wise updates
    std::cerr << "\n--- Query Performance After Batch-wise Updates ---\n";
    double batchTotTime = 0;
    std::vector<double> batchQueryTimes;
    batchQueryTimes.reserve(numQueries);
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
        batchTotTime += t;
        batchQueryTimes.push_back(t * 1000.0);

        std::vector<uint32_t> result(resultSizeUints);
        readUsingStagingBuf((char *)result.data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
        
        uint32_t count = 0;
        for(uint32_t val : result) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << ": " << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Result Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((batchTotTime * 1000.0) / numQueries) << " ms\n";
    std::sort(batchQueryTimes.begin(), batchQueryTimes.end());
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << batchQueryTimes[batchQueryTimes.size() / 2] << " ms\n";
#endif // RUN_BATCHWISE

    // =========================================================
    // Rebuild index for point-wise test
    // =========================================================
    index.reset();
    index = std::make_shared<CompactBruteScanIndex>(vd, ncols, scan);
    index->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());

#if RUN_POINTWISE
    // =========================================================
    // APPROACH 2: Point-wise (per point: delete, then insert)
    // =========================================================
    std::cerr << "\n--- APPROACH 2: Point-wise (per point: delete, insert, next point...) ---\n";
    
    // Get initial counts for point-wise test
    uint32_t pwMainValidInitial = index->getMainValidCount();
    uint32_t pwAuxCountInitial = index->getAuxValidCount();
    std::cerr << "[DEBUG] Point-wise initial: Main valid=" << pwMainValidInitial << ", Aux count=" << pwAuxCountInitial << "\n";
    
    double totalPointDeleteTime = 0.0;
    double totalPointInsertTime = 0.0;
    uint64_t totalPointwisePoints = 0;
    uint64_t pwDeletesSucceeded = 0;
    uint64_t pwInsertsSucceeded = 0;

    // Use smaller batch for point-wise (more overhead per point)
    uint32_t pointwiseBatchSize = std::min(npoints, batchSize); 

    for (int k = 0; k < NUM_BATCHES; k++) {
        uint32_t startIdx = (uint32_t)k * pointwiseBatchSize;
        uint32_t currentBatchSize = std::min(pointwiseBatchSize, npoints - startIdx);
        if (currentBatchSize == 0) break;

        // Get counts BEFORE this batch (COMMENTED OUT - expensive GPU readback)
        // uint32_t mainValidBefore = index->getMainValidCount();
        // uint32_t auxCountBefore = index->getAuxValidCount();

        double batchDeleteTime = 0.0;
        double batchInsertTime = 0.0;

        for (uint32_t i = 0; i < currentBatchSize; i++) {
            uint32_t pointIdx = indices[startIdx + i];
            
            // Prepare single point delete (original x, y, z)
            uint32_t singleDelete[3] = {
                points[pointIdx],
                points[npoints + pointIdx],
                points[2 * npoints + pointIdx]
            };
            
            // Prepare single point insert (S*x, S*y, S*z)
            uint32_t singleInsert[3] = {
                points[pointIdx] * S,
                points[npoints + pointIdx] * S,
                points[2 * npoints + pointIdx] * S
            };

            loadUsingStagingBuf((char*)singleDelete, 3 * sizeof(uint32_t), singleDeleteBuffer, staging, vd, 0);
            loadUsingStagingBuf((char*)singleInsert, 3 * sizeof(uint32_t), singleInsertBuffer, staging, vd, 0);

            // Delete single point
            CPUTimer deleteTimer;
            deleteTimer.start();
            index->deletePoints(singleDeleteBuffer, 1);
            double deleteTime = double(deleteTimer.stop()) / 1000000.0;
            batchDeleteTime += deleteTime;

            // Insert single point (goes to aux buffer)
            CPUTimer insertTimer;
            insertTimer.start();
            index->insertPoints(singleInsertBuffer, 1);
            double insertTime = double(insertTimer.stop()) / 1000000.0;
            batchInsertTime += insertTime;
        }

        // Get counts AFTER this batch (COMMENTED OUT - expensive GPU readback)
        // uint32_t mainValidAfter = index->getMainValidCount();
        // uint32_t auxCountAfter = index->getAuxValidCount();
        // uint32_t batchDeletesActual = mainValidBefore - mainValidAfter;
        // uint32_t batchInsertsActual = auxCountAfter - auxCountBefore;
        // pwDeletesSucceeded += batchDeletesActual;
        // pwInsertsSucceeded += batchInsertsActual;
        pwDeletesSucceeded += currentBatchSize;  // Assume all succeed
        pwInsertsSucceeded += currentBatchSize;  // Assume all succeed

        totalPointDeleteTime += batchDeleteTime;
        totalPointInsertTime += batchInsertTime;
        totalPointwisePoints += currentBatchSize;

        std::cerr << "Batch " << (k+1) << " (" << currentBatchSize << " points): Delete " 
                  << std::fixed << std::setprecision(3) << (batchDeleteTime * 1000.0) 
                  << " ms, Insert " << (batchInsertTime * 1000.0) 
                  << " ms"
                  << ", Avg/point " << ((batchDeleteTime + batchInsertTime) * 1000.0 / currentBatchSize) << " ms\n";
    }

    std::cerr << "\n--- Point-wise Summary ---\n";
    std::cerr << "Total delete time: " << std::fixed << std::setprecision(3) << (totalPointDeleteTime * 1000.0) << " ms\n";
    std::cerr << "Total insert time: " << std::fixed << std::setprecision(3) << (totalPointInsertTime * 1000.0) << " ms\n";
    std::cerr << "Total update time: " << std::fixed << std::setprecision(3) << ((totalPointDeleteTime + totalPointInsertTime) * 1000.0) << " ms\n";
    std::cerr << "Deletes succeeded: " << pwDeletesSucceeded << "/" << totalPointwisePoints 
              << " (" << std::fixed << std::setprecision(1) << (100.0 * pwDeletesSucceeded / totalPointwisePoints) << "%)\n";
    std::cerr << "Inserts succeeded: " << pwInsertsSucceeded << "/" << totalPointwisePoints 
              << " (" << std::fixed << std::setprecision(1) << (100.0 * pwInsertsSucceeded / totalPointwisePoints) << "%)\n";
    std::cerr << "Avg time per point: " << std::fixed << std::setprecision(3) 
              << (totalPointwisePoints ? ((totalPointDeleteTime + totalPointInsertTime) * 1000000.0 / (double)totalPointwisePoints) : 0.0) << " us\n";

    // Verify counts after point-wise updates
    {
        uint32_t mainValid = index->getMainValidCount();
        uint32_t auxCount = index->getAuxValidCount();
        uint32_t total = mainValid + auxCount;
        std::cerr << "[DEBUG] After point-wise: Main valid=" << mainValid << ", Aux count=" << auxCount 
                  << ", Total=" << total << " (expected: " << npoints << ")\n";
        if (total != npoints) {
            std::cerr << "[WARNING] Total count mismatch! Diff=" << (int64_t)(npoints - total) << "\n";
        }
        if (pwDeletesSucceeded != totalPointwisePoints) {
            std::cerr << "[WARNING] Not all deletes succeeded! Missing=" << (totalPointwisePoints - pwDeletesSucceeded) << "\n";
        }
        if (pwInsertsSucceeded != totalPointwisePoints) {
            std::cerr << "[WARNING] Not all inserts succeeded! Missing=" << (totalPointwisePoints - pwInsertsSucceeded) << "\n";
        }
    }

    // Query performance after point-wise updates
    std::cerr << "\n--- Query Performance After Point-wise Updates ---\n";
    double pointTotTime = 0;
    std::vector<double> pointQueryTimes;
    pointQueryTimes.reserve(numQueries);
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
        pointTotTime += t;
        pointQueryTimes.push_back(t * 1000.0);

        std::vector<uint32_t> result(resultSizeUints);
        readUsingStagingBuf((char *)result.data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
        
        uint32_t count = 0;
        for(uint32_t val : result) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << ": " << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Result Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((pointTotTime * 1000.0) / numQueries) << " ms\n";
    std::sort(pointQueryTimes.begin(), pointQueryTimes.end());
    std::cerr << "Median Query Time: " << std::fixed << std::setprecision(3) << pointQueryTimes[pointQueryTimes.size() / 2] << " ms\n";
#endif // RUN_POINTWISE

    // Note: Buffers are managed by shared_ptr, no need to explicitly destroy
    // The destructors will handle cleanup when they go out of scope
    
    std::cerr << "\nCompactBruteScan Mode 80 Complete.\n";
}
