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

// Set RUNRASTER to 1 to run RasterScan2D, 0 to run CompactScanIndex
#ifndef RUNRASTER
#define RUNRASTER 0

#endif

// Distribution names for dataId 0-4
static const std::vector<std::string> distributionFiles = {
    "uniform.bin",
    "normal.bin",
    "zipf1.1.bin",
    "zipf1.3.bin",
    "zipf1.5.bin",
    "tpcc.bin"
};

static const std::vector<std::string> distributionNames = {
    "uniform",
    "normal",
    "zipf1.1",
    "zipf1.3",
    "zipf1.5",
    "tpcc"
};

// Query generation strategy based on distribution type
enum class QueryStrategy {
    CENTERED,    // For uniform/normal: center queries in data range
    FROM_MIN     // For zipf: start queries from minimum (where data clusters)
};

static QueryStrategy getQueryStrategy(int dataId) {
    switch (dataId) {
        case 0: return QueryStrategy::CENTERED;
        case 1: return QueryStrategy::CENTERED;
        case 2: return QueryStrategy::FROM_MIN;
        case 3: return QueryStrategy::FROM_MIN;
        case 4: return QueryStrategy::FROM_MIN;
        case 5: return QueryStrategy::CENTERED;  // TPC-C
        default: return QueryStrategy::CENTERED;
    }
}

static void generateAndSaveQueries(
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

    QueryStrategy strategy = getQueryStrategy(dataId);
    std::cerr << "Query strategy: " << (strategy == QueryStrategy::CENTERED ? "CENTERED" : "FROM_MIN") << "\n";

    for (int q = 0; q < numQueries; q++) {
        double overallSelectivity = (q + 1) * 0.1;
        double perDimSelectivity = std::pow(overallSelectivity, 1.0 / ncols);

        for (int c = 0; c < ncols; c++) {
            uint64_t range = (uint64_t)maxval[c] - (uint64_t)minval[c];
            uint64_t queryRange = (uint64_t)(range * perDimSelectivity);

            uint32_t lo, hi;
            if (strategy == QueryStrategy::CENTERED) {
                uint64_t margin = (range - queryRange) / 2;
                lo = minval[c] + (uint32_t)margin;
                hi = minval[c] + (uint32_t)(margin + queryRange);
            } else {
                lo = minval[c];
                hi = minval[c] + (uint32_t)queryRange;
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

void testCompactIndexAndCompareStreaming(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    if (dataId < 0 || dataId >= (int)distributionFiles.size()) {
        std::cerr << "ERROR: Invalid dataId " << dataId << ". Must be 0-4.\n";
        std::cerr << "  0: uniform, 1: normal, 2: zipf1.1, 3: zipf1.3, 4: zipf1.5\n";
        return;
    }

    uint32_t millions = g_npoints / 1000000;
    std::string plainDataFolder = PROJECT_DIR + "data/data_" + std::to_string(millions) + "m_" + std::to_string(g_dim) + "c";
    std::string dataFile = plainDataFolder + "/" + distributionFiles[dataId];

    std::cerr << "\n========================================\n";
    std::cerr << "MODE 32: Mode 22 (Plain Data) + Streaming Build\n";
    std::cerr << "Distribution: " << distributionNames[dataId] << " (dataId=" << dataId << ")\n";
    std::cerr << "Data file: " << dataFile << "\n";
    std::cerr << "Columns: " << g_dim << "\n";
    std::cerr << "========================================\n";

    int32_t ncols = g_dim;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<uint32_t> points;

    readPlainDataCPU(dataFile, npoints, minval, maxval, points, ncols);
    std::cerr << "Dataset: " << npoints << " points\n";

    std::string queryFolder = PROJECT_DIR + "tests/test1";
    std::string queryFile = queryFolder + "/" + distributionNames[dataId] + ".txt";
    int numQueries = 10;
    generateAndSaveQueries(minval, maxval, ncols, queryFile, dataId, numQueries);

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

    uint32_t resultSizeUints = (npoints + 31) / 32;

    vkcore::SinglePassScan *scan = (vkcore::SinglePassScan *) op.getFunction(vkcore::FunctionType::SinglePassScan);

    vkcore::PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);

    const uint32_t BATCH_POINTS = 50000000;

#if RUNRASTER == 1
    std::cerr << "\n--- [RasterScan2D Streaming] ---\n";

    vkcore::ReduceMax *reduce = (vkcore::ReduceMax *) op.getFunction(vkcore::FunctionType::ReduceMax);
    PBufferCache bufs(new CommonBufferPool(vd));

    RasterScan2D rs(vd, bufs, scan, reduce, ncols);

    GPUMemoryTool::printGPUMemoryStatus(vd, "Before RasterScan2D build");
    std::cerr << "Building RasterScan2D Index (streaming) ...\n";

    double minRsBuildTime = 1e9;
    PRasterIndex rsIndex;
    for(int k=0; k<3; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";
        CPUTimer rsBuildTimer;
        rsBuildTimer.start();
        rsIndex = rs.buildIndexStreaming(points, npoints, minval.data(), maxval.data(), staging, BATCH_POINTS);
        double bt = double(rsBuildTimer.stop()) / 1000000.0;
        if(bt < minRsBuildTime) minRsBuildTime = bt;
        if(k < 2) rsIndex.reset();
    }

    double rsBuildTime = minRsBuildTime;
    std::cerr << ">>> RasterScan2D Index build time: " << (rsBuildTime * 1000.0) << " ms\n";
    GPUMemoryTool::printGPUMemoryStatus(vd, "After RasterScan2D build");

    std::vector<std::vector<uint32_t>> rasterResults(numQueries);

    std::cerr << "\n--- RasterScan2D Query Performance ---\n";
    double rsTotTime = 0;

    for (int i = 0; i < numQueries; i++) {
        int in = i * 6;
        std::vector<uint32_t> queries = {targets[in], targets[in+2], targets[in+1], targets[in+3], targets[in+4], targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);

        CPUTimer rsQTimer;
        rsQTimer.start();
        rs.runRangeQueries(rsIndex, queryBuffer, 1);
        double t = double(rsQTimer.stop()) / 1000000.0;
        rsTotTime += t;

        rasterResults[i].resize(resultSizeUints);
        readUsingStagingBuf((char *)rasterResults[i].data(), resultSizeUints * sizeof(uint32_t), bufs->resBuffer, staging, vd);

        uint32_t count = 0;
        for(uint32_t val : rasterResults[i]) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << ": " << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Result Count: " << count << "\n";
    }

    std::cerr << ">>> Average Query Time: " << std::fixed << std::setprecision(3) << ((rsTotTime * 1000.0) / numQueries) << " ms\n";

    bufs->destroy();
    queryBuffer->destroy();

    std::cerr << "\nMode 32 (RasterScan2D Streaming) Complete.\n";

#else
    std::cerr << "\n--- [CompactScanIndex Streaming] ---\n";

    PCompactScanIndex compactIndex = std::make_shared<CompactScanIndex>(vd, ncols, scan);
    compactIndex->initialize();

    GPUMemoryTool::printGPUMemoryStatus(vd, "Before CompactScanIndex build");

    std::cerr << "\nBuilding Compact Index (streaming, taking min of 3 runs)...\n";
    double minBuildTime = 1e9;
    for(int k=0; k<3; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";
        CPUTimer buildTimer;
        buildTimer.start();
        compactIndex->buildIndexStreaming(points, npoints, minval.data(), maxval.data(), staging, BATCH_POINTS);
        double bt = double(buildTimer.stop()) / 1000000.0;
        if(bt < minBuildTime) minBuildTime = bt;
    }

    double buildTime = minBuildTime;
    std::cerr << "Compact Index build time: " << (buildTime * 1000.0) << " ms\n";
    GPUMemoryTool::printGPUMemoryStatus(vd, "After CompactScanIndex build");

    vkcore::PBuffer compactResultBuffer(new Buffer(vd));
    compactResultBuffer->create(resultSizeUints * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);

    std::vector<std::vector<uint32_t>> compactResults(numQueries);

    std::cerr << "\n--- Compact Index Query Performance ---\n";
    double compactTotTime = 0;
    for (int i = 0; i < numQueries; i++) {
        int in = i * 6;
        std::vector<uint32_t> queries = {targets[in], targets[in+1], targets[in+2], targets[in+3], targets[in+4], targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);

        CPUTimer qTimer;
        qTimer.start();
        compactIndex->runRangeQueries(queryBuffer, 1, compactResultBuffer);
        double t = double(qTimer.stop()) / 1000000.0;
        compactTotTime += t;

        compactResults[i].resize(resultSizeUints);
        readUsingStagingBuf((char *)compactResults[i].data(), resultSizeUints * sizeof(uint32_t), compactResultBuffer, staging, vd);

        uint32_t count = 0;
        for(uint32_t val : compactResults[i]) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << ": " << std::fixed << std::setprecision(3) << (t * 1000.0) << " ms, Result Count: " << count << "\n";
    }

    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(3) << ((compactTotTime * 1000.0) / numQueries) << " ms\n";

    compactResultBuffer->destroy();
    queryBuffer->destroy();

    std::cerr << "\nMode 32 (CompactScanIndex Streaming) Complete.\n";
#endif
}
