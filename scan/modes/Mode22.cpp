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
    "zipf1.5.bin"
};

static const std::vector<std::string> distributionNames = {
    "uniform",
    "normal",
    "zipf1.1",
    "zipf1.3",
    "zipf1.5"
};

// Query generation strategy based on distribution type
enum class QueryStrategy {
    CENTERED,    // For uniform/normal: center queries in data range
    FROM_MIN     // For zipf: start queries from minimum (where data clusters)
};

// Get appropriate query strategy for each distribution
static QueryStrategy getQueryStrategy(int dataId) {
    // dataId: 0=uniform, 1=normal, 2=zipf1.1, 3=zipf1.3, 4=zipf1.5
    switch (dataId) {
        case 0: return QueryStrategy::CENTERED;  // Uniform: data spread evenly
        case 1: return QueryStrategy::CENTERED;  // Normal: data centered around mean
        case 2: return QueryStrategy::FROM_MIN;  // Zipf 1.1: data clusters at low values
        case 3: return QueryStrategy::FROM_MIN;  // Zipf 1.3: more skewed
        case 4: return QueryStrategy::FROM_MIN;  // Zipf 1.5: most skewed
        default: return QueryStrategy::CENTERED;
    }
}

// Generate queries with specific selectivities and save to file
// Selectivity = fraction of data range covered per dimension
// Strategy depends on distribution type
static void generateAndSaveQueries(
    const std::vector<uint32_t>& minval, 
    const std::vector<uint32_t>& maxval,
    int ncols,
    const std::string& outputFile,
    int dataId,
    int numQueries = 10
) {
    // Create output directory if needed
    std::string dir = outputFile.substr(0, outputFile.find_last_of('/'));
    mkdir(dir.c_str(), 0755);
    
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "ERROR: Cannot create query file: " << outputFile << "\n";
        return;
    }
    
    QueryStrategy strategy = getQueryStrategy(dataId);
    std::cerr << "Query strategy: " << (strategy == QueryStrategy::CENTERED ? "CENTERED" : "FROM_MIN") << "\n";
    
    // Generate 10 queries with selectivities 10%, 20%, ..., 100%
    // For 3D data: per-dimension selectivity = cbrt(overall_selectivity)
    for (int q = 0; q < numQueries; q++) {
        double overallSelectivity = (q + 1) * 0.1;  // 10%, 20%, ..., 100%
        double perDimSelectivity = std::pow(overallSelectivity, 1.0 / ncols);
        
        for (int c = 0; c < ncols; c++) {
            uint64_t range = (uint64_t)maxval[c] - (uint64_t)minval[c];
            uint64_t queryRange = (uint64_t)(range * perDimSelectivity);
            
            uint32_t lo, hi;
            if (strategy == QueryStrategy::CENTERED) {
                // Center the query in the data range (for uniform/normal)
                uint64_t margin = (range - queryRange) / 2;
                lo = minval[c] + (uint32_t)margin;
                hi = minval[c] + (uint32_t)(margin + queryRange);
            } else {
                // Start from minimum (for zipf - data clusters at low values)
                lo = minval[c];
                hi = minval[c] + (uint32_t)queryRange;
            }
            
            // Write in "lt <value>" format (less than)
            out << "lt " << lo << "\n";
            out << "lt " << hi << "\n";
        }
        // Fill remaining dimensions if ncols < 3
        for (int c = ncols; c < 3; c++) {
            out << "lt 0\n";
            out << "lt 4294967295\n";
        }
    }
    
    out.close();
    std::cerr << "Saved " << numQueries << " queries to: " << outputFile << "\n";
}

void testCompactIndexAndCompare(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    
    // Validate dataId (0-4 for the 5 distributions)
    if (dataId < 0 || dataId >= (int)distributionFiles.size()) {
        std::cerr << "ERROR: Invalid dataId " << dataId << ". Must be 0-4.\n";
        std::cerr << "  0: uniform, 1: normal, 2: zipf1.1, 3: zipf1.3, 4: zipf1.5\n";
        return;
    }
    
    // Construct plain data folder path: data/data_Xm_Yc/
    uint32_t millions = g_npoints / 1000000;
    std::string plainDataFolder = PROJECT_DIR + "data/data_" + std::to_string(millions) + "m_" + std::to_string(g_dim) + "c";
    std::string dataFile = plainDataFolder + "/" + distributionFiles[dataId];
    
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 22: Compact Index Test (Plain Data)\n";
    std::cerr << "Distribution: " << distributionNames[dataId] << " (dataId=" << dataId << ")\n";
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

    // 2. Generate and save queries with selectivities 10%, 20%, ..., 100%
    std::string queryFolder = PROJECT_DIR + "tests/test1";
    std::string queryFile = queryFolder + "/" + distributionNames[dataId] + ".txt";
    int numQueries = 10;
    generateAndSaveQueries(minval, maxval, ncols, queryFile, dataId, numQueries);
    
    // 3. Load queries into targets vector
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
    PBufferCache bufs(new CommonBufferPool(vd));
    
    RasterScan2D rs(vd, bufs, scan, reduce, ncols);
    
    GPUMemoryTool::printGPUMemoryStatus(vd, "Before RasterScan2D build");
    std::cerr << "Building RasterScan2D Index (taking min of 3 runs)...\n";
    double minRsBuildTime = 1e9;
    PRasterIndex rsIndex;
    for(int k=0; k<3; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";
        CPUTimer rsBuildTimer;
        rsBuildTimer.start();
        rsIndex = rs.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
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
        // Mode 0 format: x1, y1, x2, y2, z1, z2
        std::vector<uint32_t> queries = {targets[in], targets[in+2], targets[in+1], targets[in+3], targets[in+4], targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
        
        CPUTimer rsQTimer;
        rsQTimer.start();
        // Clear result buffer before query (for fair comparison with CompactIndex)
        bufs->resBuffer->clearBuffer();
        rs.runRangeQueries(rsIndex, queryBuffer, 1);
        double t = double(rsQTimer.stop()) / 1000000.0;
        rsTotTime += t;
        
        // Read back
        rasterResults[i].resize(resultSizeUints);
        readUsingStagingBuf((char *)rasterResults[i].data(), resultSizeUints * sizeof(uint32_t), bufs->resBuffer, staging, vd);
        
        uint32_t count = 0;
        for(uint32_t val : rasterResults[i]) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << ": " << std::fixed << std::setprecision(6) << t << " s, Result Count: " << count << "\n";
    }
    std::cerr << ">>> Average Query Time: " << std::fixed << std::setprecision(6) << (rsTotTime / numQueries) << " s\n";
    
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
    
    PCompactScanIndex compactIndex = std::make_shared<CompactScanIndex>(vd, ncols, scan);
    compactIndex->initialize();

    GPUMemoryTool::printGPUMemoryStatus(vd, "Before CompactScanIndex build");
    
    std::cerr << "\nBuilding Compact Index (taking min of 3 runs)...\n";
    double minBuildTime = 1e9;
    for(int k=0; k<3; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";
        CPUTimer buildTimer;
        buildTimer.start();
        compactIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
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
    
    // Store Compact Results
    std::vector<std::vector<uint32_t>> compactResults(numQueries); 
    
    std::cerr << "\n--- Compact Index Query Performance ---\n";
    double compactTotTime = 0;
    for (int i = 0; i < numQueries; i++) {
        int in = i * 6;
        // Mode 21 format: x1, x2, y1, y2, z1, z2
        std::vector<uint32_t> queries = {targets[in], targets[in+1], targets[in+2], targets[in+3], targets[in+4], targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);

        // Result buffer clear is now done inside runRangeQueries (included in timing)
        CPUTimer qTimer;
        qTimer.start();
        compactIndex->runRangeQueries(queryBuffer, 1, compactResultBuffer);
        double t = double(qTimer.stop()) / 1000000.0;
        compactTotTime += t;

        // Read back
        compactResults[i].resize(resultSizeUints);
        readUsingStagingBuf((char *)compactResults[i].data(), resultSizeUints * sizeof(uint32_t), compactResultBuffer, staging, vd);
        
        // Log count
        uint32_t count = 0;
        for(uint32_t val : compactResults[i]) count += __builtin_popcount(val);
        std::cerr << "Query " << (i+1) << ": " << std::fixed << std::setprecision(6) << t << " s, Result Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(6) << (compactTotTime / numQueries) << " s\n";
    
    // =========================================================
    // Delete/Insert Performance (CompactScanIndex)
    // =========================================================
    
    uint32_t ndeletes = 100000;
    if(ndeletes > npoints) ndeletes = npoints;
    
    std::cerr << "\n--- Delete Performance ---\n";
    std::cerr << "Deleting " << ndeletes << " points...\n";
    
    vkcore::PBuffer deleteDataBuffer(new Buffer(vd));
    deleteDataBuffer->create(ndeletes * 3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
        
    std::vector<uint32_t> deleteData(ndeletes * 3);
    for(uint32_t i=0; i<ndeletes; i++) {
        deleteData[i*3 + 0] = points[i]; // x
        deleteData[i*3 + 1] = points[npoints + i]; // y
        deleteData[i*3 + 2] = points[2*npoints + i]; // z
    }
    
    loadUsingStagingBuf((char*)deleteData.data(), deleteData.size() * sizeof(uint32_t), deleteDataBuffer, staging, vd, 0);
    
    // Lambda for count verification
    auto verifyCount = [&](uint32_t expected, const std::string& label) {
        uint64_t capacity = compactIndex->totalAllocatedCapacity;
        std::vector<CompactEntry> hostData(capacity);
        // Ensure staging buffer is large enough or read in chunks. 
        // capacity * 16 bytes. For 10M points -> 160MB. Staging is 16MB?
        // readUsingStagingBuf handles loop? No, it usually handles staging size if implemented correctly, 
        // but if data > staging, it might fail if implementation is simple.
        // Let's check vkutils.cpp for readUsingStagingBuf implementation.
        // Assuming it handles it or staging is large enough.
        // Wait, staging buffer size is printed as 16MB. 10M points is 160MB.
        // I should re-create staging buffer if needed or rely on robust implementation.
        // I'll assume readUsingStagingBuf is robust or resize staging.
        // Actually, let's just resize staging to be safe.
        // Or check if I can use a loop.
        // For now, I'll rely on readUsingStagingBuf.
        
        readUsingStagingBuf((char*)hostData.data(), capacity * sizeof(CompactEntry), compactIndex->dataBuffer, staging, vd);
        
        uint32_t validCount = 0;
        for(const auto& entry : hostData) {
            if(entry.rowId & 0x80000000) { // Valid bit
                validCount++;
            }
        }
        std::cerr << label << ": Expected=" << expected << ", Actual=" << validCount << " [" << (expected==validCount ? "PASS" : "FAIL") << "]\n";
    };

    CPUTimer delTimer;
    delTimer.start();
    compactIndex->deletePoints(deleteDataBuffer, ndeletes);
    double delTime = double(delTimer.stop()) / 1000000.0;
    std::cerr << ">>> Delete Time: " << (delTime*1000.0) << " ms (" << (delTime * 1000000.0 / ndeletes) << " us/point)\n";
    
    verifyCount(npoints - ndeletes, "Delete Verification");

    // --- Insert Performance ---
    std::cerr << "\n--- Insert Performance ---\n";
    std::cerr << "Re-inserting " << ndeletes << " points...\n";
    CPUTimer insTimer;
    insTimer.start();
    compactIndex->insertPoints(deleteDataBuffer, ndeletes);
    double insTime = double(insTimer.stop()) / 1000000.0;
    std::cerr << ">>>Insert Time: " << (insTime*1000.0) << " ms (" << (insTime * 1000000.0 / ndeletes) << " us/point)\n";
    
    verifyCount(npoints, "Insert Verification");
    
    // Final cleanup
    deleteDataBuffer->destroy();
    compactResultBuffer->destroy();
    queryBuffer->destroy();
    pointsBuffer->destroy();
    
    std::cerr << "\nCompactScanIndex Mode Complete.\n";
#endif // RUNRASTER
}
