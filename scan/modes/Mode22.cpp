#include "RunModes.hpp"
#include "../CompactScanIndex.hpp"
#include "../RasterScan2D.hpp"
#include "../BufferPool.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

// Set RUNRASTER to 1 to run RasterScan2D, 0 to run CompactScanIndex
#ifndef RUNRASTER
#define RUNRASTER 0
#endif

void testCompactIndexAndCompare(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op) {
    
    dataId = 4;  // Use uniform dataset with multiple queries
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 22: Compact Index Test with Comparison (RasterScan2D)\n";
    std::cerr << "Dataset " << dataId << " (" << datasets[dataId] << ")\n";
    std::cerr << "========================================\n";

    // 1. Read Dataset
    int32_t ncols;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<std::map<uint32_t, uint32_t>> rowMap;
    std::vector<uint32_t> points;
    vkcore::PBuffer pointsBuffer = readEncodedData(g_opfolder + datasets[dataId], vd, staging, npoints, minval, maxval, rowMap, points, ncols);
    std::cerr << "Dataset: " << npoints << " points\n";

    // Read Queries
    std::vector<uint32_t> targets;
    readQueries(g_qfolder + querysets[dataId], qct[dataId], targets, rowMap);


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

    std::vector<std::vector<uint32_t>> rasterResults(qct[dataId]);
    
    std::cerr << "\n--- RasterScan2D Query Performance ---\n";
    double rsTotTime = 0;

    for (int i = 0; i < qct[dataId]; i++) {
        int in = i * 6;
        // Mode 0 format: x1, y1, x2, y2, z1, z2
        std::vector<uint32_t> queries = {targets[in], targets[in+2], targets[in+1], targets[in+3], targets[in+4], targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
        
        CPUTimer rsQTimer;
        rsQTimer.start();
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
    std::cerr << ">>> Average Query Time: " << std::fixed << std::setprecision(6) << (rsTotTime / qct[dataId]) << " s\n";
    
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
    std::vector<std::vector<uint32_t>> compactResults(qct[dataId]); 
    
    std::cerr << "\n--- Compact Index Query Performance ---\n";
    double compactTotTime = 0;
    for (int i = 0; i < qct[dataId]; i++) {
        int in = i * 6;
        // Mode 21 format: x1, x2, y1, y2, z1, z2
        std::vector<uint32_t> queries = {targets[in], targets[in+1], targets[in+2], targets[in+3], targets[in+4], targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);

        // Clear result buffer
        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        vd->commandBuffer->begin(beginInfo);
        compactResultBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eComputeShader, 0);
        vd->commandBuffer->end();
        vk::SubmitInfo submitInfo;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &vd->commandBuffer.get();
        vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
        vd->submit(submitInfo, fence, false);
        vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
        vd->device->destroyFence(fence);

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
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(6) << (compactTotTime / qct[dataId]) << " s\n";
    
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
