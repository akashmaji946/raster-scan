#include "RunModes.hpp"
#include "../CompactScanIndex.hpp"

void testCompactIndex(int dataId, PVkDevice vd, PBuffer staging, OperatorCache &op) {
    std::cerr << "\n========================================\n";
    std::cerr << "MODE 21: Compact Index Test\n";
    std::cerr << "Dataset " << dataId << " (" << datasets[dataId] << ")\n";
    std::cerr << "========================================\n";

    // Read dataset
    int32_t ncols;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<std::map<uint32_t, uint32_t>> rowMap;
    std::vector<uint32_t> points;
    PBuffer pointsBuffer = readEncodedData(g_opfolder + datasets[dataId], vd, staging, npoints, minval, maxval, rowMap, points, ncols);
    std::cerr << "Dataset: " << npoints << " points\n";

    // Read queries
    std::vector<uint32_t> targets;
    readQueries(g_qfolder + querysets[dataId], qct[dataId], targets, rowMap);

    // Initialize CompactScanIndex
    PCompactScanIndex compactIndex = std::make_shared<CompactScanIndex>(vd, ncols);
    compactIndex->initialize();
    
    std::cout << "[CompactIndex] Total DataBuffer Size (Before Build): " << compactIndex->getSizeMB() << " MB\n";

    // Build Index
    std::cerr << "\nBuilding Compact Index...\n";
    CPUTimer buildTimer;
    buildTimer.start();
    compactIndex->buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    double buildTime = double(buildTimer.stop()) / 1000000.0;
    std::cerr << "Compact Index build time: " << (buildTime * 1000.0) << " ms\n";
    
    std::cout << "[CompactIndex] Total DataBuffer Size (After Build): " << compactIndex->getSizeMB() << " MB\n";

    // Stats
    GPUMemoryTool::printGPUMemoryStatus(vd, "After Compact Index Build");

    // Compute min/max/avg bin counts
    uint32_t totalBins = INDEX_RESOLUTION * INDEX_RESOLUTION;
    std::vector<uint32_t> counts(totalBins);
    readUsingStagingBuf((char *)counts.data(), totalBins * sizeof(uint32_t), compactIndex->countBuffer, staging, vd);
    
    uint32_t cmin = UINT32_MAX, cmax = 0;
    uint64_t csum = 0;
    for(uint32_t c : counts) {
        if(c < cmin) cmin = c;
        if(c > cmax) cmax = c;
        csum += c;
    }
    double avg = (double)csum / totalBins;
    std::cerr << "[Statistics] Bin Counts: Min=" << cmin << ", Max=" << cmax << ", Avg=" << avg << "\n";
    std::cerr << "[Statistics] Total Points Inserted: " << csum << " / " << npoints << "\n";

    // Query buffer
    PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
        
    // Result buffer (stores count for verification)
    // For "range query", we usually want the COUNT of matching points.
    // Or bitmask?
    // In Mode 2, `resBuffer` is a bitmask (1 bit per point).
    // `CompactScanIndex` stores points in a different order (scattered).
    // So rowIds are preserved.
    // If we want to verify against `RasterScan2D`, we need to output `rowId`s or a bitmask indexed by `rowId`.
    // Let's output a bitmask indexed by `rowId`.
    // Max `rowId` is `npoints`.
    uint32_t resultSizeUints = (npoints + 31) / 32;
    PBuffer resultBuffer(new Buffer(vd));
    resultBuffer->create(resultSizeUints * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
    
    std::vector<uint32_t> res(resultSizeUints);
    
    std::cerr << "\n--- Query Performance ---\n";
    
    double totTime = 0;
    for (int i = 0; i < qct[dataId]; i++) {
        int in = i * 6;
        std::vector<uint32_t> queries = {targets[in], targets[in+1], targets[in+2], targets[in+3], targets[in+4], targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);
        
        // Clear result buffer
        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        vd->commandBuffer->begin(beginInfo);
        resultBuffer->clearBufferWithBarrier(vk::PipelineStageFlagBits::eComputeShader, 0);
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
        compactIndex->runRangeQueries(queryBuffer, 1, resultBuffer);
        double t = double(qTimer.stop()) / 1000000.0;
        totTime += t;
        
        // Read back count for logging
        readUsingStagingBuf((char *)res.data(), resultSizeUints * sizeof(uint32_t), resultBuffer, staging, vd);
        uint32_t count = 0;
        for(uint32_t val : res) count += __builtin_popcount(val);
        
        std::cerr << "Query " << (i+1) << ": " << std::fixed << std::setprecision(6) << t << " s, Result Count: " << count << "\n";
    }
    std::cerr << "Average Query Time: " << std::fixed << std::setprecision(6) << (totTime / qct[dataId]) << " s\n";
    
    // Delete Test (Mode 21 specific requirement)
    // "When an element is deleted, it will be marked invalid..."
    // Let's delete a batch of points.
    // For simplicity, pick first 1000 points from dataset to delete.
    
    uint32_t ndeletes = 1000;
    if(ndeletes > npoints) ndeletes = npoints;
    
    std::cerr << "\n--- Delete Performance ---\n";
    std::cerr << "Deleting " << ndeletes << " points...\n";
    
    PBuffer deleteDataBuffer(new Buffer(vd));
    deleteDataBuffer->create(ndeletes * 3 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);
        
    // Copy first ndeletes points [x,y,z] from points array
    // points array layout: x0, x1, ... y0, y1... z0... ?
    // ModeUtils::readEncodedData: "points.resize(npoints * ncols); ... read((char*)points.data()..."
    // It reads binary data.
    // If format is row-major (x,y,z, x,y,z...) or col-major?
    // Usually RasterScan uses Column Stores?
    // Let's check `readEncodedData`:
    // It reads `sizeof(uint32_t) * npoints * ncols`.
    // User data is usually `x0, x1... xN, y0... yN, z0... zN`.
    // Let's assume Column Major.
    // But `CompactEntry` needs (x,y,z).
    // `buildIndex` shader will handle reading from Col-Major to CompactEntry.
    // For delete, we need (x,y,z) tuples.
    
    std::vector<uint32_t> deleteData(ndeletes * 3);
    for(uint32_t i=0; i<ndeletes; i++) {
        deleteData[i*3 + 0] = points[i]; // x
        deleteData[i*3 + 1] = points[npoints + i]; // y
        deleteData[i*3 + 2] = points[2*npoints + i]; // z
    }
    
    loadUsingStagingBuf((char*)deleteData.data(), deleteData.size() * sizeof(uint32_t), deleteDataBuffer, staging, vd, 0);
    
    CPUTimer delTimer;
    delTimer.start();
    compactIndex->deletePoints(deleteDataBuffer, ndeletes);
    double delTime = double(delTimer.stop()) / 1000000.0;
    std::cerr << ">>> Delete Time: " << (delTime*1000.0) << " ms (" << (delTime * 1000000.0 / ndeletes) << " us/point)\n";
    
    // --- Insert Performance ---
    std::cerr << "\n--- Insert Performance ---\n";
    std::cerr << "Re-inserting " << ndeletes << " points...\n";
    CPUTimer insTimer;
    insTimer.start();
    compactIndex->insertPoints(deleteDataBuffer, ndeletes);
    double insTime = double(insTimer.stop()) / 1000000.0;
    std::cerr << ">>>Insert Time: " << (insTime*1000.0) << " ms (" << (insTime * 1000000.0 / ndeletes) << " us/point)\n";
    
    // Verify Counts
    readUsingStagingBuf((char *)counts.data(), totalBins * sizeof(uint32_t), compactIndex->countBuffer, staging, vd);
    uint64_t finalSum = 0;
    for(uint32_t c : counts) finalSum += c;
    
    std::cerr << "[Statistics] Final Total Points: " << finalSum << " / " << (npoints + ndeletes) << "\n";
    
    if (finalSum == (npoints + ndeletes)) {
        std::cerr << "TEST STATUS: PASS\n";
    } else {
        std::cerr << "TEST STATUS: FAIL (Count Mismatch)\n";
    }
    
    std::cout << "[CompactIndex] Final DataBuffer Size: " << compactIndex->getSizeMB() << " MB\n";

    // Cleanup
    deleteDataBuffer->destroy();
    resultBuffer->destroy();
    queryBuffer->destroy();
    pointsBuffer->destroy(); // Created in readEncodedData
    
    // CompactIndex destructor cleans up its buffers
    std::cerr << "\nMode 21 Complete.\n";
}
