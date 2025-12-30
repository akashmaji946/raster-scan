#include "RunModes.hpp"

void testCPUVerification(PVkDevice vd, PBuffer staging) {
    std::cerr << "\n================================================================================\n";
    std::cerr << "  MODE 10: CPU VERIFICATION TEST (Accuracy Check)\n";
    std::cerr << "================================================================================\n\n";

    // Configuration
    const uint32_t NUM_BATCHES = 10;
    const uint32_t NUM_ROUNDS = 3;
    const int ncols = 3;
    
    // Use a smaller dataset for CPU verification to be fast
    std::string datasetName = "uniform"; // Default
    
    std::cerr << "[Config] Data folder:       " << g_opfolder << "\n";
    std::cerr << "[Config] Dataset:           " << datasetName << "\n";
    std::cerr << "[Config] Number of batches: " << NUM_BATCHES << "\n";
    std::cerr << "[Config] Number of rounds:  " << NUM_ROUNDS << "\n";

    // Read data
    std::string dataFileName = g_opfolder + datasetName + "-data.bin";
    std::ifstream binfile(dataFileName, std::ios::binary | std::ios::ate);
    if (binfile.fail()) {
        std::cerr << "ERROR: Cannot open file: " << dataFileName << "\n";
        return;
    }
    size_t sizeInBytes = binfile.tellg();
    binfile.close();
    
    uint32_t npoints = uint32_t(sizeInBytes / (ncols * sizeof(uint32_t)));
    
    // // Limit to 1M for CPU performance
    // if (npoints > 10000000) {
    //     std::cerr << "[Config] Limiting points to 10,000,000 for CPU verification speed.\n";
    //     npoints = 10000000;
    // }
    
    const uint32_t NUM_POINTS = npoints;
    const uint32_t BATCH_SIZE = NUM_POINTS / NUM_BATCHES;
    
    std::vector<uint32_t> rowMajorData(NUM_POINTS * ncols);
    binfile.open(dataFileName, std::ios::binary);
    binfile.read((char*)rowMajorData.data(), NUM_POINTS * ncols * sizeof(uint32_t));
    binfile.close();
    
    // Prepare CPU points
    std::vector<Point3D> cpuPoints;
    cpuPoints.reserve(NUM_POINTS);
    
    std::vector<uint32_t> points(NUM_POINTS * ncols);
    uint32_t minVal[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint32_t maxVal[3] = {0, 0, 0};
    
    for (uint32_t i = 0; i < NUM_POINTS; i++) {
        uint32_t x = rowMajorData[i * ncols + 0];
        uint32_t y = rowMajorData[i * ncols + 1];
        uint32_t z = rowMajorData[i * ncols + 2];
        
        points[i] = x;
        points[NUM_POINTS + i] = y;
        points[2 * NUM_POINTS + i] = z;
        
        cpuPoints.push_back({x, y, z});
        
        if (x < minVal[0]) minVal[0] = x;
        if (x > maxVal[0]) maxVal[0] = x;
        if (y < minVal[1]) minVal[1] = y;
        if (y > maxVal[1]) maxVal[1] = y;
        if (z < minVal[2]) minVal[2] = z;
        if (z > maxVal[2]) maxVal[2] = z;
    }
    maxVal[0]++; maxVal[1]++; maxVal[2]++;
    
    std::cerr << "[Phase 0] Loaded " << NUM_POINTS << " points\n";
    
    // Initial GPU Build
    PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(NUM_POINTS * ncols * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)points.data(), NUM_POINTS * ncols * sizeof(uint32_t), 
                        pointsBuffer, staging, vd, 0);
    
    PBufferCache bufs(new CommonBufferPool(vd));
    RasterScanIndexUpdate rsUpdate(vd, bufs, ncols);
    
    std::cerr << "\n[Phase 1] Building initial index on GPU...\n";
    rsUpdate.pageAlloc.reset(new PageAllocator(vd, MAX_PAGES));
    PLinkedListIndex index(new LinkedListIndex(vd, NUM_POINTS, rsUpdate.pageAlloc));
    index->minVal[0] = minVal[0]; index->minVal[1] = minVal[1]; index->minVal[2] = minVal[2];
    index->maxVal[0] = maxVal[0]; index->maxVal[1] = maxVal[1]; index->maxVal[2] = maxVal[2];
    
    for (int i = 0; i < ncols; i++) {
        index->binRange[i] = (index->maxVal[i] - index->minVal[i] + INDEX_RESOLUTION - 1) / INDEX_RESOLUTION;
        if (index->binRange[i] == 0) index->binRange[i] = 1;
    }
    
    rsUpdate.initializeBitmapWithAllocation(0);
    rsUpdate.insertPointsWithBitmapV2(index, pointsBuffer, NUM_POINTS, 0);
    
    // Prepare for cycles
    std::vector<uint32_t> shuffledIndices(NUM_POINTS);
    std::iota(shuffledIndices.begin(), shuffledIndices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Initial sort of CPU points for set operations
    std::sort(cpuPoints.begin(), cpuPoints.end());
    
    // Create readback buffer (Host Visible) - 32MB initial size
    PBuffer readbackBuffer(new Buffer(vd));
    readbackBuffer->create(32 * 1024 * 1024, vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadOnly);

    // Create Result Buffer for verification (Internal) - Linear array of valid points
    PBuffer resultBuffer(new Buffer(vd));
    resultBuffer->create(MAX_PAGES * 16, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, MemoryType::Internal);

    // Create Count Buffer (Internal)
    PBuffer countBuffer(new Buffer(vd));
    countBuffer->create(sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, MemoryType::Internal);

    // Create Count Staging Buffer (Host Visible)
    PBuffer countStagingBuffer(new Buffer(vd));
    countStagingBuffer->create(sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadOnly);

    // Verification Lambda
    auto verify = [&](const std::vector<Point3D>& groundTruth, const char* label) {
        // Run verifyIndex (traverses linked list, populates resultBuffer and countBuffer)
        rsUpdate.verifyIndex(index, resultBuffer, countBuffer);
        
        // Copy count to staging
        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        vd->commandBuffer->begin(beginInfo);
        
        vk::BufferCopy countCopy(0, 0, sizeof(uint32_t));
        vd->commandBuffer->copyBuffer(countBuffer->buf, countStagingBuffer->buf, countCopy);
        
        vd->commandBuffer->end();
        
        vk::SubmitInfo submitInfo;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &vd->commandBuffer.get();
        vk::Fence fence = vd->device->createFence(vk::FenceCreateInfo());
        vd->submit(submitInfo, fence, false);
        vd->waitForFences(fence, VK_TRUE, UINT64_MAX);
        vd->device->destroyFence(fence);
        
        // Read count
        uint32_t validCount = 0;
        countStagingBuffer->readData((char*)&validCount, sizeof(uint32_t));
        
        if (validCount != groundTruth.size()) {
             std::cerr << "  [FAIL] " << label << ": Count mismatch! Expected: " << groundTruth.size() << ", Actual: " << validCount << "\n";
        }
        
        if (validCount == 0) {
            if (groundTruth.empty()) {
                std::cerr << "  [PASS] " << label << ": Verified (0 points).\n";
            } else {
                std::cerr << "  [FAIL] " << label << ": No points found on GPU.\n";
            }
            return;
        }

        // Copy resultBuffer to readbackBuffer
        vk::DeviceSize copySize = validCount * 16; // uvec4 = 16 bytes
        
        if (copySize > readbackBuffer->size) {
            std::cerr << "  [INFO] Resizing readback buffer to " << copySize << " bytes\n";
            readbackBuffer->destroy();
            readbackBuffer->create(copySize * 3 / 2, vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadOnly);
        }
        
        // Manual copy
        vk::CommandBufferBeginInfo beginInfo2(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        vd->commandBuffer->begin(beginInfo2);
        
        vk::BufferCopy copyRegion(0, 0, copySize);
        vd->commandBuffer->copyBuffer(resultBuffer->buf, readbackBuffer->buf, copyRegion);
        
        vd->commandBuffer->end();
        
        vk::SubmitInfo submitInfo2;
        submitInfo2.commandBufferCount = 1;
        submitInfo2.pCommandBuffers = &vd->commandBuffer.get();
        
        vk::Fence fence2 = vd->device->createFence(vk::FenceCreateInfo());
        vd->submit(submitInfo2, fence2, false);
        vd->waitForFences(fence2, VK_TRUE, UINT64_MAX);
        vd->device->destroyFence(fence2);
        
        // Read back data
        std::vector<uint32_t> rawData(validCount * 4);
        readbackBuffer->readData((char*)rawData.data(), copySize);
        
        std::vector<Point3D> gpuPoints;
        gpuPoints.reserve(validCount);
        
        for (uint32_t i = 0; i < validCount; i++) {
            uint32_t offset = i * 4; 
            uint32_t x = rawData[offset + 0];
            uint32_t y = rawData[offset + 1];
            uint32_t z = rawData[offset + 2];
            gpuPoints.push_back({x, y, z});
        }
        
        std::sort(gpuPoints.begin(), gpuPoints.end());
        
        bool exactMatch = true;
        if (groundTruth.size() != gpuPoints.size()) {
            exactMatch = false;
        } else {
            for (size_t i = 0; i < groundTruth.size(); i++) {
                if (!(groundTruth[i] == gpuPoints[i])) {
                    exactMatch = false;
                    std::cerr << "  [FAIL] " << label << ": Mismatch at index " << i << "\n";
                    std::cerr << "    CPU: (" << groundTruth[i].x << ", " << groundTruth[i].y << ", " << groundTruth[i].z << ")\n";
                    std::cerr << "    GPU: (" << gpuPoints[i].x << ", " << gpuPoints[i].y << ", " << gpuPoints[i].z << ")\n";
                    break;
                }
            }
        }
        
        if (exactMatch) std::cerr << "  [PASS] " << label << ": Verified (" << groundTruth.size() << " points).\n";
        else std::cerr << "  [FAIL] " << label << ": Verification Failed.\n";
    };

    // Verify initial state
    verify(cpuPoints, "Initial Build");

    // ========================================================================
    // Cycles
    // ========================================================================
    for (uint32_t round = 0; round < NUM_ROUNDS; round++) {
        std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), gen);
        
        std::cerr << "\n--- Round " << (round + 1) << "/" << NUM_ROUNDS << " ---\n";
        
        for (uint32_t batch = 0; batch < NUM_BATCHES; batch++) {
            std::cerr << "  Batch " << (batch + 1) << "/" << NUM_BATCHES << ": ";
            
            // Identify points for this batch
            std::vector<Point3D> batchPoints;
            batchPoints.reserve(BATCH_SIZE);
            
            // Prepare GPU buffer data
            std::vector<uint32_t> batchData(BATCH_SIZE * ncols);
            for (uint32_t i = 0; i < BATCH_SIZE; i++) {
                uint32_t srcIdx = shuffledIndices[batch * BATCH_SIZE + i];
                uint32_t x = points[srcIdx];
                uint32_t y = points[NUM_POINTS + srcIdx];
                uint32_t z = points[2 * NUM_POINTS + srcIdx];
                
                batchPoints.push_back({x, y, z});
                
                batchData[i] = x;
                batchData[BATCH_SIZE + i] = y;
                batchData[2 * BATCH_SIZE + i] = z;
            }
            
            // Sort batch points for set operations
            std::sort(batchPoints.begin(), batchPoints.end());
            
            // --- DELETE ---
            PBuffer batchBuffer(new Buffer(vd));
            batchBuffer->create(BATCH_SIZE * ncols * sizeof(uint32_t),
                vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                MemoryType::Internal);
            loadUsingStagingBuf((char*)batchData.data(), BATCH_SIZE * ncols * sizeof(uint32_t),
                                batchBuffer, staging, vd, 0);
                                
            rsUpdate.deletePointsByData(index, batchBuffer, BATCH_SIZE);
            
            // CPU Update: Remove batch
            std::vector<Point3D> nextCpuPoints;
            nextCpuPoints.reserve(cpuPoints.size() - batchPoints.size());
            std::set_difference(cpuPoints.begin(), cpuPoints.end(),
                                batchPoints.begin(), batchPoints.end(),
                                std::back_inserter(nextCpuPoints));
            cpuPoints = std::move(nextCpuPoints);
            
            // Verify Delete
            std::cerr << "Delete -> ";
            verify(cpuPoints, "After Delete");
            
            // --- INSERT ---
            rsUpdate.insertPointsWithBitmapV2(index, batchBuffer, BATCH_SIZE, 0);
            
            // CPU Update: Add batch
            nextCpuPoints.clear();
            nextCpuPoints.reserve(cpuPoints.size() + batchPoints.size());
            std::merge(cpuPoints.begin(), cpuPoints.end(),
                           batchPoints.begin(), batchPoints.end(),
                           std::back_inserter(nextCpuPoints));
            cpuPoints = std::move(nextCpuPoints);
            
            // Verify Insert
            std::cerr << "        Insert -> ";
            verify(cpuPoints, "After Insert");
            
            batchBuffer->destroy();
            batchBuffer.reset();
        }
        // do compaction after each round
        std::cout << "Compacting...\n";
        rsUpdate.compactPages(index);
    }
    
    std::cerr << "\nCPU Verification Test COMPLETE!\n";
    
    // Cleanup
    readbackBuffer->destroy();
    readbackBuffer.reset();
    resultBuffer->destroy();
    resultBuffer.reset();
    countBuffer->destroy();
    countBuffer.reset();
    countStagingBuffer->destroy();
    countStagingBuffer.reset();
    pointsBuffer->destroy();
    pointsBuffer.reset();
    index->headPtrBuffer->destroy();
    index.reset();
    rsUpdate.freeBitmap.reset();
    rsUpdate.pageAlloc->pageBuffer->destroy();
    rsUpdate.pageAlloc->allocCounterBuffer->destroy();
    rsUpdate.pageAlloc.reset();
    bufs.reset();
    vd->device->waitIdle();
}
