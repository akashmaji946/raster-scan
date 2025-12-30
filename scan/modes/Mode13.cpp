#include "RunModes.hpp"

void testCPUVerificationVarying(PVkDevice vd, PBuffer staging) {
    std::cerr << "\n================================================================================\n";
    std::cerr << "  MODE 13: CPU VERIFICATION WITH VARYING BATCHES\n";
    std::cerr << "================================================================================\n\n";

    const uint32_t NUM_BATCHES = 10;
    const uint32_t NUM_ROUNDS = 2;
    const int ncols = 3;
    std::string datasetName = "zipf1.1"; 
    
    std::string dataFileName = g_opfolder + datasetName + "-data.bin";
    std::ifstream binfile(dataFileName, std::ios::binary | std::ios::ate);
    if (binfile.fail()) {
        std::cerr << "ERROR: Cannot open file: " << dataFileName << "\n";
        return;
    }
    size_t sizeInBytes = binfile.tellg();
    binfile.close();
    
    uint32_t npoints = uint32_t(sizeInBytes / (ncols * sizeof(uint32_t)));
    if (npoints > 10000000) npoints = 10000000;
    const uint32_t NUM_POINTS = npoints;
    
    // Varying batch sizes
    std::vector<uint32_t> batchSizes(NUM_BATCHES);
    std::vector<uint32_t> batchOffsets(NUM_BATCHES);
    {
        std::mt19937 splitRng(42);
        std::vector<uint32_t> splits;
        splits.push_back(0);
        splits.push_back(NUM_POINTS);
        std::uniform_int_distribution<uint32_t> splitDist(1, NUM_POINTS - 1);
        for(uint32_t i=0; i < NUM_BATCHES-1; i++) splits.push_back(splitDist(splitRng));
        std::sort(splits.begin(), splits.end());
        for(uint32_t i=0; i < NUM_BATCHES; i++) {
            batchSizes[i] = splits[i+1] - splits[i];
            batchOffsets[i] = splits[i];
        }
    }
    
    std::vector<uint32_t> rowMajorData(NUM_POINTS * ncols);
    binfile.open(dataFileName, std::ios::binary);
    binfile.read((char*)rowMajorData.data(), NUM_POINTS * ncols * sizeof(uint32_t));
    binfile.close();
    
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
        if (x < minVal[0]) minVal[0] = x; if (x > maxVal[0]) maxVal[0] = x;
        if (y < minVal[1]) minVal[1] = y; if (y > maxVal[1]) maxVal[1] = y;
        if (z < minVal[2]) minVal[2] = z; if (z > maxVal[2]) maxVal[2] = z;
    }
    maxVal[0]++; maxVal[1]++; maxVal[2]++;
    
    PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(NUM_POINTS * ncols * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)points.data(), NUM_POINTS * ncols * sizeof(uint32_t), 
                        pointsBuffer, staging, vd, 0);
    
    PBufferCache bufs(new CommonBufferPool(vd));
    RasterScanIndexUpdate rsUpdate(vd, bufs, ncols);
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
    
    std::vector<uint32_t> shuffledIndices(NUM_POINTS);
    std::iota(shuffledIndices.begin(), shuffledIndices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::sort(cpuPoints.begin(), cpuPoints.end());
    
    // Verification buffers
    PBuffer readbackBuffer(new Buffer(vd));
    readbackBuffer->create(32 * 1024 * 1024, vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadOnly);
    PBuffer resultBuffer(new Buffer(vd));
    resultBuffer->create(MAX_PAGES * 16, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, MemoryType::Internal);
    PBuffer countBuffer(new Buffer(vd));
    countBuffer->create(sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, MemoryType::Internal);
    PBuffer countStagingBuffer(new Buffer(vd));
    countStagingBuffer->create(sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadOnly);
    
    auto verify = [&](const std::vector<Point3D>& groundTruth, const char* label) {
        rsUpdate.verifyIndex(index, resultBuffer, countBuffer);
        
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
        
        uint32_t validCount = 0;
        countStagingBuffer->readData((char*)&validCount, sizeof(uint32_t));
        
        if (validCount != groundTruth.size()) {
             std::cerr << "  [FAIL] " << label << ": Count mismatch! Expected: " << groundTruth.size() << ", Actual: " << validCount << "\n";
        }
        
        if (validCount == 0) {
            if (groundTruth.empty()) std::cerr << "  [PASS] " << label << ": Verified (0 points).\n";
            else std::cerr << "  [FAIL] " << label << ": No points found on GPU.\n";
            return;
        }

        vk::DeviceSize copySize = validCount * 16;
        if (copySize > readbackBuffer->size) {
            readbackBuffer->destroy();
            readbackBuffer->create(copySize * 3 / 2, vk::BufferUsageFlagBits::eTransferDst, MemoryType::ReadOnly);
        }
        
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
        
        std::vector<uint32_t> rawData(validCount * 4);
        readbackBuffer->readData((char*)rawData.data(), copySize);
        
        std::vector<Point3D> gpuPoints;
        gpuPoints.reserve(validCount);
        for (uint32_t i = 0; i < validCount; i++) {
            uint32_t offset = i * 4; 
            gpuPoints.push_back({rawData[offset + 0], rawData[offset + 1], rawData[offset + 2]});
        }
        std::sort(gpuPoints.begin(), gpuPoints.end());
        
        bool exactMatch = true;
        if (groundTruth.size() != gpuPoints.size()) exactMatch = false;
        else {
            for (size_t i = 0; i < groundTruth.size(); i++) {
                if (!(groundTruth[i] == gpuPoints[i])) {
                    exactMatch = false;
                    break;
                }
            }
        }
        if (exactMatch) std::cerr << "  [PASS] " << label << ": Verified (" << groundTruth.size() << " points).\n";
        else std::cerr << "  [FAIL] " << label << ": Verification Failed.\n";
    };
    
    verify(cpuPoints, "Initial Build");
    
    for (uint32_t round = 0; round < NUM_ROUNDS; round++) {
        std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), gen);
        std::cerr << "\n--- Round " << (round + 1) << "/" << NUM_ROUNDS << " ---\n";
        
        for (uint32_t batch = 0; batch < NUM_BATCHES; batch++) {
            uint32_t currentBatchSize = batchSizes[batch];
            if (currentBatchSize == 0) continue;
            
            std::cerr << "  Batch " << (batch + 1) << "/" << NUM_BATCHES << " (Size " << currentBatchSize << "):\n";
            
            std::vector<Point3D> batchPoints;
            batchPoints.reserve(currentBatchSize);
            std::vector<uint32_t> batchData(currentBatchSize * ncols);
            uint32_t batchStart = batchOffsets[batch];
            
            for (uint32_t i = 0; i < currentBatchSize; i++) {
                uint32_t srcIdx = shuffledIndices[batchStart + i];
                uint32_t x = points[srcIdx];
                uint32_t y = points[NUM_POINTS + srcIdx];
                uint32_t z = points[2 * NUM_POINTS + srcIdx];
                batchPoints.push_back({x, y, z});
                batchData[i] = x;
                batchData[currentBatchSize + i] = y;
                batchData[2 * currentBatchSize + i] = z;
            }
            std::sort(batchPoints.begin(), batchPoints.end());
            
            PBuffer batchBuffer(new Buffer(vd));
            batchBuffer->create(currentBatchSize * ncols * sizeof(uint32_t),
                vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                MemoryType::Internal);
            loadUsingStagingBuf((char*)batchData.data(), currentBatchSize * ncols * sizeof(uint32_t),
                                batchBuffer, staging, vd, 0);
            
            CPUTimer deleteTimer;
            deleteTimer.start();
            rsUpdate.deletePointsByData(index, batchBuffer, currentBatchSize);
            double deleteTime = double(deleteTimer.stop()) / 1000000.0;
            
            std::vector<Point3D> nextCpuPoints;
            nextCpuPoints.reserve(cpuPoints.size() - batchPoints.size());
            std::set_difference(cpuPoints.begin(), cpuPoints.end(),
                                batchPoints.begin(), batchPoints.end(),
                                std::back_inserter(nextCpuPoints));
            cpuPoints = std::move(nextCpuPoints);
            
            std::cerr << "    Del: " << (deleteTime*1000) << " ms (" << (deleteTime*1000000.0/currentBatchSize) << " us/pt) -> ";
            verify(cpuPoints, "After Delete");
            
            CPUTimer insertTimer;
            insertTimer.start();
            rsUpdate.insertPointsWithBitmapV2(index, batchBuffer, currentBatchSize, 0);
            double insertTime = double(insertTimer.stop()) / 1000000.0;
            
            nextCpuPoints.clear();
            nextCpuPoints.reserve(cpuPoints.size() + batchPoints.size());
            std::merge(cpuPoints.begin(), cpuPoints.end(),
                           batchPoints.begin(), batchPoints.end(),
                           std::back_inserter(nextCpuPoints));
            cpuPoints = std::move(nextCpuPoints);
            
            std::cerr << "    Ins: " << (insertTime*1000) << " ms (" << (insertTime*1000000.0/currentBatchSize) << " us/pt) -> ";
            verify(cpuPoints, "After Insert");
            
            batchBuffer->destroy();
            batchBuffer.reset();
        }
        std::cout << "Compacting...\n";
        rsUpdate.compactPages(index);
    }
    
    std::cerr << "\nMode 13 COMPLETE!\n";
    readbackBuffer->destroy(); readbackBuffer.reset();
    resultBuffer->destroy(); resultBuffer.reset();
    countBuffer->destroy(); countBuffer.reset();
    countStagingBuffer->destroy(); countStagingBuffer.reset();
    pointsBuffer->destroy(); pointsBuffer.reset();
    index->headPtrBuffer->destroy(); index.reset();
    rsUpdate.freeBitmap.reset();
    rsUpdate.pageAlloc->pageBuffer->destroy();
    rsUpdate.pageAlloc->allocCounterBuffer->destroy();
    rsUpdate.pageAlloc.reset();
    bufs.reset();
    vd->device->waitIdle();
}
