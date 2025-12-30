#include "RunModes.hpp"

void testDeleteByDataWithDistributionsVarying(PVkDevice vd, PBuffer staging) {
    std::cerr << "\n================================================================================\n";
    std::cerr << "  MODE 11: DELETE-BY-DATA WITH VARYING DISTRIBUTIONS & VARYING BATCHES\n";
    std::cerr << "================================================================================\n\n";
    
    // Configuration - same as mode=8
    const uint32_t NUM_BATCHES = 10;
    const uint32_t VALUE_RANGE = 1000000;
    
    std::cerr << "[Config] Data folder:       " << g_opfolder << "\n";
    std::cerr << "[Config] Number of batches: " << NUM_BATCHES << "\n";
    std::cerr << "[Config] Columns:           " << g_dim << "\n";
    std::cerr << "[Config] Value range:       0 to " << VALUE_RANGE << "\n\n";
    
    // Test each distribution
    for (int dataId = 0; dataId < nDataset; dataId++) {
        std::cerr << "\n================================================================================\n";
        std::cerr << "  DISTRIBUTION: " << datasets[dataId] << " (" << (dataId + 1) << "/" << nDataset << ")\n";
        std::cerr << "================================================================================\n\n";
        
        // ========================================================================
        // Phase 0: Read encoded data
        // ========================================================================
        std::cerr << "[Phase 0] Reading encoded data from " << g_opfolder << datasets[dataId] << "...\n";
        
        std::string dataFileName = g_opfolder + datasets[dataId] + "-data.bin";
        std::ifstream binfile(dataFileName, std::ios::binary | std::ios::ate);
        if (binfile.fail()) {
            std::cerr << "ERROR: Cannot open file: " << dataFileName << "\n";
            continue;
        }
        size_t sizeInBytes = binfile.tellg();
        binfile.close();
        
        const int ncols = 3;
        uint32_t npoints = uint32_t(sizeInBytes / (ncols * sizeof(uint32_t)));
        
        uint32_t maxPoints = std::min(32000000u, (uint32_t)(MAX_PAGES * 5 / 10));
        if (npoints > maxPoints) {
            std::cerr << "[WARNING] Limiting points from " << npoints << " to " << maxPoints << "\n";
            npoints = maxPoints;
        }
        
        const uint32_t NUM_POINTS = npoints;
        
        // Generate varying batch sizes
        std::vector<uint32_t> batchSizes(NUM_BATCHES);
        std::vector<uint32_t> batchOffsets(NUM_BATCHES);
        uint32_t maxBatchSize = 0;
        {
            std::mt19937 splitRng(42 + dataId);
            std::vector<uint32_t> splits;
            splits.push_back(0);
            splits.push_back(NUM_POINTS);
            
            std::uniform_int_distribution<uint32_t> splitDist(1, NUM_POINTS - 1);
            for(uint32_t i=0; i < NUM_BATCHES-1; i++) {
                splits.push_back(splitDist(splitRng));
            }
            std::sort(splits.begin(), splits.end());
            
            for(uint32_t i=0; i < NUM_BATCHES; i++) {
                batchSizes[i] = splits[i+1] - splits[i];
                batchOffsets[i] = splits[i];
                if (batchSizes[i] > maxBatchSize) maxBatchSize = batchSizes[i];
            }
        }
        
        std::cerr << "[Config] MAX_PAGES:         " << MAX_PAGES << "\n";
        std::cerr << "[Config] Total points:      " << NUM_POINTS << "\n";
        std::cerr << "[Config] Max Batch size:    " << maxBatchSize << "\n";
        std::cerr << "[Config] Max pages needed:  " << (NUM_POINTS + maxBatchSize) << " (approx)\n\n";
        
        std::vector<uint32_t> rowMajorData(NUM_POINTS * ncols);
        binfile.open(dataFileName, std::ios::binary);
        binfile.read((char*)rowMajorData.data(), NUM_POINTS * ncols * sizeof(uint32_t));
        binfile.close();
        
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
            
            if (x < minVal[0]) minVal[0] = x;
            if (x > maxVal[0]) maxVal[0] = x;
            if (y < minVal[1]) minVal[1] = y;
            if (y > maxVal[1]) maxVal[1] = y;
            if (z < minVal[2]) minVal[2] = z;
            if (z > maxVal[2]) maxVal[2] = z;
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
        
        // Phase 1: Initial Build
        std::cerr << "\n[Phase 1] Building initial index...\n";
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
        
        // Phase 2: Create batches
        std::cerr << "\n[Phase 2] Creating batches...\n";
        std::mt19937 rng(42);
        std::vector<uint32_t> shuffledIndices(NUM_POINTS);
        std::iota(shuffledIndices.begin(), shuffledIndices.end(), 0);
        std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), rng);
        
        PBuffer batchBuffer(new Buffer(vd));
        batchBuffer->create(maxBatchSize * ncols * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            MemoryType::Internal);
        
        std::vector<uint32_t> batchData(maxBatchSize * ncols);
        
        double totalDeleteTime = 0.0, totalInsertTime = 0.0;
        
        // Phase 3: Delete/Insert
        std::cerr << "\n[Phase 3] Starting delete/insert cycle...\n";
        for (uint32_t batch = 0; batch < NUM_BATCHES; batch++) {
            uint32_t currentBatchSize = batchSizes[batch];
            if (currentBatchSize == 0) continue;
            
            uint32_t batchStart = batchOffsets[batch];
            for (uint32_t i = 0; i < currentBatchSize; i++) {
                uint32_t idx = shuffledIndices[batchStart + i];
                batchData[i] = points[idx];
                batchData[currentBatchSize + i] = points[NUM_POINTS + idx];
                batchData[2 * currentBatchSize + i] = points[2 * NUM_POINTS + idx];
            }
            
            loadUsingStagingBuf((char*)batchData.data(), currentBatchSize * ncols * sizeof(uint32_t), 
                                batchBuffer, staging, vd, 0);
            
            CPUTimer deleteTimer;
            deleteTimer.start();
            rsUpdate.deletePointsByData(index, batchBuffer, currentBatchSize);
            double deleteTime = double(deleteTimer.stop()) / 1000000.0;
            totalDeleteTime += deleteTime;
            
            CPUTimer insertTimer;
            insertTimer.start();
            rsUpdate.insertPointsWithBitmapV2(index, batchBuffer, currentBatchSize, 0);
            double insertTime = double(insertTimer.stop()) / 1000000.0;
            totalInsertTime += insertTime;
            
            std::cerr << "  Batch " << (batch+1) << " (Size: " << currentBatchSize << "): "
                      << "Del: " << (deleteTime*1000) << " ms (" << (deleteTime*1000000.0/currentBatchSize) << " us/pt), "
                      << "Ins: " << (insertTime*1000) << " ms (" << (insertTime*1000000.0/currentBatchSize) << " us/pt)\n";
        }
        
        // Phase 5: Compaction
        std::cerr << "\n[Phase 5] Compaction...\n";
        CPUTimer compactTimer;
        compactTimer.start();
        rsUpdate.compactPages(index);
        double compactTime = double(compactTimer.stop()) / 1000000.0;
        
        auto stats = rsUpdate.getAllocationStats(index);
        std::cerr << "[Final] Allocated: " << stats.allocatedPages << " (Expected: " << NUM_POINTS << ")\n";
        std::cerr << "Status: " << (stats.allocatedPages == NUM_POINTS ? "PASS" : "FAIL") << "\n";
        
        // Cleanup
        batchBuffer->destroy();
        batchBuffer.reset();
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
    std::cerr << "Mode 11 COMPLETE.\n";
}
