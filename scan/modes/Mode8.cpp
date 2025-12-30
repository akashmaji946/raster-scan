#include "RunModes.hpp"

void testDeleteByDataWithDistributions(PVkDevice vd, PBuffer staging) {
    std::cerr << "\n================================================================================\n";
    std::cerr << "  MODE 8: DELETE-BY-DATA PERFORMANCE WITH VARYING DISTRIBUTIONS\n";
    std::cerr << "================================================================================\n\n";
    
    // Configuration - same as mode=7
    const uint32_t NUM_BATCHES = 5;
    const uint32_t VALUE_RANGE = 1000000;  // Same as mode=7
    
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
        // Phase 0: Read encoded data and convert to column-major format
        // ========================================================================
        std::cerr << "[Phase 0] Reading encoded data from " << g_opfolder << datasets[dataId] << "...\n";
        
        // Read the raw data file directly (row-major format: x0,y0,z0, x1,y1,z1, ...)
        std::string dataFileName = g_opfolder + datasets[dataId] + "-data.bin";
        std::ifstream binfile(dataFileName, std::ios::binary | std::ios::ate);
        if (binfile.fail()) {
            std::cerr << "ERROR: Cannot open file: " << dataFileName << "\n";
            continue;
        }
        size_t sizeInBytes = binfile.tellg();
        binfile.close();
        
        const int ncols = 3;  // Always 3 columns for this test
        uint32_t npoints = uint32_t(sizeInBytes / (ncols * sizeof(uint32_t)));
        
        // Limit points to fit in memory (same logic as mode=7)
        uint32_t maxPoints = std::min(32000000u, (uint32_t)(MAX_PAGES * 5 / 10));
        if (npoints > maxPoints) {
            std::cerr << "[WARNING] Limiting points from " << npoints << " to " << maxPoints << "\n";
            npoints = maxPoints;
        }
        
        const uint32_t NUM_POINTS = npoints;
        const uint32_t BATCH_SIZE = NUM_POINTS / NUM_BATCHES;
        
        std::cerr << "[Config] MAX_PAGES:         " << MAX_PAGES << "\n";
        std::cerr << "[Config] Total points:      " << NUM_POINTS << "\n";
        std::cerr << "[Config] Batch size:        " << BATCH_SIZE << "\n";
        std::cerr << "[Config] Max pages needed:  " << (NUM_POINTS + NUM_BATCHES * BATCH_SIZE) << "\n\n";
        
        // Read row-major data from file
        std::vector<uint32_t> rowMajorData(NUM_POINTS * ncols);
        binfile.open(dataFileName, std::ios::binary);
        binfile.read((char*)rowMajorData.data(), NUM_POINTS * ncols * sizeof(uint32_t));
        binfile.close();
        
        // Convert to column-major format (same as mode=7 generates)
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
        
        std::cerr << "[Phase 0] Loaded and converted " << NUM_POINTS << " points to column-major format\n";
        
        // Create main points buffer (same as mode=7)
        PBuffer pointsBuffer(new Buffer(vd));
        pointsBuffer->create(NUM_POINTS * ncols * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            MemoryType::Internal);
        loadUsingStagingBuf((char*)points.data(), NUM_POINTS * ncols * sizeof(uint32_t), 
                            pointsBuffer, staging, vd, 0);
        
        // Create index (same as mode=7)
        PBufferCache bufs(new CommonBufferPool(vd));
        RasterScanIndexUpdate rsUpdate(vd, bufs, ncols);
        
        // Phase 1: Initial index build
        std::cerr << "\n[Phase 1] Building initial index with " << NUM_POINTS << " points...\n";
        
        CPUTimer buildTimer;
        buildTimer.start();
        
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
        
        double buildTime = double(buildTimer.stop()) / 1000000.0;
        std::cerr << "[Phase 1] Initial build time: " << (buildTime * 1000) << " ms\n";
        
        auto stats = rsUpdate.getAllocationStats(index);
        std::cerr << "[Phase 1] Initial allocated: " << stats.allocatedPages << " (expected: " << NUM_POINTS << ")\n";
        if (stats.allocatedPages != NUM_POINTS) {
            std::cerr << "[Phase 1] ERROR: Initial allocation mismatch!\n";
            pointsBuffer->destroy();
            continue;
        }
        
        // Phase 2: Create batches
        std::cerr << "\n[Phase 2] Creating " << NUM_BATCHES << " batches of " << BATCH_SIZE << " points each...\n";
        
        std::mt19937 rng(42);
        std::vector<uint32_t> shuffledIndices(NUM_POINTS);
        for (uint32_t i = 0; i < NUM_POINTS; i++) {
            shuffledIndices[i] = i;
        }
        std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), rng);
        
        PBuffer batchBuffer(new Buffer(vd));
        batchBuffer->create(BATCH_SIZE * ncols * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            MemoryType::Internal);
        
        std::vector<uint32_t> batchData(BATCH_SIZE * ncols);
        
        double totalDeleteTime = 0.0;
        double totalInsertTime = 0.0;
        std::vector<double> deleteTimesMs(NUM_BATCHES);
        std::vector<double> insertTimesMs(NUM_BATCHES);
        
        // Phase 3: Delete/Insert cycle
        std::cerr << "\n[Phase 3] Starting delete/insert cycle for " << NUM_BATCHES << " batches...\n";
        std::cerr << "--------------------------------------------------------------------------------\n";
        
        for (uint32_t batch = 0; batch < NUM_BATCHES; batch++) {
            std::cerr << "\n--- Batch " << (batch + 1) << "/" << NUM_BATCHES << " ---\n";
            
            uint32_t batchStart = batch * BATCH_SIZE;
            for (uint32_t i = 0; i < BATCH_SIZE; i++) {
                uint32_t idx = shuffledIndices[batchStart + i];
                batchData[i] = points[idx];
                batchData[BATCH_SIZE + i] = points[NUM_POINTS + idx];
                batchData[2 * BATCH_SIZE + i] = points[2 * NUM_POINTS + idx];
            }
            
            loadUsingStagingBuf((char*)batchData.data(), BATCH_SIZE * ncols * sizeof(uint32_t), 
                                batchBuffer, staging, vd, 0);
            
            uint32_t allocatedBefore = rsUpdate.getAllocationStats(index).allocatedPages;
            
            CPUTimer deleteTimer;
            deleteTimer.start();
            rsUpdate.deletePointsByData(index, batchBuffer, BATCH_SIZE);
            double deleteTime = double(deleteTimer.stop()) / 1000000.0;
            deleteTimesMs[batch] = deleteTime * 1000.0;
            totalDeleteTime += deleteTime;
            
            std::cerr << "  DELETE: " << BATCH_SIZE << " points marked invalid in " 
                      << deleteTimesMs[batch] << " ms\n";
            
            CPUTimer insertTimer;
            insertTimer.start();
            rsUpdate.insertPointsWithBitmapV2(index, batchBuffer, BATCH_SIZE, 0);
            double insertTime = double(insertTimer.stop()) / 1000000.0;
            insertTimesMs[batch] = insertTime * 1000.0;
            totalInsertTime += insertTime;
            
            uint32_t allocatedAfter = rsUpdate.getAllocationStats(index).allocatedPages;
            uint32_t newPagesAllocated = allocatedAfter - allocatedBefore;
            
            std::cerr << "  INSERT: " << newPagesAllocated << "/" << BATCH_SIZE 
                      << " new pages in " << insertTimesMs[batch] << " ms\n";
            
            uint32_t expectedAllocated = NUM_POINTS + (batch + 1) * BATCH_SIZE;
            std::cerr << "  ALLOCATED: " << allocatedAfter << " (expected: " << expectedAllocated << ")\n";
        }
        
        // Phase 4: State before compaction
        std::cerr << "\n================================================================================\n";
        std::cerr << "  BEFORE COMPACTION\n";
        std::cerr << "================================================================================\n";
        
        stats = rsUpdate.getAllocationStats(index);
        uint32_t expectedBeforeCompact = NUM_POINTS + NUM_BATCHES * BATCH_SIZE;
        std::cerr << "\n[State Before Compaction]\n";
        std::cerr << "  - Total allocated pages:  " << stats.allocatedPages << "\n";
        std::cerr << "  - Expected allocated:     " << expectedBeforeCompact << "\n";
        std::cerr << "  - Valid pages (data):     " << NUM_POINTS << "\n";
        std::cerr << "  - Invalid pages (deleted):" << (NUM_BATCHES * BATCH_SIZE) << "\n";
        
        // Phase 5: Run compaction
        std::cerr << "\n[Phase 5] Running compaction to clean up invalid pages...\n";
        
        CPUTimer compactTimer;
        compactTimer.start();
        rsUpdate.compactPages(index);
        double compactTime = double(compactTimer.stop()) / 1000000.0;
        
        std::cerr << "[Phase 5] Compaction time: " << (compactTime * 1000.0) << " ms\n";
        
        // Final Summary
        std::cerr << "\n================================================================================\n";
        std::cerr << "  FINAL SUMMARY (AFTER COMPACTION)\n";
        std::cerr << "================================================================================\n";
        
        stats = rsUpdate.getAllocationStats(index);
        std::cerr << "\n[Final State]\n";
        std::cerr << "  - Total allocated pages:  " << stats.allocatedPages << "\n";
        std::cerr << "  - Free pages:             " << stats.freePages << "\n";
        std::cerr << "  - Expected valid:         " << NUM_POINTS << "\n";
        std::cerr << "  - Final verification:     " << (stats.allocatedPages == NUM_POINTS ? "PASS" : "FAIL") << "\n";
        
        // Timing summary
        std::cerr << "\n[Timing Summary]\n";
        std::cerr << "  - Initial build time:     " << (buildTime * 1000.0) << " ms\n";
        std::cerr << "  - Total delete time:      " << (totalDeleteTime * 1000.0) << " ms\n";
        std::cerr << "  - Total insert time:      " << (totalInsertTime * 1000.0) << " ms\n";
        std::cerr << "  - Compaction time:        " << (compactTime * 1000.0) << " ms\n";
        std::cerr << "  - Avg delete per batch:   " << (totalDeleteTime * 1000.0 / NUM_BATCHES) << " ms\n";
        std::cerr << "  - Avg insert per batch:   " << (totalInsertTime * 1000.0 / NUM_BATCHES) << " ms\n";
        
        // Per-batch breakdown
        std::cerr << "\n[Per-Batch Timing (ms)]\n";
        std::cerr << "  Batch   Delete     Insert\n";
        std::cerr << "  -----   ------     ------\n";
        for (uint32_t i = 0; i < NUM_BATCHES; i++) {
            std::cerr << "  " << std::setw(5) << (i + 1) 
                      << "   " << std::setw(8) << std::fixed << std::setprecision(1) << deleteTimesMs[i]
                      << "   " << std::setw(8) << insertTimesMs[i] << "\n";
        }
        
        // Throughput
        double totalPoints = NUM_BATCHES * BATCH_SIZE;
        std::cerr << "\n[Throughput]\n";
        std::cerr << "  - Delete throughput: " << (totalPoints / totalDeleteTime / 1000000.0) << " M points/sec\n";
        std::cerr << "  - Insert throughput: " << (totalPoints / totalInsertTime / 1000000.0) << " M points/sec\n";
        
        std::cerr << "\n================================================================================\n";
        std::cerr << "Distribution " << datasets[dataId] << " COMPLETE!\n";
        std::cerr << "================================================================================\n\n";
        
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
    
    std::cerr << "\n================================================================================\n";
    std::cerr << "Delete-by-data with varying distributions test COMPLETE!\n";
    std::cerr << "================================================================================\n\n";
}
