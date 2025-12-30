#include "RunModes.hpp"

void testRobustnessWithReverseCycles(PVkDevice vd, PBuffer staging) {
    std::cerr << "\n================================================================================\n";
    std::cerr << "  MODE 9: ROBUSTNESS TEST - DELETE/INSERT + COMPACT (RUN N times)\n";
    std::cerr << "================================================================================\n\n";
    
    // Configuration - same as mode=8
    const uint32_t NUM_BATCHES = 10;
    const uint32_t NUM_ROUNDS = 10;  // Run the cycle N times
    const int ncols = 3;
    
    std::cerr << "[Config] Data folder:       " << g_opfolder << "\n";
    std::cerr << "[Config] Number of batches: " << NUM_BATCHES << "\n";
    std::cerr << "[Config] Number of rounds:  " << NUM_ROUNDS << "\n";
    std::cerr << "[Config] Columns:           " << g_dim << "\n\n";
    
    // Use uniform distribution for robustness test (fastest)
    
    // ========================================================================
    // Phase 0: Read encoded data and convert to column-major format
    // ========================================================================
    
    // Use the single distribution loaded (g_opfolder + datasets[0] usually, but let's use the first available)
    // Actually, mode=9 usually takes a -g argument or relies on defaults.
    // The previous implementation used 'dataFileName' which was likely setup in main or hardcoded.
    // We'll mimic Mode 8's loading but just for one dataset.
    
    // Construct filename assuming standard structure if not provided
    std::string datasetName = (nDataset > 0) ? datasets[1] : "uniform"; 
    // Wait, the user provided folder structure suggests encodedData folder.
    // Mode 8 iterates datasets. We'll pick "uniform" if available, or just the first one.
    // Let's use the logic from previous Mode 9 implementation which read row-major data.
    
    std::cerr << "================================================================================\n";
    std::cerr << "  DISTRIBUTION: " << datasetName << "\n";
    std::cerr << "================================================================================\n\n";
    
    std::cerr << "[Phase 0] Reading encoded data from " << g_opfolder << datasetName << "...\n";
    
    std::string dataFileName = g_opfolder + datasetName + "-data.bin";
    std::ifstream binfile(dataFileName, std::ios::binary | std::ios::ate);
    if (binfile.fail()) {
        std::cerr << "ERROR: Cannot open file: " << dataFileName << "\n";
        return;
    }
    size_t sizeInBytes = binfile.tellg();
    binfile.close();
    
    uint32_t npoints = uint32_t(sizeInBytes / (ncols * sizeof(uint32_t)));
    
    // Limit points to fit in memory
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
    
    // IMPORTANT: Add 1 to maxVal to match Mode 8 logic (exclusive upper bound)
    maxVal[0]++; maxVal[1]++; maxVal[2]++;
    
    std::cerr << "[Phase 0] Loaded and converted " << NUM_POINTS << " points to column-major format\n";
    std::cerr << "[Phase 0] Data range X: " << minVal[0] << " to " << maxVal[0] << "\n";
    std::cerr << "[Phase 0] Data range Y: " << minVal[1] << " to " << maxVal[1] << "\n";
    std::cerr << "[Phase 0] Data range Z: " << minVal[2] << " to " << maxVal[2] << "\n";
    
    PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(NUM_POINTS * ncols * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)points.data(), NUM_POINTS * ncols * sizeof(uint32_t), 
                        pointsBuffer, staging, vd, 0);
    
    PBufferCache bufs(new CommonBufferPool(vd));
    RasterScanIndexUpdate rsUpdate(vd, bufs, ncols);
    
    // ========================================================================
    // Phase 1: Initial index build
    // ========================================================================
    std::cerr << "\n[Phase 1] Building initial index with " << NUM_POINTS << " points...\n";
    
    CPUTimer buildTimer;
    buildTimer.start();
    
    rsUpdate.pageAlloc.reset(new PageAllocator(vd, MAX_PAGES));
    PLinkedListIndex index(new LinkedListIndex(vd, NUM_POINTS, rsUpdate.pageAlloc));
    index->minVal[0] = minVal[0]; index->minVal[1] = minVal[1]; index->minVal[2] = minVal[2];
    index->maxVal[0] = maxVal[0]; index->maxVal[1] = maxVal[1]; index->maxVal[2] = maxVal[2];
    
    // Use ceiling division for bin range (same as Mode 8)
    for (int i = 0; i < ncols; i++) {
        index->binRange[i] = (index->maxVal[i] - index->minVal[i] + INDEX_RESOLUTION - 1) / INDEX_RESOLUTION;
        if (index->binRange[i] == 0) index->binRange[i] = 1;
    }
    
    rsUpdate.initializeBitmapWithAllocation(0);
    rsUpdate.insertPointsWithBitmapV2(index, pointsBuffer, NUM_POINTS, 0);
    
    double buildTime = double(buildTimer.stop()) / 1000000.0;
    std::cerr << "[Phase 1] Initial build time: " << (buildTime * 1000) << " ms\n";
    
    // Verify initial allocation
    auto stats = rsUpdate.getAllocationStats(index);
    std::cerr << "[Phase 1] Initial allocated: " << stats.allocatedPages << " (expected: " << NUM_POINTS << ")\n";
    if (stats.allocatedPages != NUM_POINTS) {
        std::cerr << "[Phase 1] ERROR: Initial allocation mismatch!\n";
        return;
    }
    
    // Reuse batch buffer
    PBuffer batchBuffer(new Buffer(vd));
    batchBuffer->create(BATCH_SIZE * ncols * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
        
    std::vector<uint32_t> batchData(BATCH_SIZE * ncols);
    
    // Prepare random batch indices
    std::vector<uint32_t> shuffledIndices(NUM_POINTS);
    std::iota(shuffledIndices.begin(), shuffledIndices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // ========================================================================
    // Robustness Loops
    // ========================================================================
    for (uint32_t round = 0; round < NUM_ROUNDS; round++) {
        std::cerr << "\n================================================================================\n";
        std::cerr << "  ROUND " << (round + 1) << "/" << NUM_ROUNDS << ": DELETE/INSERT CYCLES\n";
        std::cerr << "================================================================================\n";
        
        // Shuffle indices for this round to have different batches
        std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), gen);
        
        // Forward Pass: Delete then Insert for each batch
        std::cerr << "\n[Forward Pass] Processing " << NUM_BATCHES << " batches (Delete -> Insert)...\n";
        
        for (uint32_t batch = 0; batch < NUM_BATCHES; batch++) {
            std::cerr << "\n--- Batch " << (batch + 1) << "/" << NUM_BATCHES << " ---\n";
            
            // Prepare batch data
            uint32_t batchStart = batch * BATCH_SIZE;
            for (uint32_t i = 0; i < BATCH_SIZE; i++) {
                uint32_t idx = shuffledIndices[batchStart + i];
                batchData[i] = points[idx];
                batchData[BATCH_SIZE + i] = points[NUM_POINTS + idx];
                batchData[2 * BATCH_SIZE + i] = points[2 * NUM_POINTS + idx];
            }
            loadUsingStagingBuf((char*)batchData.data(), BATCH_SIZE * ncols * sizeof(uint32_t), 
                                batchBuffer, staging, vd, 0);
            
            // --- DELETE ---
            CPUTimer deleteTimer;
            deleteTimer.start();
            rsUpdate.deletePointsByData(index, batchBuffer, BATCH_SIZE);
            double deleteTime = double(deleteTimer.stop()) / 1000000.0;
            
            // --- INSERT ---
            CPUTimer insertTimer;
            insertTimer.start();
            rsUpdate.insertPointsWithBitmapV2(index, batchBuffer, BATCH_SIZE, 0);
            double insertTime = double(insertTimer.stop()) / 1000000.0;
            
            std::cerr << "  Batch " << (batch+1) << ": Del " << (deleteTime*1000) << "ms, Ins " << (insertTime*1000) << "ms\n";
            
            // Verify allocation grew
            // Expected: allocated = previous + BATCH_SIZE (since deleted are not freed yet)
            // But we do compaction at end of round, so within round it grows.
        }
        
        // Check stats before compaction
        stats = rsUpdate.getAllocationStats(index);
        std::cerr << "\n[Before Compaction] Allocated: " << stats.allocatedPages 
                  << " (Expected approx: " << (NUM_POINTS + NUM_POINTS) << ")\n"; // Initial + Inserted (Deleted are just invalid)
        
        // --- COMPACTION ---
        std::cerr << "[Compaction] Running compaction...\n";
        CPUTimer compactTimer;
        compactTimer.start();
        rsUpdate.compactPages(index);
        double compactTime = double(compactTimer.stop()) / 1000000.0;
        std::cerr << "  Compaction time: " << (compactTime * 1000.0) << " ms\n";
        
        // Verify stats after compaction
        stats = rsUpdate.getAllocationStats(index);
        std::cerr << "[After Compaction] Allocated: " << stats.allocatedPages << " (Expected: " << NUM_POINTS << ")\n";
        
        if (stats.allocatedPages != NUM_POINTS) {
            std::cerr << "ERROR: Compaction failed to restore correct page count!\n";
            // return; // Don't abort, try next round to see if it recovers or gets worse
        } else {
            std::cerr << "Status: PASS\n";
        }
    }
    
    std::cerr << "\n================================================================================\n";
    std::cerr << "Robustness test COMPLETE!\n";
    std::cerr << "================================================================================\n\n";
    
    // Cleanup
    pointsBuffer->destroy();
    pointsBuffer.reset();
    batchBuffer->destroy();
    batchBuffer.reset();
    
    index->headPtrBuffer->destroy();
    index.reset();
    rsUpdate.freeBitmap.reset();
    rsUpdate.pageAlloc->pageBuffer->destroy();
    rsUpdate.pageAlloc->allocCounterBuffer->destroy();
    rsUpdate.pageAlloc.reset();
    bufs.reset();
    vd->device->waitIdle();
}
