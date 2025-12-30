#include "RunModes.hpp"

void testDeleteByData(PVkDevice vd, PBuffer staging) {
    std::cerr << "\n================================================================================\n";
    std::cerr << "  MODE 7: DELETE-BY-DATA BATCH CYCLE TEST\n";
    std::cerr << "================================================================================\n\n";
    
    // Configuration
    // We need: NUM_POINTS + NUM_BATCHES * BATCH_SIZE <= MAX_PAGES
    // With BATCH_SIZE = NUM_POINTS / NUM_BATCHES, total pages = NUM_POINTS * 2
    // So NUM_POINTS <= MAX_PAGES / 2, with some margin for safety
    const uint32_t NUM_BATCHES = 10;         
    const uint32_t NUM_POINTS = std::min(10000000u, (uint32_t)(MAX_PAGES * 5 / 10));  
    const uint32_t BATCH_SIZE = NUM_POINTS / NUM_BATCHES;
    const uint32_t VALUE_RANGE = 1000000;    // Random values in range [0, 1M)
    
    std::cerr << "[Config] MAX_PAGES:         " << MAX_PAGES << "\n";
    std::cerr << "[Config] Total points:      " << NUM_POINTS << "\n";
    std::cerr << "[Config] Number of batches: " << NUM_BATCHES << "\n";
    std::cerr << "[Config] Batch size:        " << BATCH_SIZE << "\n";
    std::cerr << "[Config] Value range:       0 to " << VALUE_RANGE << "\n";
    std::cerr << "[Config] Max pages needed:  " << (NUM_POINTS + NUM_BATCHES * BATCH_SIZE) << "\n\n";
    
    // Random number generator
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, VALUE_RANGE - 1);
    
    // ========================================================================
    // Phase 0: Generate all random data points
    // ========================================================================
    std::cerr << "[Phase 0] Generating " << NUM_POINTS << " random points...\n";
    CPUTimer genTimer;
    genTimer.start();
    
    std::vector<uint32_t> points(NUM_POINTS * 3);
    for (uint32_t i = 0; i < NUM_POINTS; i++) {
        points[i] = dist(rng);                    // x
        points[NUM_POINTS + i] = dist(rng);       // y
        points[2 * NUM_POINTS + i] = dist(rng);   // z
    }
    
    double genTime = double(genTimer.stop()) / 1000000.0;
    std::cerr << "[Phase 0] Data generation time: " << genTime << " secs\n";
    
    // Create main points buffer for initial insert
    PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(NUM_POINTS * 3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)points.data(), NUM_POINTS * 3 * sizeof(uint32_t), 
                        pointsBuffer, staging, vd, 0);
    
    // Min/max values
    uint32_t minVal[3] = {0, 0, 0};
    uint32_t maxVal[3] = {VALUE_RANGE, VALUE_RANGE, VALUE_RANGE};
    
    // Create index
    PBufferCache bufs(new CommonBufferPool(vd));
    RasterScanIndexUpdate rsUpdate(vd, bufs, 3);
    
    // ========================================================================
    // Phase 1: Initial index build with all NUM_POINTS
    // ========================================================================
    std::cerr << "\n[Phase 1] Building initial index with " << NUM_POINTS << " points...\n";
    
    CPUTimer buildTimer;
    buildTimer.start();
    
    // Create PageAllocator
    rsUpdate.pageAlloc.reset(new PageAllocator(vd, MAX_PAGES));
    
    // Create LinkedListIndex
    PLinkedListIndex index(new LinkedListIndex(vd, NUM_POINTS, rsUpdate.pageAlloc));
    index->minVal[0] = minVal[0]; index->minVal[1] = minVal[1]; index->minVal[2] = minVal[2];
    index->maxVal[0] = maxVal[0]; index->maxVal[1] = maxVal[1]; index->maxVal[2] = maxVal[2];
    
    // Calculate bin range
    for (int i = 0; i < 3; i++) {
        index->binRange[i] = (index->maxVal[i] - index->minVal[i] + INDEX_RESOLUTION - 1) / INDEX_RESOLUTION;
        if (index->binRange[i] == 0) index->binRange[i] = 1;
    }
    
    // Initialize bitmap with all pages free
    rsUpdate.initializeBitmapWithAllocation(0);
    
    // Insert all points using V2 pipeline
    rsUpdate.insertPointsWithBitmapV2(index, pointsBuffer, NUM_POINTS, 0);
    
    double buildTime = double(buildTimer.stop()) / 1000000.0;
    std::cerr << "[Phase 1] Initial build time: " << buildTime << " secs\n";
    
    // Verify initial state
    auto stats = rsUpdate.getAllocationStats(index);
    std::cerr << "[Phase 1] Initial allocated: " << stats.allocatedPages << " (expected: " << NUM_POINTS << ")\n";
    if (stats.allocatedPages != NUM_POINTS) {
        std::cerr << "[Phase 1] ERROR: Initial allocation mismatch!\n";
        return;
    }
    std::cerr << "[Phase 1] Initial state VERIFIED.\n";
    
    // ========================================================================
    // Phase 2: Create 10 batches (shuffle and partition)
    // ========================================================================
    std::cerr << "\n[Phase 2] Creating " << NUM_BATCHES << " batches of " << BATCH_SIZE << " points each...\n";
    
    // Shuffle indices to create random batches
    std::vector<uint32_t> shuffledIndices(NUM_POINTS);
    for (uint32_t i = 0; i < NUM_POINTS; i++) {
        shuffledIndices[i] = i;
    }
    std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), rng);
    
    // Create batch buffers (reusable buffer for each batch)
    PBuffer batchBuffer(new Buffer(vd));
    batchBuffer->create(BATCH_SIZE * 3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    // Storage for batch data (CPU side)
    std::vector<uint32_t> batchData(BATCH_SIZE * 3);
    
    // Timing accumulators
    double totalDeleteTime = 0.0;
    double totalInsertTime = 0.0;
    std::vector<double> deleteTimesMs(NUM_BATCHES);
    std::vector<double> insertTimesMs(NUM_BATCHES);
    
    // ========================================================================
    // Phase 3: Delete/Insert cycle for each batch
    // ========================================================================
    std::cerr << "\n[Phase 3] Starting delete/insert cycle for " << NUM_BATCHES << " batches...\n";
    std::cerr << "--------------------------------------------------------------------------------\n";
    
    for (uint32_t batch = 0; batch < NUM_BATCHES; batch++) {
        std::cerr << "\n--- Batch " << (batch + 1) << "/" << NUM_BATCHES << " ---\n";
        
        // Prepare batch data from shuffled indices
        uint32_t batchStart = batch * BATCH_SIZE;
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            uint32_t idx = shuffledIndices[batchStart + i];
            batchData[i] = points[idx];                        // x
            batchData[BATCH_SIZE + i] = points[NUM_POINTS + idx];  // y
            batchData[2 * BATCH_SIZE + i] = points[2 * NUM_POINTS + idx];  // z
        }
        
        // Upload batch data to GPU
        loadUsingStagingBuf((char*)batchData.data(), BATCH_SIZE * 3 * sizeof(uint32_t), 
                            batchBuffer, staging, vd, 0);
        
        // Get allocated pages before this batch (from bitmap)
        uint32_t allocatedBefore = rsUpdate.getAllocationStats(index).allocatedPages;
        
        // --- DELETE ---
        // Only marks pages as invalid, does NOT modify bitmap or linked list structure
        CPUTimer deleteTimer;
        deleteTimer.start();
        rsUpdate.deletePointsByData(index, batchBuffer, BATCH_SIZE);
        double deleteTime = double(deleteTimer.stop()) / 1000000.0;
        deleteTimesMs[batch] = deleteTime * 1000.0;
        totalDeleteTime += deleteTime;
        
        std::cerr << "  DELETE: " << BATCH_SIZE << " points marked invalid in " 
                  << deleteTimesMs[batch] << " ms\n";
        
        // --- INSERT ---
        // Allocates NEW pages from bitmap (doesn't reuse deleted pages)
        CPUTimer insertTimer;
        insertTimer.start();
        rsUpdate.insertPointsWithBitmapV2(index, batchBuffer, BATCH_SIZE, 0);
        double insertTime = double(insertTimer.stop()) / 1000000.0;
        insertTimesMs[batch] = insertTime * 1000.0;
        totalInsertTime += insertTime;
        
        // Verify insert - should have allocated BATCH_SIZE new pages
        uint32_t allocatedAfter = rsUpdate.getAllocationStats(index).allocatedPages;
        uint32_t newPagesAllocated = allocatedAfter - allocatedBefore;
        
        std::cerr << "  INSERT: " << newPagesAllocated << "/" << BATCH_SIZE 
                  << " new pages in " << insertTimesMs[batch] << " ms\n";
        
        // After each batch: allocated = initial + batch_num * BATCH_SIZE
        // (because we allocate new pages for inserts, old invalid pages stay)
        uint32_t expectedAllocated = NUM_POINTS + (batch + 1) * BATCH_SIZE;
        std::cerr << "  ALLOCATED: " << allocatedAfter << " (expected: " << expectedAllocated << ")\n";
    }
    
    // ========================================================================
    // Phase 4: State before compaction
    // ========================================================================
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
    
    // ========================================================================
    // Phase 5: Run compaction
    // ========================================================================
    std::cerr << "\n[Phase 5] Running compaction to clean up invalid pages...\n";
    
    CPUTimer compactTimer;
    compactTimer.start();
    rsUpdate.compactPages(index);
    double compactTime = double(compactTimer.stop()) / 1000000.0;
    
    std::cerr << "[Phase 5] Compaction time: " << (compactTime * 1000.0) << " ms\n";
    
    // ========================================================================
    // Final Summary
    // ========================================================================
    std::cerr << "\n================================================================================\n";
    std::cerr << "  FINAL SUMMARY (AFTER COMPACTION)\n";
    std::cerr << "================================================================================\n";
    
    // Final verification
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
    std::cerr << "\n[Per-Batch Timing]\n";
    for (uint32_t b = 0; b < NUM_BATCHES; b++) {
        std::cerr << "  - Batch " << std::setw(2) << (b+1) << ": Del " 
                  << std::fixed << std::setprecision(2) << deleteTimesMs[b] << " ms, Ins " 
                  << std::fixed << std::setprecision(2) << insertTimesMs[b] << " ms\n";
    }
    
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
}
