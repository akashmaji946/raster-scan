#include "RunModes.hpp"

// Helper function to print allocation stats with verification
void printAllocationStats(const RasterScanIndexUpdate::AllocationStats& stats, const std::string& label) {
    std::cerr << label << ":\n";
    std::cerr << "          - Total pages:       " << stats.totalPages << "\n";
    std::cerr << "          - Allocated (in use):" << stats.allocatedPages << "\n";
    std::cerr << "          - Free (in bitmap):  " << stats.freePages << "\n";
    
    // Verification: allocated + free = total
    bool totalCheck = (stats.allocatedPages + stats.freePages == stats.totalPages);
    
    std::cerr << "          [Check] allocated + free = total: " 
              << (totalCheck ? "PASS" : "FAIL") << " ("
              << stats.allocatedPages << " + " << stats.freePages << " = " << stats.totalPages << ")\n";
}

void testBitmapFreeSpaceManagement(PVkDevice vd, PBuffer staging) {
    std::cerr << "\n";
    std::cerr << "================================================================================\n";
    std::cerr << "  MODE 6: BITMAP-BASED FREE SPACE MANAGEMENT STRESS TEST\n";
    std::cerr << "================================================================================\n";
    std::cerr << "\n";
    
    // Configuration
    // MAX_PAGES is usually 128 million (defined in GPUMemoryTool.hpp or similar)
    // We want to test alloc/free cycles
    const uint32_t NUM_POINTS = 10000000; // 10M points
    const uint32_t NUM_CYCLES = 5;
    
    std::cerr << "[Config] MAX_PAGES: " << MAX_PAGES << "\n";
    std::cerr << "[Config] NUM_POINTS: " << NUM_POINTS << "\n";
    std::cerr << "[Config] NUM_CYCLES: " << NUM_CYCLES << "\n\n";
    
    // Create random data points
    std::vector<uint32_t> points(NUM_POINTS * 3);
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, 1000000);
    
    std::cerr << "[Phase 0] Generating " << NUM_POINTS << " random points...\n";
    for (uint32_t i = 0; i < NUM_POINTS; i++) {
        points[i] = dist(rng);                    // x
        points[NUM_POINTS + i] = dist(rng);       // y
        points[2 * NUM_POINTS + i] = dist(rng);   // z
    }
    
    // Create points buffer
    PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(NUM_POINTS * 3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)points.data(), NUM_POINTS * 3 * sizeof(uint32_t), pointsBuffer, staging, vd, 0);
    
    // Create index
    PBufferCache bufs(new CommonBufferPool(vd));
    RasterScanIndexUpdate rsUpdate(vd, bufs, 3);
    
    // ========================================================================
    // Phase 1: Initial Index Build
    // ========================================================================
    std::cerr << "\n[Phase 1] Building initial index with " << NUM_POINTS << " points...\n";
    
    // Setup manual initialization
    rsUpdate.pageAlloc.reset(new PageAllocator(vd, MAX_PAGES));
    PLinkedListIndex index(new LinkedListIndex(vd, NUM_POINTS, rsUpdate.pageAlloc));
    
    // Initialize bitmap
    // At start: 0 allocated, MAX_PAGES free
    rsUpdate.initializeBitmapWithAllocation(0);
    
    auto stats = rsUpdate.getAllocationStats(index);
    printAllocationStats(stats, "Initial State");
    
    // Insert points using bitmap allocation
    std::cerr << "  -> Inserting " << NUM_POINTS << " points using bitmap allocator...\n";
    CPUTimer insertTimer;
    insertTimer.start();
    // Use V2 which is faster
    rsUpdate.insertPointsWithBitmapV2(index, pointsBuffer, NUM_POINTS, 0);
    double insertTime = double(insertTimer.stop()) / 1000000.0;
    std::cerr << "  -> Insert time: " << insertTime << " secs\n";
    
    stats = rsUpdate.getAllocationStats(index);
    printAllocationStats(stats, "After Initial Insert");
    
    if (stats.allocatedPages != NUM_POINTS) {
        std::cerr << "ERROR: Allocated pages (" << stats.allocatedPages << ") != NUM_POINTS (" << NUM_POINTS << ")\n";
        return;
    }
    
    // ========================================================================
    // Cycles: Delete, Compact, Insert
    // ========================================================================
    
    // We will delete ALL points, compact (should free everything), then re-insert
    
    for (int cycle = 1; cycle <= NUM_CYCLES; cycle++) {
        std::cerr << "\n[Cycle " << cycle << "] Delete -> Compact -> Insert\n";
        
        // --- DELETE ---
        // To delete all points, we can use deletePointsByData with the same points buffer
        // Or simpler: just use deletePoints (by rowId) if we had rowIds.
        // But here pointsBuffer has x,y,z columns.
        // Let's use deletePointsByData which matches x,y,z and marks invalid.
        // NOTE: deletePointsByData is expensive (O(N) per delete).
        // Since we want to delete ALL, maybe just resetting is easier?
        // But we want to test the mechanisms.
        // Let's delete a subset to be faster, say 50%.
        
        uint32_t nDelete = NUM_POINTS / 2;
        std::cerr << "  -> Deleting " << nDelete << " points (50%)...\n";
        
        // Create a buffer with first 50% of points
        PBuffer deleteBuffer(new Buffer(vd));
        deleteBuffer->create(nDelete * 3 * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            MemoryType::Internal);
            
        // Copy first 50% from CPU vector
        // Need to extract columns
        std::vector<uint32_t> deletePoints(nDelete * 3);
        for(uint32_t i=0; i<nDelete; i++) {
            deletePoints[i] = points[i];
            deletePoints[nDelete+i] = points[NUM_POINTS+i];
            deletePoints[2*nDelete+i] = points[2*NUM_POINTS+i];
        }
        loadUsingStagingBuf((char*)deletePoints.data(), nDelete * 3 * sizeof(uint32_t), deleteBuffer, staging, vd, 0);
        
        CPUTimer delTimer;
        delTimer.start();
        rsUpdate.deletePointsByData(index, deleteBuffer, nDelete);
        double delTime = double(delTimer.stop()) / 1000000.0;
        std::cerr << "  -> Delete time: " << delTime << " secs\n";
        
        // After delete, pages are marked invalid but still allocated
        stats = rsUpdate.getAllocationStats(index);
        printAllocationStats(stats, "After Delete (Before Compact)");
        
        // Allocated should still be same
        // But valid pages inside should decrease? getAllocationStats doesn't count valid bits deeply
        
        // --- COMPACT ---
        std::cerr << "  -> Compacting pages (freeing invalid pages)...\n";
        CPUTimer compactTimer;
        compactTimer.start();
        rsUpdate.compactPages(index);
        double compactTime = double(compactTimer.stop()) / 1000000.0;
        std::cerr << "  -> Compact time: " << compactTime << " secs\n";
        
        stats = rsUpdate.getAllocationStats(index);
        printAllocationStats(stats, "After Compact");
        
        // Expectation: Allocated should drop by nDelete (approx, collisions might affect it)
        // Actually, deletePointsByData marks pages as invalid if data matches.
        // Since we deleted exactly what we inserted (and unique randoms likely), 
        // we expect exactly nDelete pages to be freed.
        std::cerr << "  -> Freed pages: " << (stats.freePages - (MAX_PAGES - NUM_POINTS)) << "\n";
        
        // --- INSERT ---
        // Re-insert the deleted points
        std::cerr << "  -> Re-inserting " << nDelete << " points...\n";
        CPUTimer reinsertTimer;
        reinsertTimer.start();
        rsUpdate.insertPointsWithBitmapV2(index, deleteBuffer, nDelete, 0); // rowIdOffset doesn't matter for this test
        double reinsertTime = double(reinsertTimer.stop()) / 1000000.0;
        std::cerr << "  -> Insert time: " << reinsertTime << " secs\n";
        
        stats = rsUpdate.getAllocationStats(index);
        printAllocationStats(stats, "After Re-insert");
        
        // Should be back to full allocation
        if (stats.allocatedPages != NUM_POINTS) {
            std::cerr << "WARNING: Allocation count " << stats.allocatedPages << " != expected " << NUM_POINTS << "\n";
        }
        
        deleteBuffer->destroy();
        deleteBuffer.reset();
    }
    
    std::cerr << "\n================================================================================\n";
    std::cerr << "STRESS TEST COMPLETE\n";
    std::cerr << "================================================================================\n\n";
    
    // Cleanup
    pointsBuffer->destroy();
    pointsBuffer.reset();
    index->headPtrBuffer->destroy();
    index.reset();
    rsUpdate.freeBitmap.reset(); // Important to release bitmap memory
    rsUpdate.pageAlloc->pageBuffer->destroy();
    rsUpdate.pageAlloc->allocCounterBuffer->destroy();
    rsUpdate.pageAlloc.reset();
    bufs.reset();
}
