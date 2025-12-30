#include "RunModes.hpp"

void testRangeDelete(PVkDevice vd, PBuffer staging, std::string datasetName) {
    std::string testFolder = std::string(PROJECT_DIR) + "tests/dynamic/" + datasetName + "_range/";
    
    std::cerr << "\n========================================\n";
    std::cerr << "TESTING RANGE DELETE OPERATIONS (" << datasetName << ")\n";
    std::cerr << "========================================\n";
    
    // Load initial data from binary file
    std::ifstream dataFile(testFolder + "initial_data.bin", std::ios::binary);
    if (!dataFile.is_open()) {
        std::cerr << "Error: Cannot open " << testFolder << "initial_data.bin\n";
        return;
    }
    
    uint32_t nInitial;
    dataFile.read(reinterpret_cast<char*>(&nInitial), sizeof(uint32_t));
    
    // Limit to MAX_PAGES
    uint32_t nPointsToLoad = std::min(nInitial, (uint32_t)MAX_PAGES);
    if (nPointsToLoad < nInitial) {
        std::cerr << "WARNING: Dataset has " << nInitial << " points but MAX_PAGES=" << MAX_PAGES 
                  << ". Loading only " << nPointsToLoad << " points.\n";
    }
    
    // Stream data directly into column-major format
    std::vector<uint32_t> points(nPointsToLoad * 3);
    uint32_t minVal[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint32_t maxVal[3] = {0, 0, 0};
    
    const uint32_t batchSize = 100000;
    std::vector<uint32_t> batch(batchSize * 4);
    
    for (uint32_t i = 0; i < nPointsToLoad; i += batchSize) {
        uint32_t count = std::min(batchSize, nPointsToLoad - i);
        dataFile.read(reinterpret_cast<char*>(batch.data()), count * 4 * sizeof(uint32_t));
        
        for (uint32_t j = 0; j < count; j++) {
            uint32_t x = batch[j * 4 + 0];
            uint32_t y = batch[j * 4 + 1];
            uint32_t z = batch[j * 4 + 2];
            
            points[i + j] = x;
            points[nPointsToLoad + i + j] = y;
            points[2 * nPointsToLoad + i + j] = z;
            
            minVal[0] = std::min(minVal[0], x);
            minVal[1] = std::min(minVal[1], y);
            minVal[2] = std::min(minVal[2], z);
            maxVal[0] = std::max(maxVal[0], x);
            maxVal[1] = std::max(maxVal[1], y);
            maxVal[2] = std::max(maxVal[2], z);
        }
    }
    dataFile.close();
    std::cerr << "Loaded " << nPointsToLoad << " initial points\n";
    
    // Load delete ranges
    std::ifstream deleteFile(testFolder + "delete_ranges.txt");
    if (!deleteFile.is_open()) {
        std::cerr << "Error: Cannot open " << testFolder << "delete_ranges.txt\n";
        return;
    }
    
    uint32_t nDeleteRanges;
    deleteFile >> nDeleteRanges;
    std::vector<std::array<uint32_t, 6>> deleteRanges(nDeleteRanges);
    for (uint32_t i = 0; i < nDeleteRanges; i++) {
        deleteFile >> deleteRanges[i][0] >> deleteRanges[i][1] 
                   >> deleteRanges[i][2] >> deleteRanges[i][3]
                   >> deleteRanges[i][4] >> deleteRanges[i][5];
    }
    deleteFile.close();
    std::cerr << "Loaded " << nDeleteRanges << " delete ranges\n";
    
    // Load test queries
    std::ifstream queryFile(testFolder + "test_queries.txt");
    if (!queryFile.is_open()) {
        std::cerr << "Error: Cannot open " << testFolder << "test_queries.txt\n";
        return;
    }
    
    uint32_t nTestQueries;
    queryFile >> nTestQueries;
    std::vector<std::array<uint32_t, 6>> testQueries(nTestQueries);
    for (uint32_t i = 0; i < nTestQueries; i++) {
        queryFile >> testQueries[i][0] >> testQueries[i][1] 
                  >> testQueries[i][2] >> testQueries[i][3]
                  >> testQueries[i][4] >> testQueries[i][5];
    }
    queryFile.close();
    std::cerr << "Loaded " << nTestQueries << " test queries\n";
    
    // Create points buffer
    PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(nPointsToLoad * 3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)points.data(), nPointsToLoad * 3 * sizeof(uint32_t), pointsBuffer, staging, vd, 0);
    
    // Create index
    PBufferCache bufs(new CommonBufferPool(vd));
    RasterScanIndexUpdate rsUpdate(vd, bufs, 3);
    
    std::cerr << "Building index with " << nPointsToLoad << " points...\n";
    PLinkedListIndex index = rsUpdate.buildIndex(pointsBuffer, nPointsToLoad, minVal, maxVal);
    
    // Query buffer
    PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    // Allocate result buffer size based on loaded points * 2 (for insert test) + margin
    // After insert, we'll have up to 2 * nPointsToLoad unique rowIds
    uint32_t arrsize = uint32_t(std::ceil(double(nPointsToLoad * 2 + 1000000) / 32));
    std::vector<uint32_t> results(arrsize);
    
    // Helper to run a query and count matches
    auto runQuery = [&](uint32_t x1, uint32_t x2, uint32_t y1, uint32_t y2, uint32_t z1, uint32_t z2) -> uint32_t {
        std::vector<uint32_t> qdata = {x1, y1, x2, y2, z1, z2};
        loadUsingStagingBuf((char*)qdata.data(), 6 * sizeof(uint32_t), queryBuffer, staging, vd, 0);
        rsUpdate.runRangeQueries(index, queryBuffer, 1);
        readUsingStagingBuf((char*)results.data(), arrsize * sizeof(uint32_t), bufs->resBuffer, staging, vd);
        
        uint32_t count = 0;
        for (uint32_t j = 0; j < arrsize; j++) {
            count += __builtin_popcount(results[j]);
        }
        return count;
    };
    
    // Run test queries before deletion
    auto runAllQueries = [&](const std::string& phase) {
        std::cerr << "\n--- " << phase << " ---\n";
        for (size_t i = 0; i < testQueries.size(); i++) {
            auto& q = testQueries[i];
            uint32_t count = runQuery(q[0], q[1], q[2], q[3], q[4], q[5]);
            std::cerr << "Query " << (i+1) << " [x:" << q[0] << "-" << q[1] 
                      << ", y:" << q[2] << "-" << q[3] 
                      << ", z:" << q[4] << "-" << q[5] << "]: " << count << " matches\n";
        }
    };
    
    // Phase 1: Run test queries on initial data
    runAllQueries("Phase 1: Initial Data (Before Deletions)");
    
    // Phase 2: Delete each range
    std::cerr << "\nPhase 2: Deleting " << nDeleteRanges << " ranges...\n";
    for (size_t i = 0; i < deleteRanges.size(); i++) {
        auto& r = deleteRanges[i];
        uint32_t deleteRange[6] = {r[0], r[1], r[2], r[3], r[4], r[5]};
        rsUpdate.deleteRange(index, deleteRange);
    }
    
    // Phase 3: Run test queries after deletion
    runAllQueries("Phase 2: After Deletions");
    
    // Phase 4: Load insert data and insert into index
    std::cerr << "\nPhase 3: Loading and inserting new data...\n";
    std::ifstream insertFile(testFolder + "initial_data.bin", std::ios::binary);
    if (!insertFile.is_open()) {
        std::cerr << "Error: Cannot open " << testFolder << "initial_data.bin\n";
        std::cerr << "Skipping insert phase\n";
    } else {
        uint32_t nInsert;
        insertFile.read(reinterpret_cast<char*>(&nInsert), sizeof(uint32_t));
        
        // Limit to available space (MAX_PAGES - current points)
        uint32_t nInsertToLoad = std::min(nInsert, (uint32_t)MAX_PAGES - nPointsToLoad);
        if (nInsertToLoad < nInsert) {
            std::cerr << "WARNING: Insert data has " << nInsert << " points but only " << nInsertToLoad 
                      << " can fit. Loading " << nInsertToLoad << " points.\n";
        }
        
        // Load insert data
        std::vector<uint32_t> insertPoints(nInsertToLoad * 3);
        for (uint32_t i = 0; i < nInsertToLoad; i += batchSize) {
            uint32_t count = std::min(batchSize, nInsertToLoad - i);
            insertFile.read(reinterpret_cast<char*>(batch.data()), count * 4 * sizeof(uint32_t));
            
            for (uint32_t j = 0; j < count; j++) {
                uint32_t x = batch[j * 4 + 0];
                uint32_t y = batch[j * 4 + 1];
                uint32_t z = batch[j * 4 + 2];
                
                insertPoints[i + j] = x;
                insertPoints[nInsertToLoad + i + j] = y;
                insertPoints[2 * nInsertToLoad + i + j] = z;
            }
        }
        insertFile.close();
        std::cerr << "Loaded " << nInsertToLoad << " insert points\n";
        
        // Create insert buffer
        PBuffer insertBuffer(new Buffer(vd));
        insertBuffer->create(nInsertToLoad * 3 * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            MemoryType::Internal);
        loadUsingStagingBuf((char*)insertPoints.data(), nInsertToLoad * 3 * sizeof(uint32_t), insertBuffer, staging, vd, 0);
        
        // Insert points into index with rowIdOffset = nPointsToLoad
        // This ensures inserted points get unique rowIds (nPointsToLoad to nPointsToLoad + nInsertToLoad - 1)
        std::cerr << "Inserting " << nInsertToLoad << " points into index with rowIdOffset=" << nPointsToLoad << "...\n";
        rsUpdate.insertPoints(index, insertBuffer, nInsertToLoad, nPointsToLoad);
        std::cerr << "Insert complete\n";
        
        // Phase 4: Run test queries after insert (BEFORE second delete)
        // The newly inserted data is all valid, so results should be ~2x Phase 2
        runAllQueries("Phase 3: After Insert (Should be ~2x Phase 2)");
        
        // Phase 5: Run delete again on the combined data
        std::cerr << "\nPhase 4: Deleting ranges again on combined data...\n";
        for (size_t i = 0; i < deleteRanges.size(); i++) {
            auto& r = deleteRanges[i];
            uint32_t deleteRange[6] = {r[0], r[1], r[2], r[3], r[4], r[5]};
            rsUpdate.deleteRange(index, deleteRange);
        }
        
        // Phase 6: Run test queries after second deletion
        // Should be same as Phase 2 (deleted from both original and inserted data)
        runAllQueries("Phase 4: After Second Delete (Should be same as Phase 2)");
        
        insertBuffer->destroy();
        insertBuffer.reset();
    }
    
    std::cerr << "\n========================================\n";
    std::cerr << "RANGE DELETE TEST COMPLETE (" << datasetName << ")\n";
    std::cerr << "========================================\n\n";
}
