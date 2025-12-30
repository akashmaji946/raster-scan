#include "RunModes.hpp"

void testDynamicOperations(PVkDevice vd, PBuffer staging, std::string datasetName) {

    std::string testFolder =  std::string(PROJECT_DIR) + "tests/dynamic/" + datasetName + "/";
    
    std::cerr << "\n========================================\n";
    std::cerr << "TESTING DYNAMIC INDEX OPERATIONS (" << datasetName << ")\n";
    std::cerr << "========================================\n";
    
    // Read initial data
    std::ifstream initFile(testFolder + "initial_data.bin", std::ios::binary);
    if (!initFile.is_open()) {
        std::cerr << "ERROR: Could not open " << testFolder << "initial_data.bin\n";
        return;
    }
    uint32_t nInitial;
    initFile.read(reinterpret_cast<char*>(&nInitial), sizeof(nInitial));
    
    // Limit to MAX_PAGES since we use 1-item-per-page
    uint32_t nPointsToLoad = std::min(nInitial, (uint32_t)MAX_PAGES);
    if (nPointsToLoad < nInitial) {
        std::cerr << "WARNING: Dataset has " << nInitial << " points but MAX_PAGES=" << MAX_PAGES 
                  << ". Loading only " << nPointsToLoad << " points.\n";
    }
    
    // Read points directly into column format to save memory
    std::vector<uint32_t> points(nPointsToLoad * 3);
    uint32_t minVal[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint32_t maxVal[3] = {0, 0, 0};
    
    for (uint32_t i = 0; i < nPointsToLoad; i++) {
        uint32_t x, y, z, rowId;
        initFile.read(reinterpret_cast<char*>(&x), sizeof(uint32_t));
        initFile.read(reinterpret_cast<char*>(&y), sizeof(uint32_t));
        initFile.read(reinterpret_cast<char*>(&z), sizeof(uint32_t));
        initFile.read(reinterpret_cast<char*>(&rowId), sizeof(uint32_t));
        
        points[i] = x;
        points[nPointsToLoad + i] = y;
        points[2 * nPointsToLoad + i] = z;
        
        minVal[0] = std::min(minVal[0], x);
        maxVal[0] = std::max(maxVal[0], x);
        minVal[1] = std::min(minVal[1], y);
        maxVal[1] = std::max(maxVal[1], y);
        minVal[2] = std::min(minVal[2], z);
        maxVal[2] = std::max(maxVal[2], z);
    }
    initFile.close();
    nInitial = nPointsToLoad;  // Update to actual loaded count
    std::cerr << "Loaded " << nInitial << " initial points\n";
    
    // Create points buffer
    PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(nInitial * 3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)points.data(), nInitial * 3 * sizeof(uint32_t), pointsBuffer, staging, vd, 0);
    
    // Create index
    PBufferCache bufs(new CommonBufferPool(vd));
    RasterScanIndexUpdate rsUpdate(vd, bufs, 3);
    
    std::cerr << "Building index with " << nInitial << " points...\n";
    PLinkedListIndex index = rsUpdate.buildIndex(pointsBuffer, nInitial, minVal, maxVal);
    
    // Read test queries
    std::ifstream queryFile(testFolder + "test_queries.txt");
    if (!queryFile.is_open()) {
        std::cerr << "ERROR: Could not open " << testFolder << "test_queries.txt\n";
        return;
    }
    uint32_t nQueries;
    queryFile >> nQueries;
    std::vector<uint32_t> queries(nQueries * 6);
    for (uint32_t i = 0; i < nQueries; i++) {
        queryFile >> queries[i * 6] >> queries[i * 6 + 1] >> queries[i * 6 + 2]
                  >> queries[i * 6 + 3] >> queries[i * 6 + 4] >> queries[i * 6 + 5];
    }
    queryFile.close();
    
    // Query buffer
    PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    
    // Account for inserts - use a larger buffer for result bitmap
    uint32_t maxRowId = nInitial + 1000000;  // Allow for many inserts
    uint32_t arrsize = uint32_t(std::ceil(double(maxRowId) / 32));
    std::vector<uint32_t> results(arrsize);
    
    auto runQueries = [&](const std::string& phase) {
        std::cerr << "\n--- " << phase << " ---\n";
        for (uint32_t q = 0; q < nQueries; q++) {
            // Query file format: x1 x2 y1 y2 z1 z2
            // Shader expects: qrange=(x1,y1,x2,y2), zrange=(z1,z2)
            uint32_t x1 = queries[q * 6];
            uint32_t x2 = queries[q * 6 + 1];
            uint32_t y1 = queries[q * 6 + 2];
            uint32_t y2 = queries[q * 6 + 3];
            uint32_t z1 = queries[q * 6 + 4];
            uint32_t z2 = queries[q * 6 + 5];
            
            std::vector<uint32_t> qdata = {x1, y1, x2, y2, z1, z2};
            loadUsingStagingBuf((char*)qdata.data(), 6 * sizeof(uint32_t), queryBuffer, staging, vd, 0);
            rsUpdate.runRangeQueries(index, queryBuffer, 1);
            readUsingStagingBuf((char*)results.data(), arrsize * sizeof(uint32_t), bufs->resBuffer, staging, vd);
            
            uint32_t count = 0;
            for (uint32_t j = 0; j < arrsize; j++) {
                count += __builtin_popcount(results[j]);
            }
            std::cerr << "Query " << (q + 1) << " [x:" << x1 << "-" << x2 << ", y:"
                      << y1 << "-" << y2 << ", z:" << z1 << "-" << z2 
                      << "]: " << count << " matches\n";
        }
    };
    
    // Phase 1: Initial queries
    runQueries("After Initial Build");
    
    // Phase 2: Delete some points
    std::ifstream deleteFile(testFolder + "delete_rowids.txt");
    uint32_t nDeletes;
    deleteFile >> nDeletes;
    std::vector<uint32_t> deleteRowIds(nDeletes);
    for (uint32_t i = 0; i < nDeletes; i++) {
        deleteFile >> deleteRowIds[i];
    }
    deleteFile.close();
    
    PBuffer deleteBuffer(new Buffer(vd));
    deleteBuffer->create(nDeletes * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)deleteRowIds.data(), nDeletes * sizeof(uint32_t), deleteBuffer, staging, vd, 0);
    
    std::cerr << "\nDeleting " << nDeletes << " points...\n";
    rsUpdate.deletePoints(index, deleteBuffer, nDeletes);
    
    runQueries("After Delete");
    
    // Phase 3: Insert new points
    std::ifstream insertFile(testFolder + "insert_data.bin", std::ios::binary);
    uint32_t nInserts;
    insertFile.read(reinterpret_cast<char*>(&nInserts), sizeof(nInserts));
    std::vector<uint32_t> insertData(nInserts * 4);
    for (uint32_t i = 0; i < nInserts; i++) {
        insertFile.read(reinterpret_cast<char*>(&insertData[i * 4]), sizeof(uint32_t));
        insertFile.read(reinterpret_cast<char*>(&insertData[i * 4 + 1]), sizeof(uint32_t));
        insertFile.read(reinterpret_cast<char*>(&insertData[i * 4 + 2]), sizeof(uint32_t));
        insertFile.read(reinterpret_cast<char*>(&insertData[i * 4 + 3]), sizeof(uint32_t));
    }
    insertFile.close();
    
    // Reorganize insert data into column format
    std::vector<uint32_t> insertPoints(nInserts * 3);
    for (uint32_t i = 0; i < nInserts; i++) {
        insertPoints[i] = insertData[i * 4];
        insertPoints[nInserts + i] = insertData[i * 4 + 1];
        insertPoints[2 * nInserts + i] = insertData[i * 4 + 2];
    }
    
    PBuffer insertBuffer(new Buffer(vd));
    insertBuffer->create(nInserts * 3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)insertPoints.data(), nInserts * 3 * sizeof(uint32_t), insertBuffer, staging, vd, 0);
    
    std::cerr << "\nInserting " << nInserts << " new points...\n";
    rsUpdate.insertPoints(index, insertBuffer, nInserts);
    
    runQueries("After Insert");
    
    std::cerr << "\n========================================\n";
    std::cerr << "DYNAMIC OPERATIONS TEST COMPLETE\n";
    std::cerr << "========================================\n\n";
}
