#include "RunModes.hpp"

void compareResults(int dataId, PVkDevice vd, PBuffer staging, OperatorCache &op) {

    std::cerr << "\n========================================\n";
    std::cerr << "COMPARING RESULTS: Dataset " << dataId << " (" << datasets[dataId] << ")\n";
    std::cerr << "========================================\n";

    // Read dataset
    int32_t ncols;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<std::map<uint32_t, uint32_t>> rowMap;
    std::vector<uint32_t> points;
    PBuffer pointsBuffer = readEncodedData(g_opfolder + datasets[dataId], vd, staging, npoints, minval, maxval, rowMap, points, ncols);
    std::cerr << "Dataset: " << npoints << " points\n";

    // Limit points if exceeds MAX_PAGES
    uint32_t npointsLimited = std::min(npoints, (uint32_t)MAX_PAGES);
    if (npoints > MAX_PAGES) {
        std::cerr << "[WARNING] Limiting points from " << npoints << " to " << npointsLimited << " due to MAX_PAGES\n";
    }

    // Read queries
    std::vector<uint32_t> targets;
    readQueries(g_qfolder + querysets[dataId], qct[dataId], targets, rowMap);

    // Setup for RasterScan2D (original)
    SinglePassScan *scan = (SinglePassScan *) op.getFunction(FunctionType::SinglePassScan);
    ReduceMax *reduce = (ReduceMax *) op.getFunction(FunctionType::ReduceMax);
    PBufferCache bufs1(new CommonBufferPool(vd));
    RasterScan2D rs(vd, bufs1, scan, reduce, ncols);
    
    GPUMemoryTool::printGPUMemoryStatus(vd, "Before RasterScan2D build");
    std::cerr << "\nBuilding RasterScan2D index (taking min of 3 runs)...\n";
    double minBuildTime1 = 1e9;
    PRasterIndex index1;
    for(int k=0; k<3; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";
        CPUTimer buildTimer1;
        buildTimer1.start();
        index1 = rs.buildIndex(pointsBuffer, npointsLimited, minval.data(), maxval.data());
        double bt = double(buildTimer1.stop()) / 1000000.;
        if(bt < minBuildTime1) minBuildTime1 = bt;
        if(k < 2) index1.reset(); // Release buffers
    }
    double buildTime1 = minBuildTime1;
    std::cerr << "RasterScan2D build time: " << buildTime1 << " secs\n";
    GPUMemoryTool::printGPUMemoryStatus(vd, "After RasterScan2D build");

    // Setup for RasterScanIndexUpdate (new)
    // Note: PageAllocator is created lazily in buildIndex() to defer memory allocation
    PBufferCache bufs2(new CommonBufferPool(vd));
    RasterScanIndexUpdate rsUpdate(vd, bufs2, ncols);
    
    GPUMemoryTool::printGPUMemoryStatus(vd, "Before RasterScanIndexUpdate build");
    std::cerr << "Building RasterScanIndexUpdate index (taking min of 3 runs)...\n";
    double minBuildTime2 = 1e9;
    PLinkedListIndex index2;
    for(int k=0; k<3; k++) {
        if(k > 0) std::cerr << "  Run " << k+1 << "...\n";
        CPUTimer buildTimer2;
        buildTimer2.start();
        index2 = rsUpdate.buildIndex(pointsBuffer, npointsLimited, minval.data(), maxval.data());
        double bt = double(buildTimer2.stop()) / 1000000.;
        if(bt < minBuildTime2) minBuildTime2 = bt;
        if(k < 2) index2.reset();
    }
    double buildTime2 = minBuildTime2;
    std::cerr << "RasterScanIndexUpdate build time: " << buildTime2 << " secs\n";
    GPUMemoryTool::printGPUMemoryStatus(vd, "After RasterScanIndexUpdate build");

    // Query buffer
    PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t), 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | 
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 
        MemoryType::Internal);

    uint32_t arrsize = uint32_t(std::ceil(double(npointsLimited) / 32));
    std::vector<uint32_t> res1(arrsize), res2(arrsize);

    double totTime1 = 0, totTime2 = 0;
    int allMatch = 1;

    std::cerr << "\n--- Query Results ---\n";
    std::cerr << std::setw(6) << "Query" 
              << std::setw(12) << "RS2D_Time" 
              << std::setw(12) << "RSIU_Time"
              << std::setw(12) << "RS2D_Rows"
              << std::setw(12) << "RSIU_Rows"
              << std::setw(10) << "Match?\n";
    std::cerr << std::string(62, '-') << "\n";

    for (int i = 0; i < qct[dataId]; i++) {
        int in = i * 6;
        std::vector<uint32_t> queries = {targets[in], targets[in+2], targets[in+1], targets[in+3], targets[in+4], targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(), queries.size() * sizeof(uint32_t), queryBuffer, staging, vd, 0);

        // Run RasterScan2D query
        CPUTimer timer1;
        timer1.start();
        rs.runRangeQueries(index1, queryBuffer, 1);
        double t1 = double(timer1.stop()) / 1000000.;
        totTime1 += t1;
        readUsingStagingBuf((char *)res1.data(), arrsize * sizeof(uint32_t), bufs1->resBuffer, staging, vd);

        // Run RasterScanIndexUpdate query
        CPUTimer timer2;
        timer2.start();
        rsUpdate.runRangeQueries(index2, queryBuffer, 1);
        double t2 = double(timer2.stop()) / 1000000.;
        totTime2 += t2;
        readUsingStagingBuf((char *)res2.data(), arrsize * sizeof(uint32_t), bufs2->resBuffer, staging, vd);

        // Count matching rows for each
        uint32_t count1 = 0, count2 = 0;
        for (uint32_t j = 0; j < arrsize; j++) {
            count1 += __builtin_popcount(res1[j]);
            count2 += __builtin_popcount(res2[j]);
        }

        // Check if results match
        bool match = (res1 == res2);
        if (!match) allMatch = 0;

        std::cerr << std::setw(6) << (i + 1)
                  << std::setw(12) << std::fixed << std::setprecision(6) << t1
                  << std::setw(12) << std::fixed << std::setprecision(6) << t2
                  << std::setw(12) << count1
                  << std::setw(12) << count2
                  << std::setw(10) << (match ? "YES" : "NO") << "\n";
    }

    std::cerr << std::string(62, '-') << "\n";
    std::cerr << "\n--- Summary ---\n";
    std::cerr << "Points compared: " << npointsLimited << "\n";
    std::cerr << "RasterScan2D:       Build=" << std::fixed << std::setprecision(4) << buildTime1 
              << "s, Avg Query=" << std::setprecision(6) << totTime1/qct[dataId] << "s\n";
    std::cerr << "RasterScanIndexUpdate: Build=" << std::fixed << std::setprecision(4) << buildTime2 
              << "s, Avg Query=" << std::setprecision(6) << totTime2/qct[dataId] << "s\n";
    std::cerr << "All results match: " << (allMatch ? "YES" : "NO") << "\n";
    std::cerr << "Speed Up: " << (buildTime1 > 0 ? (buildTime2 / buildTime1) : 0.0) << "x for build, " << (totTime1 > 0 ? (totTime2 / totTime1) : 0.0) << "x for query on average\n";
    std::cerr << "========================================\n\n";
}
