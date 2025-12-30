#include "RunModes.hpp"

void queriesGPUWithOutput(int dataId, PVkDevice vd, PBuffer staging, OperatorCache &op, bool test) {
    
    SinglePassScan *scan = (SinglePassScan *) op.getFunction(FunctionType::SinglePassScan);
    ReduceMax *reduce = (ReduceMax *) op.getFunction(FunctionType::ReduceMax);
    PBufferCache bufs(new CommonBufferPool(vd));

    std::cerr << "=================== GPU Query Execution ===================\n";
    std::cerr << "reading dataset: " << dataId << " (" << datasets[dataId] << ")\n";
    int32_t ncols;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<std::map<uint32_t, uint32_t>> rowMap;
    std::vector<uint32_t> points;
    PBuffer pointsBuffer = readEncodedData(g_opfolder + datasets[dataId],vd,staging,npoints,minval,maxval,rowMap,points,ncols);
    
    // Calculate input file size (4 bytes per uint32_t)
    uint32_t fileSize = npoints * ncols * 4;
    std::cerr << "input file size: " << fileSize << " with " << npoints << " points\n";
    
    // Print GPU memory info
    std::cerr << "[GPU Memory] Points Buffer                 :    " << std::fixed << std::setprecision(3) 
              << (double)fileSize / (1024.0 * 1024.0) << " MB (" << fileSize << " bytes)\n";
    
    std::cerr << "finished reading data " << minval[0] << ";" << maxval[0] << ",  " << minval[1] << ";" << maxval[1] << ",  " << minval[2] << ";" << maxval[2] << "\n";

    RasterScan2D rs(vd, bufs, scan, reduce, ncols);

    std::cerr << "building index\n";
    CPUTimer indexTimer;
    indexTimer.start();
    PRasterIndex index = rs.buildIndex(pointsBuffer,npoints,minval.data(),maxval.data());
    uint64_t indexTime = indexTimer.stop();
    double indexTimeMs = double(indexTime) / 1000.0;
    std::cerr << "finished building index with max bin size: " << index->maxBinCt << "\n";
    std::cerr << "stats -- min: " << index->minVal[0] << ";" << index->minVal[1] << ", max: " << index->maxVal[0] << ";" << index->maxVal[1]  << ", range per bin: " << index->binRange[0]  << ";" << index->binRange[1]  << "\n";
    std::cerr << "[TIME]index build time: " << indexTimeMs << " ms\n";

    std::vector<uint32_t> targets;
    readQueries(g_qfolder + querysets[dataId], qct[dataId], targets, rowMap);

    for(int i = 0;i < targets.size()/2;i ++) {
        if(i % 3 == 0) std::cerr << "Query: " << i / 3 << "\n";
        std::cerr << targets[i * 2] << "," << targets[i * 2 + 1] << "\n";
    }

    // Output file setup
    std::string outputDir = PROJECT_DIR + "compare/gpu_files/";
    std::string outputFile = outputDir + datasets[dataId] + "_results.txt";
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "ERROR: Cannot open output file: " << outputFile << "\n";
        return;
    }

    // Execute queries.
    PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t),vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst,MemoryType::Internal);
    double totTime = 0;
    int nct = 0;
    int totalMatches = 0;
    
    for(int i = 0;i < qct[dataId];i ++) {
        int in = i * 6;
        // x1,y1,x2,y2,z1,z2
        std::vector<uint32_t> queries = {targets[in],targets[in+2],targets[in+1],targets[in+3],targets[in+4],targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(),queries.size() * sizeof(uint32_t),queryBuffer,staging,vd,0);
        CPUTimer timer;
        timer.start();
        rs.runRangeQueries(index,queryBuffer,1);
        uint64_t t = timer.stop();
        double ts = double(t) / 1000000.;
        totTime += ts;
        nct ++;

        // Extract matching rowIds
        uint32_t arrsize = uint32_t(std::ceil(double(npoints) / 32));
        std::vector<uint32_t> res(arrsize);
        readUsingStagingBuf((char *)res.data(),arrsize * sizeof(uint32_t),bufs->resBuffer,staging,vd);
        
        std::vector<uint32_t> matchingRowIds;
        for(uint32_t j = 0;j < npoints;j ++) {
            uint32_t ind = j >> 5;
            uint32_t bit = 1 << (j & 0x1f);
            uint32_t resbit = res[ind] & bit;
            resbit >>= (j & 0x1f);
            if(resbit) {
                matchingRowIds.push_back(j);
            }
        }

        // Write results to file
        uint32_t x1 = targets[in];
        uint32_t x2 = targets[in + 1];
        uint32_t y1 = targets[in + 2];
        uint32_t y2 = targets[in + 3];
        uint32_t z1 = targets[in + 4];
        uint32_t z2 = targets[in + 5];
        
        outFile << "Query " << (i + 1) << " [x:" << x1 << "-" << x2 << ", y:" << y1 << "-" << y2 
                << ", z:" << z1 << "-" << z2 << "]\n";
        outFile << "Matches: " << matchingRowIds.size() << "\n";
        outFile << "RowIds: ";
        for (size_t j = 0; j < matchingRowIds.size(); j++) {
            if (j > 0) outFile << ",";
            outFile << matchingRowIds[j];
        }
        outFile << "\n\n";
        
        totalMatches += matchingRowIds.size();
        
        // Print to stderr in CPU format
        std::cerr << "Query " << (i + 1) << " [x:" << x1 << "-" << x2 << ", y:" << y1 << "-" << y2 
                  << ", z:" << z1 << "-" << z2 << "]: " << matchingRowIds.size() << " matches in " << ts << " secs\n";
    }

    outFile << "=== Summary ===\n";
    outFile << "Dataset: " << datasets[dataId] << "\n";
    outFile << "Total points: " << npoints << "\n";
    outFile << "Queries executed: " << nct << "\n";
    outFile << "Total matches: " << totalMatches << "\n";
    outFile << "Avg. query time: " << (totTime / nct) << " secs\n";
    outFile << "Throughput: " << (nct / totTime) << " qps\n";

    outFile.close();
    
    std::cerr << "Results written to: " << outputFile << "\n";
    std::cerr << "Avg. query time over " << nct << " queries: " << totTime / nct << " secs\n";
    std::cerr << "Throughput: " << nct/totTime << " qps\n";
    std::cerr << "\n";
}
