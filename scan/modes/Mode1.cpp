#include "RunModes.hpp"

void queries3dIndexUpdate(int dataId, PVkDevice vd, PBuffer staging, bool test) {


    PBufferCache bufs(new CommonBufferPool(vd));

    std::cerr << "\n=================== RasterScanIndexUpdate Pipeline ===================\n";
    std::cerr << "reading dataset: " << dataId << " (" << datasets[dataId] << ")\n";
    int32_t ncols;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<std::map<uint32_t, uint32_t>> rowMap;
    std::vector<uint32_t> points;
    PBuffer pointsBuffer = readEncodedData(g_opfolder + datasets[dataId],vd,staging,npoints,minval,maxval,rowMap,points,ncols);

    // Limit points if exceeds MAX_PAGES
    if (npoints > MAX_PAGES) {
        std::cerr << "[WARNING] Limiting points from " << npoints << " to " << MAX_PAGES << " due to MAX_PAGES\n";
        npoints = MAX_PAGES;
    }
    std::cerr << "finished reading data " << minval[0] << ";" << maxval[0] << ",  " << minval[1] << ";" << maxval[1] << ",  " << minval[2] << ";" << maxval[2] << "\n";

    // Create RasterScanIndexUpdate instance
    RasterScanIndexUpdate rsUpdate(vd, bufs, ncols);

    // Build linked list index
    std::cerr << "Building linked list index\n";
    CPUTimer buildTimer;
    buildTimer.start();
    PLinkedListIndex index = rsUpdate.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    uint64_t buildTime = buildTimer.stop();
    std::cerr << ">>> Finished building linked list index in " << (double(buildTime) / 1000.0) << " ms\n";
    std::cerr << "stats -- min: " << index->minVal[0] << ";" << index->minVal[1] << ", max: " << index->maxVal[0] << ";" << index->maxVal[1]  << ", range per bin: " << index->binRange[0]  << ";" << index->binRange[1]  << "\n";

    std::vector<uint32_t> targets;
    readQueries(g_qfolder + querysets[dataId], qct[dataId], targets, rowMap);

    for(int i = 0;i < targets.size()/2;i ++) {
        if(i % 3 == 0) std::cerr << "Query: " << i / 3 << "\n";
        std::cerr << targets[i * 2] << "," << targets[i * 2 + 1] << "\n";
    }

    // Execute queries
    PBuffer queryBuffer(new Buffer(vd));
    queryBuffer->create(6 * sizeof(uint32_t),vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst,MemoryType::Internal);
    double totTime = 0;
    int nct = 0;
    
    for(int i = 0;i < qct[dataId];i ++) {
        int in = i * 6;
        // x1,y1,x2,y2,z1,z2
        std::vector<uint32_t> queries = {targets[in],targets[in+2],targets[in+1],targets[in+3],targets[in+4],targets[in+5]};
        loadUsingStagingBuf((char *)queries.data(),queries.size() * sizeof(uint32_t),queryBuffer,staging,vd,0);
        CPUTimer timer;
        timer.start();
        rsUpdate.runRangeQueries(index, queryBuffer, 1);
        uint64_t t = timer.stop();
        double ts = double(t) / 1000000.;
        totTime += ts;
        nct ++;
        std::cerr << "time for query " << (i + 1) << ": " << ts << " secs\n";

        if(test) {
            uint32_t arrsize = uint32_t(std::ceil(double(npoints) / 32));
            std::vector<uint32_t> res(arrsize);
            readUsingStagingBuf((char *)res.data(),arrsize * sizeof(uint32_t),bufs->resBuffer,staging,vd);
            int32_t tot = 0;
            for(uint32_t i = 0;i < npoints;i ++) {
                if(i % 1000 == 0) {
                    std::cerr << "\rtested " << i << " of " << npoints;
                }
                uint32_t pt[] = {points[i], points[i + npoints], points[i + 2 * npoints]};

                uint32_t sat = 1;
                for(int j = 0;j < 3;j ++) {
                    if(!(targets[in + j * 2] <= pt[j] && targets[in + j * 2 + 1] >= pt[j])) {
                        sat = 0;
                        break;
                    }
                }
                tot += sat;
                uint32_t ind = i >> 5;
                uint32_t bit = 1 << (i & 0x1f);
                uint32_t resbit = res[ind] & bit;
                resbit >>= (i & 0x1f);
                if(resbit != sat)
                {
                    std::cerr << "\nError!! results not match! " << i << "," << ind << "," << resbit << "," << sat << ","
                              << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
                    for(int xx = 0;xx < 9;xx ++) {
                        std::cerr << res[xx] << ",";
                    }
                    std::cerr<< "\n";
                    exit(0);
                }
            }
            std::cerr << "\nTest successful: " << tot << "\n";
        }
    }
    std::cerr << "Avg. query time over " << nct << " queries: " << totTime / nct << " secs\n";
    std::cerr << "Throughput: " << nct/totTime << " qps\n";
}
