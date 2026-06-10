// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>

#include "RasterScan2D.hpp"
#include "RasterScanIndexUpdate.hpp"
#include "GPUMemoryTool.hpp"
#include "CPUTimer.hpp"

#include <operators/OperatorCache.hpp>
#include <operators/SinglePassScan.hpp>
#include <core/vkutils.h>

#include <fstream>
#include <regex>
#include <cstring>
#include <thread>
#include <cmath>
#include <map>
#include <iomanip>
#include <random>
#include <array>

using namespace vkcore;

#define STAGING_BUFFER_SIZE (16 * 1024 * 1024)



// Flag to select pipeline:
// 0 = Original RasterScan2D pipeline
// 1 = New RasterScanIndexUpdate pipeline (naive linked list)
// 2 = Compare both pipelines above
// 3 = Test dynamic operations (rowId-based delete)
// 4 = Test dynamic updated (range delete operations)
// 5 = GPU query execution (output results to text files)
// 6 = Bitmap-based free space management stress test (insert/delete cycles)
// 7 = Delete-by-data performance test
#define USE_INDEX_UPDATE_PIPELINE 7

// Global folder paths for data and test files
const std::string PROJECT_DIR = "~/Device/IMPORTANT/raster-scan/";

std::string g_opfolder;
std::string g_qfolder;
int32_t g_dim = 3;
uint32_t g_npoints = uint32_t(50e6);  // Default: 50 million
// the experiments run for the paper
int32_t nDataset = 5;

std::vector<std::string> datasets = {
    "normal",
    "zipf1.5",
    "zipf1.3",
    "zipf1.1",
    "uniform",
};
std::vector<std::string> querysets = {
    "normal.txt",
    "zipf1.5.txt",
    "zipf1.3.txt",
    "zipf1.1.txt",
    "uniform.txt",
};
std::vector<int> qct = {
    1,
    1,
    1,
    1,
    10
};

std::vector<uint32_t> generateQueries(uint32_t nqueries) {
    std::vector<uint32_t> queries;
    for(uint32_t i = 0;i < nqueries;i ++) {
        uint32_t q = rand();
        queries.push_back(q);
    }
    return queries;
}

PBuffer readEncodedData(std::string prefix, PVkDevice vd, PBuffer staging, uint32_t &npoints, std::vector<uint32_t> &minval, std::vector<uint32_t> &maxval,
                 std::vector<std::map<uint32_t, uint32_t>> &rowMap, std::vector<uint32_t> &points, int32_t &ncols) {
    // read encodeMap
    {
        std::string fileName = prefix + "-map.bin";
        std::ifstream binfile(fileName, std::ios::binary);
        if(binfile.fail()) {
            std::cerr << "input file does not exist: " << fileName << "\n";
            exit(0);
        }
        binfile.read((char *)&ncols,sizeof(uint32_t));
        rowMap.resize(ncols);
        for(int c = 0;c < ncols;c ++) {
            uint32_t mapsize;
            binfile.read((char *)&mapsize,sizeof(uint32_t));
            std::vector<std::pair<uint32_t,uint32_t>> encs(mapsize);
            binfile.read((char *)encs.data(),sizeof(std::pair<uint32_t,uint32_t>) * mapsize);
            for(auto enc : encs) {
                rowMap[c][enc.first] = enc.second;
            }
        }
    }
    // read data
    {
        std::string fileName = prefix + "-data.bin";
        std::ifstream binfile(fileName, std::ios::binary|std::ios::ate);
        if(binfile.fail()) {
            std::cerr << "input file does not exist: " << fileName << "\n";
            exit(0);
        }
        size_t sizeInBytes = binfile.tellg();
        binfile.close();
        npoints = uint32_t(sizeInBytes / (ncols * sizeof(uint32_t)));
        std::cerr << "input file size: " << sizeInBytes << " with " << npoints << " points\n";

        points.resize(npoints * ncols);
        binfile.open(fileName, std::ios::binary);
        binfile.read((char*)points.data(), points.size() * sizeof(uint32_t));
        if(!binfile) {
            std::cerr << "ERROR: all data not read from file:" << fileName << " - " << binfile.gcount() << "," << sizeInBytes << "\n";
            binfile.close();
            exit(0);
        }
        binfile.close();
    }

    minval.resize(ncols);
    maxval.resize(ncols);

    for(int c = 0;c < ncols;c ++) {
        minval[c] = 0;
        maxval[c] = npoints;
    }
    size_t pointsBufferSize = static_cast<size_t>(npoints) * ncols * sizeof(uint32_t);
    PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(pointsBufferSize,vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst,MemoryType::Internal);
    GPU_MEM_PRINT("Points Buffer", pointsBufferSize);
    GPUMemoryTool::printGPUMemoryStatus(vd, "After points buffer allocation");
    loadUsingStagingBuf((char *)points.data(), points.size() * sizeof(uint32_t),pointsBuffer,staging,vd,0);
    return pointsBuffer;
}

std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::string s;
    s.append(1, delim);
    std::regex reg(s);
    std::vector<std::string> elems(std::sregex_token_iterator(str.begin(), str.end(), reg, -1),std::sregex_token_iterator());
    return elems;
}

std::vector<uint32_t> get_target_numbers(std::string s) {
    std::stringstream ss(s);
    std::string value;
    std::vector<uint32_t> result;
    while (std::getline(ss, value, ',')) {
        result.push_back((uint32_t)stod(value));
    }
    return result;
}

uint32_t transformQuery(int cid, uint32_t query, std::string &cmd, const std::vector<std::map<uint32_t, uint32_t>> &rowMap) {
    // queries used in the paper are present in: https://github.com/AntaresAlice/RTScan/tree/main/test
    // Since it contains only *less than* queries, we perform query transformation only for this case.
    // *greater than* can be accomplished in a similar manner.
    if (cmd == "lt") {
        // RasterScan supports only LE, so converting LT to LE
        auto it = rowMap[cid].lower_bound(query-1);
        if (it == rowMap[cid].end()) {
            return uint32_t(-1);
        }
        return it->second;
    } else if (cmd == "le") {
        auto it = rowMap[cid].upper_bound(query);
        if (it == rowMap[cid].end()) {
            return uint32_t(-1);
        }
        return it->second - 1;
    } else {
        printf("incorrect encode command.\n");
        exit(-1);
    }
}

// read ther queries used by RTScan experiments.
void readQueries(std::string fileName, int nqueries, std::vector<uint32_t> &targets, const std::vector<std::map<uint32_t, uint32_t>> &rowMap) {
    int ncols = g_dim;

    std::ifstream fin(fileName);
    if (!fin.is_open()) {
        std::cerr << "Fail to open FILE " << fileName << std::endl;
        exit(-1);
    }
    targets.resize(2 * ncols * nqueries);
    for (int i = 0; i < ncols * nqueries; i++) {
        std::string input;
        std::getline(fin, input);
        std::vector<std::string> cmds = stringSplit(input, ' ');
        if (cmds[0] == "exit") exit(0);
        if (cmds.size() > 1) {
            uint32_t th = get_target_numbers(cmds[1])[0];
            int cid = i % g_dim;
            th = transformQuery(cid,th,cmds[0],rowMap);

            // RTScan queries only had "lt". The thresholds can be set in a similar manner for other types of comparisons
            if(cmds[0] == "lt") {
                targets[i * 2] = 0;
                targets[i * 2 + 1] = th;
            }
        } else {
            printf("Error: No operand\n");
            exit(-1);
        }
    }
}

void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " [-m millions] [-c columns] [-t testfolder] [-g GPU]\n";
    std::cerr << "  -m: Number of millions of rows (default: 50)\n";
    std::cerr << "  -c: Number of columns (default: 3)\n";
    std::cerr << "  -t: Test folder name (default: test)\n";
    std::cerr << "  -g: GPU vendor (A=AMD, N=NVIDIA, D=Default, default: D)\n";
}



void queries3d(int dataId, PVkDevice vd, PBuffer staging, OperatorCache &op, bool test = false) {
    

    SinglePassScan *scan = (SinglePassScan *) op.getFunction(FunctionType::SinglePassScan);
    ReduceMax *reduce = (ReduceMax *) op.getFunction(FunctionType::ReduceMax);
    PBufferCache bufs(new CommonBufferPool(vd));

    std::cerr << "reading dataset: " << dataId << "\n";
    int32_t ncols;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<std::map<uint32_t, uint32_t>> rowMap;
    std::vector<uint32_t> points;
    PBuffer pointsBuffer = readEncodedData(g_opfolder + datasets[dataId],vd,staging,npoints,minval,maxval,rowMap,points,ncols);
    std::cerr << "finished reading data " << minval[0] << ";" << maxval[0] << ",  " << minval[1] << ";" << maxval[1] << ",  " << minval[2] << ";" << maxval[2] << "\n";

    RasterScan2D rs(vd, bufs, scan, reduce, ncols);

    std::cerr << "building index\n";
    PRasterIndex index = rs.buildIndex(pointsBuffer,npoints,minval.data(),maxval.data());
    std::cerr << "finished building index with max bin size: " << index->maxBinCt << "\n";
    std::cerr << "stats -- min: " << index->minVal[0] << ";" << index->minVal[1] << ", max: " << index->maxVal[0] << ";" << index->maxVal[1]  << ", range per bin: " << index->binRange[0]  << ";" << index->binRange[1]  << "\n";

    std::vector<uint32_t> targets;
    readQueries(g_qfolder + querysets[dataId], qct[dataId], targets, rowMap);

    for(int i = 0;i < targets.size()/2;i ++) {
        if(i % 3 == 0) std::cerr << "Query: " << i / 3 << "\n";
        std::cerr << targets[i * 2] << "," << targets[i * 2 + 1] << "\n";
    }

    // Execute queries.
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
        rs.runRangeQueries(index,queryBuffer,1);
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


void queries3dIndexUpdate(int dataId, PVkDevice vd, PBuffer staging, bool test = false) {


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
    std::cerr << "building linked list index\n";
    CPUTimer buildTimer;
    buildTimer.start();
    PLinkedListIndex index = rsUpdate.buildIndex(pointsBuffer, npoints, minval.data(), maxval.data());
    uint64_t buildTime = buildTimer.stop();
    std::cerr << "finished building linked list index in " << double(buildTime) / 1000000. << " secs\n";
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


// Compare results from both pipelines
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
    std::cerr << "\nBuilding RasterScan2D index...\n";
    CPUTimer buildTimer1;
    buildTimer1.start();
    PRasterIndex index1 = rs.buildIndex(pointsBuffer, npointsLimited, minval.data(), maxval.data());
    double buildTime1 = double(buildTimer1.stop()) / 1000000.;
    std::cerr << "RasterScan2D build time: " << buildTime1 << " secs\n";
    GPUMemoryTool::printGPUMemoryStatus(vd, "After RasterScan2D build");

    // Setup for RasterScanIndexUpdate (new)
    // Note: PageAllocator is created lazily in buildIndex() to defer memory allocation
    PBufferCache bufs2(new CommonBufferPool(vd));
    RasterScanIndexUpdate rsUpdate(vd, bufs2, ncols);
    
    std::cerr << "Building RasterScanIndexUpdate index...\n";
    CPUTimer buildTimer2;
    buildTimer2.start();
    PLinkedListIndex index2 = rsUpdate.buildIndex(pointsBuffer, npointsLimited, minval.data(), maxval.data());
    double buildTime2 = double(buildTimer2.stop()) / 1000000.;
    std::cerr << "RasterScanIndexUpdate build time: " << buildTime2 << " secs\n";

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

// Test dynamic index operations (insert, delete, update)
void testDynamicOperations(PVkDevice vd, PBuffer staging, const std::string& datasetName) {

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

// Test range-based delete operations - loads from test/dynamic/{datasetName}_range/
void testRangeDelete(PVkDevice vd, PBuffer staging, const std::string& datasetName) {
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

// GPU Query Execution with output to text files (mode=5)
void queriesGPUWithOutput(int dataId, PVkDevice vd, PBuffer staging, OperatorCache &op, bool test = false) {
    
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

// ============================================================================
// Mode 6: Bitmap-based Free Space Management Stress Test
// ============================================================================
// This test performs cycles of insert/delete operations to test:
// 1. Initial index build
// 2. Delete operations (marking pages as deleted)
// 3. Mark deleted pages as free in bitmap
// 4. Compact pages (remove deleted pages from linked lists)
// 5. Insert new data using bitmap-based allocation (reusing freed pages)
// 6. Repeat cycles and measure timing

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
    const uint32_t INITIAL_POINTS = 10000000;  // 10M initial points
    const uint32_t DELETE_BATCH = 1000000;     // 2M points to delete per cycle
    const uint32_t INSERT_BATCH = 1000000;     // 2M points to insert per cycle
    const uint32_t NUM_CYCLES = 2;             // Number of insert/delete cycles
    const uint32_t VALUE_RANGE = 10000;        // Range of x, y, z values
    
    std::cerr << "[Config] Initial points: " << INITIAL_POINTS << "\n";
    std::cerr << "[Config] Delete batch size: " << DELETE_BATCH << "\n";
    std::cerr << "[Config] Insert batch size: " << INSERT_BATCH << "\n";
    std::cerr << "[Config] Number of cycles: " << NUM_CYCLES << "\n";
    std::cerr << "[Config] Value range: 0-" << VALUE_RANGE << "\n";
    std::cerr << "\n";
    
    // Random number generator
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<uint32_t> valueDist(0, VALUE_RANGE - 1);
    
    // Generate initial data (column-major format)
    std::cerr << "[Phase 0] Generating " << INITIAL_POINTS << " random points...\n";
    CPUTimer genTimer;
    genTimer.start();
    
    std::vector<uint32_t> initialPoints(INITIAL_POINTS * 3);
    for (uint32_t i = 0; i < INITIAL_POINTS; i++) {
        initialPoints[i] = valueDist(rng);                          // x
        initialPoints[INITIAL_POINTS + i] = valueDist(rng);         // y
        initialPoints[2 * INITIAL_POINTS + i] = valueDist(rng);     // z
    }
    
    double genTime = double(genTimer.stop()) / 1000000.0;
    std::cerr << "[Phase 0] Data generation time: " << genTime << " secs\n";
    
    // Create points buffer
    PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(INITIAL_POINTS * 3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::Internal);
    loadUsingStagingBuf((char*)initialPoints.data(), INITIAL_POINTS * 3 * sizeof(uint32_t), 
                        pointsBuffer, staging, vd, 0);
    
    // Min/max values
    uint32_t minVal[3] = {0, 0, 0};
    uint32_t maxVal[3] = {VALUE_RANGE, VALUE_RANGE, VALUE_RANGE};
    
    // Create index
    PBufferCache bufs(new CommonBufferPool(vd));
    RasterScanIndexUpdate rsUpdate(vd, bufs, 3);
    
    // Phase 1: Build initial index
    std::cerr << "\n[Phase 1] Building initial index with " << INITIAL_POINTS << " points...\n";
    CPUTimer buildTimer;
    buildTimer.start();
    PLinkedListIndex index = rsUpdate.buildIndex(pointsBuffer, INITIAL_POINTS, minVal, maxVal);
    double buildTime = double(buildTimer.stop()) / 1000000.0;
    std::cerr << "[Phase 1] Index build time: " << buildTime << " secs\n";
    
    // Initialize bitmap with the initial allocation count
    // First INITIAL_POINTS pages are allocated, rest are free
    rsUpdate.initializeBitmapWithAllocation(INITIAL_POINTS);
    
    // Get initial stats
    auto stats = rsUpdate.getAllocationStats(index);
    printAllocationStats(stats, "[Phase 1] Initial allocation stats");
    
    // Track current row ID offset for inserts
    uint32_t currentRowIdOffset = INITIAL_POINTS;
    
    // Track valid (non-deleted) row IDs for unique deletes
    std::vector<uint32_t> validRowIds(INITIAL_POINTS);
    for (uint32_t i = 0; i < INITIAL_POINTS; i++) {
        validRowIds[i] = i;
    }
    uint32_t expectedAllocated = INITIAL_POINTS;
    
    // Run insert/delete cycles
    for (uint32_t cycle = 0; cycle < NUM_CYCLES; cycle++) {
        std::cerr << "\n";
        std::cerr << "========== CYCLE " << (cycle + 1) << "/" << NUM_CYCLES << " ==========\n";
        
        // Phase 2: Delete unique rows (no duplicates)
        uint32_t actualDeleteCount = std::min(DELETE_BATCH, (uint32_t)validRowIds.size());
        std::cerr << "[Cycle " << (cycle + 1) << "] Deleting " << actualDeleteCount << " unique rows...\n";
        
        // Shuffle validRowIds and take the first actualDeleteCount as deletes
        std::shuffle(validRowIds.begin(), validRowIds.end(), rng);
        std::vector<uint32_t> deleteRowIds(validRowIds.begin(), validRowIds.begin() + actualDeleteCount);
        
        // Remove deleted IDs from validRowIds
        validRowIds.erase(validRowIds.begin(), validRowIds.begin() + actualDeleteCount);
        expectedAllocated -= actualDeleteCount;
        std::cerr << "[Cycle " << (cycle + 1) << "] Expected allocated after delete: " << expectedAllocated << "\n";
        
        // Create delete buffer
        PBuffer deleteBuffer(new Buffer(vd));
        deleteBuffer->create(actualDeleteCount * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            MemoryType::Internal);
        loadUsingStagingBuf((char*)deleteRowIds.data(), actualDeleteCount * sizeof(uint32_t), 
                            deleteBuffer, staging, vd, 0);
        
        CPUTimer deleteTimer;
        deleteTimer.start();
        rsUpdate.deletePoints(index, deleteBuffer, actualDeleteCount);
        double deleteTime = double(deleteTimer.stop()) / 1000000.0;
        std::cerr << "[Cycle " << (cycle + 1) << "] Delete time: " << deleteTime << " secs\n";
        
        deleteBuffer->destroy();
        deleteBuffer.reset();
        
        // Phase 3: Mark deleted pages as free
        std::cerr << "[Cycle " << (cycle + 1) << "] Marking deleted pages as free...\n";
        CPUTimer markFreeTimer;
        markFreeTimer.start();
        rsUpdate.markDeletedPagesAsFree(index);
        double markFreeTime = double(markFreeTimer.stop()) / 1000000.0;
        std::cerr << "[Cycle " << (cycle + 1) << "] Mark-free time: " << markFreeTime << " secs\n";
        
        stats = rsUpdate.getAllocationStats(index);
        printAllocationStats(stats, "[Cycle " + std::to_string(cycle + 1) + "] After mark-free");
        
        // Phase 4: Compact pages (remove deleted from linked lists)
        std::cerr << "[Cycle " << (cycle + 1) << "] Compacting pages...\n";
        CPUTimer compactTimer;
        compactTimer.start();
        rsUpdate.compactPages(index);
        double compactTime = double(compactTimer.stop()) / 1000000.0;
        std::cerr << "[Cycle " << (cycle + 1) << "] Compact time: " << compactTime << " secs\n";
        
        stats = rsUpdate.getAllocationStats(index);
        printAllocationStats(stats, "[Cycle " + std::to_string(cycle + 1) + "] After compact");
        
        // Phase 5: Insert new data using bitmap allocation
        std::cerr << "[Cycle " << (cycle + 1) << "] Inserting " << INSERT_BATCH << " new points using bitmap allocation...\n";
        
        // Generate new random points
        std::vector<uint32_t> newPoints(INSERT_BATCH * 3);
        for (uint32_t i = 0; i < INSERT_BATCH; i++) {
            newPoints[i] = valueDist(rng);
            newPoints[INSERT_BATCH + i] = valueDist(rng);
            newPoints[2 * INSERT_BATCH + i] = valueDist(rng);
        }
        
        // Create insert buffer
        PBuffer insertBuffer(new Buffer(vd));
        insertBuffer->create(INSERT_BATCH * 3 * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            MemoryType::Internal);
        loadUsingStagingBuf((char*)newPoints.data(), INSERT_BATCH * 3 * sizeof(uint32_t), 
                            insertBuffer, staging, vd, 0);
        
        CPUTimer insertTimer;
        insertTimer.start();
        rsUpdate.insertPointsWithBitmap(index, insertBuffer, INSERT_BATCH, currentRowIdOffset);
        double insertTime = double(insertTimer.stop()) / 1000000.0;
        std::cerr << "[Cycle " << (cycle + 1) << "] Insert time: " << insertTime << " secs\n";
        
        // Add new row IDs to validRowIds
        for (uint32_t i = 0; i < INSERT_BATCH; i++) {
            validRowIds.push_back(currentRowIdOffset + i);
        }
        currentRowIdOffset += INSERT_BATCH;
        expectedAllocated += INSERT_BATCH;
        std::cerr << "[Cycle " << (cycle + 1) << "] Expected allocated after insert: " << expectedAllocated << "\n";
        
        insertBuffer->destroy();
        insertBuffer.reset();
        
        stats = rsUpdate.getAllocationStats(index);
        printAllocationStats(stats, "[Cycle " + std::to_string(cycle + 1) + "] After insert");
        
        // Phase 6: Run a test query to verify index integrity
        std::cerr << "[Cycle " << (cycle + 1) << "] Running test query...\n";
        
        PBuffer queryBuffer(new Buffer(vd));
        queryBuffer->create(6 * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            MemoryType::Internal);
        
        // Query: select all points where x,y,z in [0, VALUE_RANGE/2]
        std::vector<uint32_t> queryData = {0, 0, VALUE_RANGE/2, VALUE_RANGE/2, 0, VALUE_RANGE/2};
        loadUsingStagingBuf((char*)queryData.data(), 6 * sizeof(uint32_t), queryBuffer, staging, vd, 0);
        
        CPUTimer queryTimer;
        queryTimer.start();
        rsUpdate.runRangeQueries(index, queryBuffer, 1);
        double queryTime = double(queryTimer.stop()) / 1000000.0;
        
        // Count results
        uint32_t arrsize = (currentRowIdOffset + 31) / 32;
        std::vector<uint32_t> results(arrsize);
        readUsingStagingBuf((char*)results.data(), arrsize * sizeof(uint32_t), bufs->resBuffer, staging, vd);
        
        uint32_t matchCount = 0;
        for (uint32_t j = 0; j < arrsize; j++) {
            matchCount += __builtin_popcount(results[j]);
        }
        
        std::cerr << "[Cycle " << (cycle + 1) << "] Query time: " << queryTime << " secs, matches: " << matchCount << "\n";
        
        queryBuffer->destroy();
        queryBuffer.reset();
        
        std::cerr << "[Cycle " << (cycle + 1) << "] Cycle summary:\n";
        std::cerr << "          - Delete: " << deleteTime << " secs\n";
        std::cerr << "          - Mark-free: " << markFreeTime << " secs\n";
        std::cerr << "          - Compact: " << compactTime << " secs\n";
        std::cerr << "          - Insert (bitmap): " << insertTime << " secs\n";
        std::cerr << "          - Query: " << queryTime << " secs\n";
        std::cerr << "          - Total cycle time: " << (deleteTime + markFreeTime + compactTime + insertTime + queryTime) << " secs\n";
    }
    
    // Final summary
    std::cerr << "\n";
    std::cerr << "================================================================================\n";
    std::cerr << "  FINAL SUMMARY\n";
    std::cerr << "================================================================================\n";
    
    stats = rsUpdate.getAllocationStats(index);
    printAllocationStats(stats, "Final allocation stats");
    std::cerr << "  - Initial build time: " << buildTime << " secs\n";
    std::cerr << "  - Cycles completed: " << NUM_CYCLES << "\n";
    std::cerr << "\n";
    std::cerr << "Bitmap-based free space management test COMPLETE!\n";
    std::cerr << "================================================================================\n\n";
    
    // Cleanup
    pointsBuffer->destroy();
    pointsBuffer.reset();
}

// ============================================================================
// Mode 7: Delete-by-Data Test
// ============================================================================
void testDeleteByData(PVkDevice vd, PBuffer staging) {
    std::cerr << "\n================================================================================\n";
    std::cerr << "  MODE 7: DELETE-BY-DATA BATCH CYCLE TEST\n";
    std::cerr << "================================================================================\n\n";
    
    // Configuration
    // We need: NUM_POINTS + NUM_BATCHES * BATCH_SIZE <= MAX_PAGES
    // With BATCH_SIZE = NUM_POINTS / NUM_BATCHES, total pages = NUM_POINTS * 2
    // So NUM_POINTS <= MAX_PAGES / 2, with some margin for safety
    const uint32_t NUM_BATCHES = 10;         
    const uint32_t NUM_POINTS = std::min(32000000u, (uint32_t)(MAX_PAGES * 5 / 10));  
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
    std::cerr << "Delete-by-data batch cycle test COMPLETE!\n";
    std::cerr << "================================================================================\n\n";
    
    // Cleanup
    batchBuffer->destroy();
    batchBuffer.reset();
    pointsBuffer->destroy();
    pointsBuffer.reset();
}

// Helper function to select GPU by vendor
int selectGPUByVendor(char vendor) {
    VkEngine* engine = VkEngine::getEngine();
    int devId = -1;
    
    std::cerr << "[selectGPUByVendor] Starting device enumeration...\n";
    std::cerr.flush();
    
    // Collect device info before processing
    struct DeviceInfo {
        int id;
        std::string name;
        vk::DeviceSize heapSize;
    };
    std::vector<DeviceInfo> devices;
    
    for(int i = 0; i < 3; i++) {  // Only try first 3 devices
        try {
            std::cerr << "[selectGPUByVendor] Trying device " << i << "...\n";
            std::cerr.flush();
            PVkDevice dev = engine->getDevice(i);
            if(dev) {
                std::string name = dev->props.deviceName;
                vk::DeviceSize heap = dev->heapSize;
                std::cerr << "[selectGPUByVendor] Found device " << i << ": " << name << " (heap: " << (heap / (1024*1024*1024)) << " GB)\n";
                std::cerr.flush();
                devices.push_back({i, name, heap});
            }
        } catch(...) {
            std::cerr << "[selectGPUByVendor] Exception at device " << i << ", stopping enumeration\n";
            std::cerr.flush();
            break;
        }
    }
    
    std::cerr << "[selectGPUByVendor] Found " << devices.size() << " devices\n";
    std::cerr.flush();
    
    std::cerr << "[selectGPUByVendor] Processing vendor: " << vendor << "\n";
    std::cerr.flush();
    
    switch(vendor) {
        case 'N': // NVIDIA
        case 'n': {
            // Find NVIDIA GPU
            for(const auto& dev : devices) {
                if(dev.name.find("NVIDIA") != std::string::npos) {
                    devId = dev.id;
                    break;
                }
            }
            break;
        }
        case 'A': // AMD
        case 'a': {
            // Find AMD GPU
            for(const auto& dev : devices) {
                if(dev.name.find("AMD") != std::string::npos) {
                    devId = dev.id;
                    break;
                }
            }
            break;
        }
        case 'D': // Default - use default discrete GPU
        case 'd':
        default:
            devId = engine->getDefaultDeviceId();
    }
    
    if(devId == -1) {
        std::cerr << "Warning: GPU vendor '" << vendor << "' not found, using default device " << devId << "\n";
        std::cerr.flush();
    }
    
    std::cerr << "[selectGPUByVendor] Returning device ID: " << devId << "\n";
    std::cerr.flush();
    
    return devId;
}

int main(int argc, char* argv[]) {
    // Default values
    int m = 50;  // millions of rows
    int c = 3;   // columns
    std::string testFolder = "test";
    char gpuVendor = 'D';  // Default

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            c = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            testFolder = argv[++i];
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            gpuVendor = argv[++i][0];
        } else {
            printUsage(argv[0]);
            return 1;
        }
    }

    // Set global variables based on arguments
    g_dim = c;
    g_npoints = uint32_t(m) * 1000000;
    g_opfolder = PROJECT_DIR + "encodedData/data_" + std::to_string(m) + "m_" + std::to_string(c) + "c/";
    g_qfolder = PROJECT_DIR + "tests/" + testFolder + "/";

    std::cerr << "[Configuration] m=" << m << ", c=" << c << ", testFolder=" << testFolder << ", gpuVendor=" << gpuVendor << "\n";
    std::cerr << "[Configuration] Data folder: " << g_opfolder << "\n";
    std::cerr << "[Configuration] Test folder: " << g_qfolder << "\n";

    // Parse command-line arguments for GPU selection
    int devId = -1;
    std::cerr << "[GPU Selection] Using -g " << gpuVendor << "\n";
    std::cerr.flush();
    devId = selectGPUByVendor(gpuVendor);
    
    // If no GPU specified, use default
    if(devId == -1) {
        devId = VkEngine::getEngine()->getDefaultDeviceId();
    }
    
    std::cerr << "[GPU Selection] Selected device ID: " << devId << "\n";
    std::cerr.flush();
    
    PVkDevice vd = VkEngine::getEngine()->getDevice(devId);

    std::cerr << "\n\n**** using device " << vd->props.deviceName << " with ID = " << devId << " ****" << std::endl;
    
    // Print GPU memory info (calculated)
    uint64_t gpuMemory = vd->heapSize;
    std::cerr << "[GPU Info] Device: " << vd->props.deviceName << std::endl;
    std::cerr << "[GPU Info] VRAM:   " << GPUMemoryTool::formatBytes(gpuMemory) << std::endl;
    std::cerr << "[GPU Info] MAX_PAGES: " << MAX_PAGES << " (" << GPUMemoryTool::formatBytes(4ll * MAX_PAGES * PAGE_SIZE_UINTS ) << " page buffer)" << std::endl;
    
    // Query actual GPU memory usage from VMA
    GPUMemoryTool::printGPUMemoryUsage(vd);
    GPUMemoryTool::printGPUMemoryStatus(vd, "Initial state");
    
    PBuffer staging(new Buffer(vd));
    staging->create(STAGING_BUFFER_SIZE,vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst,MemoryType::ReadWrite);
    GPU_MEM_PRINT("Staging Buffer", STAGING_BUFFER_SIZE);
    GPUMemoryTool::printGPUMemoryStatus(vd, "After staging buffer");
    
    OperatorCache op;
    op.initialize(vd);
    GPUMemoryTool::printGPUMemoryStatus(vd, "After OperatorCache init");

    
#if USE_INDEX_UPDATE_PIPELINE == 7
    std::cerr << "\n*** Delete-by-Data Performance Test ***\n";
    testDeleteByData(vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 6
    std::cerr << "\n*** Bitmap-based Free Space Management Stress Test ***\n";
    testBitmapFreeSpaceManagement(vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 5
    std::cerr << "\n*** GPU Query Execution with Output to Files ***\n";
    for(int i = 0; i < nDataset; i++) {
        queriesGPUWithOutput(i, vd, staging, op, true);
    }
#elif USE_INDEX_UPDATE_PIPELINE == 4
    std::cerr << "\n*** Testing Range Delete Operations ***\n";
    testRangeDelete(vd, staging, "1M");
    testRangeDelete(vd, staging, "10M");
    testRangeDelete(vd, staging, "100M");
#elif USE_INDEX_UPDATE_PIPELINE == 3
    std::cerr << "\n*** Testing Dynamic Index Operations ***\n";
    testDynamicOperations(vd, staging, "1M");
    testDynamicOperations(vd, staging, "10M");
    testDynamicOperations(vd, staging, "100M");
#elif USE_INDEX_UPDATE_PIPELINE == 2
    std::cerr << "\n*** COMPARING Both Pipelines ***\n";
    for(int i = 0; i < nDataset; i++) {
        compareResults(i, vd, staging, op);
    }
#elif USE_INDEX_UPDATE_PIPELINE == 1
    std::cerr << "\n*** Running NEW RasterScanIndexUpdate (Linked List) Pipeline ***\n";
    for(int i = 0; i < nDataset; i++) {
        queries3dIndexUpdate(i, vd, staging, true);
    }
#else
    std::cerr << "\n*** Running ORIGINAL RasterScan2D Pipeline ***\n";
    for(int i = 0; i < nDataset; i++) {
        queries3d(i, vd, staging, op, true);
    }
#endif

    return 0;
}
