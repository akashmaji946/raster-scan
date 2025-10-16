// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>

#include "RasterScan2D.hpp"
#include "CPUTimer.hpp"

#include <operators/OperatorCache.hpp>
#include <operators/SinglePassScan.hpp>
#include <core/vkutils.h>

#include <fstream>
#include <regex>
#include <cstring>
#include <thread>
#include <map>

using namespace vkcore;

#define STAGING_BUFFER_SIZE (16 * 1024 * 1024)

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
    PBuffer pointsBuffer(new Buffer(vd));
    pointsBuffer->create(npoints * ncols * sizeof(uint32_t),vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst,MemoryType::Internal);
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
    int ncols = 3;

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
            int cid = i % 3;
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



void queries3d(int dataId, PVkDevice vd, PBuffer staging, OperatorCache &op, bool test = false) {
    std::string opfolder = "E:/code/RasterVsRT/encodedData/";
    std::string qfolder = "E:/code/RTScan/test/";
    std::vector<std::string> datasets = {
        "normal_data_1e8_3",
        "zipf1.5_data_1e8_3",
        "zipf1.3_data_1e8_3",
        "zipf1.1_data_1e8_3",
        "uniform_data_1e8_3",
    };
    std::vector<std::string> querysets = {
        "normal.txt",
        "zipf1.5.txt",
        "zipf1.3.txt",
        "zipf1.1.txt",
        "scan_cmd_32_3c.txt",
    };
    std::vector<int> qct = {
        1,
        1,
        1,
        1,
        10
    };

    SinglePassScan *scan = (SinglePassScan *) op.getFunction(FunctionType::SinglePassScan);
    ReduceMax *reduce = (ReduceMax *) op.getFunction(FunctionType::ReduceMax);
    PBufferCache bufs(new CommonBufferPool(vd));

    std::cerr << "reading dataset: " << dataId << "\n";
    int32_t ncols;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<std::map<uint32_t, uint32_t>> rowMap;
    std::vector<uint32_t> points;
    PBuffer pointsBuffer = readEncodedData(opfolder + datasets[dataId],vd,staging,npoints,minval,maxval,rowMap,points,ncols);
    std::cerr << "finished reading data " << minval[0] << ";" << maxval[0] << ",  " << minval[1] << ";" << maxval[1] << ",  " << minval[2] << ";" << maxval[2] << "\n";

    RasterScan2D rs(vd, bufs, scan, reduce, ncols);

    std::cerr << "building index\n";
    PRasterIndex index = rs.buildIndex(pointsBuffer,npoints,minval.data(),maxval.data());
    std::cerr << "finished building index with max bin size: " << index->maxBinCt << "\n";
    std::cerr << "stats -- min: " << index->minVal[0] << ";" << index->minVal[1] << ", max: " << index->maxVal[0] << ";" << index->maxVal[1]  << ", range per bin: " << index->binRange[0]  << ";" << index->binRange[1]  << "\n";


    std::vector<uint32_t> targets;
    readQueries(qfolder + querysets[dataId], qct[dataId], targets, rowMap);

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


int main() {
    int devId = VkEngine::getEngine()->getDefaultDeviceId();
    PVkDevice vd = VkEngine::getEngine()->getDevice(devId);
    std::cerr << "\n\n**** using device " << vd->props.deviceName << " with ID = " << devId << " ****" << std::endl;
    PBuffer staging(new Buffer(vd));
    staging->create(STAGING_BUFFER_SIZE,vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst,MemoryType::ReadWrite);
    OperatorCache op;
    op.initialize(vd);

    // the experiments run for the paper
    int32_t nDataset = 5;
    for(int i = 0;i < nDataset;i ++) {
        queries3d(i,vd,staging,op);
    }

    return 0;
}
