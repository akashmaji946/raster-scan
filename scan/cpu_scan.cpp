// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// CPU-based range query scanner for encoded data
// Executes queries on CPU and outputs matching rowIds to text files

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cstring>
#include <cmath>
#include <regex>
#include <sstream>
#include <chrono>
#include <cstdint>

// Global folder paths
const std::string PROJECT_DIR = "/home/akashmaji/Device/IMPORTANT/raster-scan/";

std::string g_opfolder;
std::string g_qfolder;
int32_t g_dim = 3;
uint32_t g_npoints = uint32_t(50e6);

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

std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::string s;
    s.append(1, delim);
    std::regex reg(s);
    std::vector<std::string> elems(std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
                                    std::sregex_token_iterator());
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

uint32_t transformQuery(int cid, uint32_t query, std::string &cmd, 
                        const std::vector<std::map<uint32_t, uint32_t>> &rowMap) {
    if (cmd == "lt") {
        auto it = rowMap[cid].lower_bound(query - 1);
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

void readQueries(std::string fileName, int nqueries, std::vector<uint32_t> &targets,
                 const std::vector<std::map<uint32_t, uint32_t>> &rowMap) {
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
            th = transformQuery(cid, th, cmds[0], rowMap);

            if (cmds[0] == "lt") {
                targets[i * 2] = 0;
                targets[i * 2 + 1] = th;
            }
        } else {
            printf("Error: No operand\n");
            exit(-1);
        }
    }
}

void readEncodedData(std::string prefix, uint32_t &npoints, 
                     std::vector<uint32_t> &minval, std::vector<uint32_t> &maxval,
                     std::vector<std::map<uint32_t, uint32_t>> &rowMap, 
                     std::vector<uint32_t> &points, int32_t &ncols) {
    // Read encodeMap
    {
        std::string fileName = prefix + "-map.bin";
        std::ifstream binfile(fileName, std::ios::binary);
        if (binfile.fail()) {
            std::cerr << "input file does not exist: " << fileName << "\n";
            exit(0);
        }
        binfile.read((char *)&ncols, sizeof(uint32_t));
        rowMap.resize(ncols);
        for (int c = 0; c < ncols; c++) {
            uint32_t mapsize;
            binfile.read((char *)&mapsize, sizeof(uint32_t));
            std::vector<std::pair<uint32_t, uint32_t>> encs(mapsize);
            binfile.read((char *)encs.data(), sizeof(std::pair<uint32_t, uint32_t>) * mapsize);
            for (auto enc : encs) {
                rowMap[c][enc.first] = enc.second;
            }
        }
    }
    
    // Read data
    {
        std::string fileName = prefix + "-data.bin";
        std::ifstream binfile(fileName, std::ios::binary | std::ios::ate);
        if (binfile.fail()) {
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
        if (!binfile) {
            std::cerr << "ERROR: all data not read from file:" << fileName << " - " 
                      << binfile.gcount() << "," << sizeInBytes << "\n";
            binfile.close();
            exit(0);
        }
        binfile.close();
    }

    minval.resize(ncols);
    maxval.resize(ncols);

    for (int c = 0; c < ncols; c++) {
        minval[c] = 0;
        maxval[c] = npoints;
    }
}

void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " [-m millions] [-c columns] [-t testfolder]\n";
    std::cerr << "  -m: Number of millions of rows (default: 50)\n";
    std::cerr << "  -c: Number of columns (default: 3)\n";
    std::cerr << "  -t: Test folder name (default: test)\n";
}

void queryCPU(int dataId) {
    std::cerr << "\n=================== CPU Query Execution ===================\n";
    std::cerr << "reading dataset: " << dataId << " (" << datasets[dataId] << ")\n";
    
    int32_t ncols;
    uint32_t npoints;
    std::vector<uint32_t> minval, maxval;
    std::vector<std::map<uint32_t, uint32_t>> rowMap;
    std::vector<uint32_t> points;
    
    readEncodedData(g_opfolder + datasets[dataId], npoints, minval, maxval, rowMap, points, ncols);
    std::cerr << "finished reading data " << minval[0] << ";" << maxval[0] << ",  " 
              << minval[1] << ";" << maxval[1] << ",  " << minval[2] << ";" << maxval[2] << "\n";

    std::vector<uint32_t> targets;
    readQueries(g_qfolder + querysets[dataId], qct[dataId], targets, rowMap);

    for (int i = 0; i < targets.size() / 2; i++) {
        if (i % 3 == 0) std::cerr << "Query: " << i / 3 << "\n";
        std::cerr << targets[i * 2] << "," << targets[i * 2 + 1] << "\n";
    }

    // Execute queries on CPU
    std::string outputDir = PROJECT_DIR + "compare/cpu_files/";
    std::string outputFile = outputDir + datasets[dataId] + "_results.txt";
    
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "ERROR: Cannot open output file: " << outputFile << "\n";
        return;
    }

    int totalMatches = 0;
    double totTime = 0;
    int nct = 0;

    for (int i = 0; i < qct[dataId]; i++) {
        int in = i * 6;
        // x1,y1,x2,y2,z1,z2
        uint32_t x1 = targets[in];
        uint32_t x2 = targets[in + 1];
        uint32_t y1 = targets[in + 2];
        uint32_t y2 = targets[in + 3];
        uint32_t z1 = targets[in + 4];
        uint32_t z2 = targets[in + 5];

        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<uint32_t> matchingRowIds;
        
        // CPU-based range query
        for (uint32_t j = 0; j < npoints; j++) {
            uint32_t pt[] = {points[j], points[j + npoints], points[j + 2 * npoints]};
            
            uint32_t sat = 1;
            for (int k = 0; k < 3; k++) {
                if (!(targets[in + k * 2] <= pt[k] && targets[in + k * 2 + 1] >= pt[k])) {
                    sat = 0;
                    break;
                }
            }
            
            if (sat) {
                matchingRowIds.push_back(j);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ts = std::chrono::duration<double>(end - start).count();
        totTime += ts;
        nct++;

        std::cerr << "Query " << (i + 1) << " [x:" << x1 << "-" << x2 << ", y:" << y1 << "-" << y2 
                  << ", z:" << z1 << "-" << z2 << "]: " << matchingRowIds.size() << " matches in " 
                  << ts << " secs\n";

        // Write results to file
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
    std::cerr << "Throughput: " << nct / totTime << " qps\n";
}

int main(int argc, char* argv[]) {
    // Default values
    int m = 50;  // millions of rows
    int c = 3;   // columns
    std::string testFolder = "test";

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            c = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            testFolder = argv[++i];
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

    std::cerr << "[Configuration] m=" << m << ", c=" << c << ", testFolder=" << testFolder << "\n";
    std::cerr << "[Configuration] Data folder: " << g_opfolder << "\n";
    std::cerr << "[Configuration] Test folder: " << g_qfolder << "\n";

    // Execute CPU queries for all datasets
    std::cerr << "\n*** Running CPU Queries ***\n";
    for (int i = 0; i < 5; i++) {
        queryCPU(i);
    }

    std::cerr << "\n*** CPU Query Execution Complete ***\n";
    return 0;
}
