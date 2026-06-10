// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**************************************************************************************************
 * This is code encodes the input datasets to the same format that RTScan uses for its experiments.
 *
 **************************************************************************************************/
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <thread>
#include <sys/stat.h>
#include <cstring>

// Global variables
const std::string PROJECT_DIR = "~/Device/IMPORTANT/raster-scan/";
int32_t g_dim = 3;
uint32_t g_npoints = uint32_t(50e6);  // Default: 50 million
std::string g_folder;
std::string g_opfolder;
std::vector<std::string> g_datasets = {
    "normal",
    "zipf1.5",
    "zipf1.3",
    "zipf1.1",
    "uniform",
};

std::vector<int32_t> sortData(const std::vector<uint32_t> vals) {
    std::vector<int32_t> ids(vals.size());
    for (int32_t i = 0; i < vals.size(); i++) {
        ids[i] = i;
    }
    std::sort(ids.begin(), ids.end(), [&vals](int32_t x, int32_t y) {
        return vals[x] < vals[y];
    });
    return ids;
}

std::vector<std::map<uint32_t, uint32_t>> encode(std::vector<std::vector<uint32_t>> &data) {
    int32_t ncols = (int32_t) data.size();
    std::vector<std::map<uint32_t, uint32_t>> rowMap(ncols);
    uint32_t npoints = (uint32_t) data[0].size();
    // encode each column
    for (int c = 0; c < ncols; c++) {
        std::cerr << "processing column " << c << std::endl;

        std::vector<uint32_t> colData = data[c];
        std::cerr << "sorting data" << std::endl;
        std::vector<int32_t> pos = sortData(colData);

        std::cerr << "generating counts for mapping" << std::endl;
        std::map<uint32_t,uint32_t> counts;
        for(uint32_t i = 0;i < npoints;i ++) {
            counts[colData[i]] ++;
        }
        std::cerr << "# unique values: " << counts.size() << std::endl;

        std::cerr << "mapping new positions..."  << std::endl;
        uint32_t st = 0;
        for (auto ct: counts) {
            rowMap[c][ct.first] = st;
            st += ct.second;
        }

        std::cerr << "assigning new positions..."  << std::endl;
        for(uint32_t i = 0;i < npoints;i ++) {
            data[c][pos[i]] = i;
        }
    }
    return rowMap;
}

void encodeDataset(const std::string& fileName, int ncols, uint32_t npoints, const std::string& opPrefix) {
    std::ifstream binfile(fileName, std::ios::binary|std::ios::ate);
    if(binfile.fail()) {
        std::cerr << "input file does not exist: " << fileName << "\n";
        exit(0);
    }
    size_t sizeInBytes = binfile.tellg();
    binfile.close();

    std::cerr << "input file size: " << sizeInBytes << "\n";
    size_t npts = sizeInBytes / (ncols * sizeof(uint32_t));
    if(npts != npoints) {
        std::cerr << "Error! mismatch in sizes: " << npts << " - " << npoints << "\n";
        exit(0);
    }
    std::vector<std::vector<uint32_t>> opoints(ncols);
    binfile.open(fileName, std::ios::binary);

    for(int c = 0;c < ncols;c ++) {
        opoints[c].resize(npoints);
        binfile.read((char*)opoints[c].data(), npoints * sizeof(uint32_t));
        if(!binfile) {
            std::cerr << "ERROR: all data not read from file:" << fileName << " - " << binfile.gcount() << "," << sizeInBytes << "\n";
            binfile.close();
            exit(0);
        }
    }
    binfile.close();

    std::vector<std::map<uint32_t, uint32_t>> rowMap = encode(opoints);

    std::vector<uint32_t> points;
    for(int c = 0;c < ncols;c ++) {
        points.insert(points.end(), opoints[c].begin(), opoints[c].end());
        std::cerr << "column size: " << c << "," << opoints[c].size() << "\n";
    }
    std::cerr << "points size: " << points.size() << "\n";
    {
        std::cerr << "writing encoded data \n";
        std::string encodedFile = opPrefix + "-data.bin";
        FILE *op = fopen(encodedFile.c_str(),"wb");
        fwrite(points.data(),sizeof(uint32_t),points.size(),op);
        fclose(op);
    }
    {
        std::cerr << "writing encoded map \n";
        std::string encodedFile = opPrefix + "-map.bin";
        FILE *op = fopen(encodedFile.c_str(),"wb");
        int32_t ncols = (int32_t) rowMap.size();
        fwrite(&ncols,sizeof(int32_t),1,op);
        std::cerr << "encoded map size: ";
        for(int32_t c = 0;c < ncols;c ++) {
            uint32_t mapsize = (uint32_t) rowMap[c].size();
            fwrite(&mapsize,sizeof(uint32_t),1,op);
            std::cerr << mapsize << ",";
            for(auto enc: rowMap[c]) {
                fwrite(&enc, sizeof(std::pair<uint32_t, uint32_t>),1,op);
            }
        }
        std::cerr << "\n";
        fclose(op);
    }
}

void encodeAllDatasets() {
    // Create output subfolder if it doesn't exist
    mkdir(g_opfolder.c_str(), 0755);

    std::cerr << "Encoding datasets from: " << g_folder << "\n";
    std::cerr << "Output folder: " << g_opfolder << "\n";
    std::cerr << "Dimensions: " << g_dim << ", Points: " << g_npoints << "\n";

    for(int i = 0; i < g_datasets.size(); i++) {
        std::cerr << "encoding " << g_folder + g_datasets[i] + ".bin" << "\n";
        encodeDataset(g_folder + g_datasets[i] + ".bin", g_dim, g_npoints, g_opfolder + g_datasets[i]);
    }
}

int main(int argc, char* argv[]) {
    // Default values
    int m = 50;  // millions of rows
    int c = 3;   // columns

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            c = atoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0] << " [-m millions] [-c columns]\n";
            std::cerr << "  -m: Number of millions of rows (default: 50)\n";
            std::cerr << "  -c: Number of columns (default: 3)\n";
            return 1;
        }
    }

    // Set global variables based on arguments
    g_dim = c;
    g_npoints = uint32_t(m) * 1000000;
    g_folder = PROJECT_DIR + "data/data_" + std::to_string(m) + "m_" + std::to_string(c) + "c/";
    g_opfolder = PROJECT_DIR + "encodedData/data_" + std::to_string(m) + "m_" + std::to_string(c) + "c/";

    encodeAllDatasets();
    return 0;
}
