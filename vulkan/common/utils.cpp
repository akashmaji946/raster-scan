// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "utils.h"

#include <string>
#include <fstream>
#include <climits>

namespace vkcore {

void validate(bool suc, std::string msg) {
    if(!suc) {
        LOG << "ERROR: " << msg;
        exit(-1);
    }
}

bool readShader(std::string fileName, std::vector<uint32_t> &code) {
    std::ifstream binfile(fileName, std::ios::binary|std::ios::ate);
    if(binfile.fail()) {
        LOG << "shader file does not exist: " << fileName;
        exit(0);
    }
    size_t sizeInBytes = binfile.tellg();
    binfile.close();

    if(sizeInBytes > INT_MAX) {
        LOG << "shader size large. change read code: " << fileName;
        binfile.close();
        return false;
    }

    int64_t size = sizeInBytes / sizeof(uint32_t);
    code.resize(size);
    binfile.open(fileName, std::ios::binary);
    binfile.read((char*)code.data(), sizeInBytes);
    if(!binfile) {
        LOG << "ERROR: all data not read from shader file:" << fileName << " - " << binfile.gcount() << "," << sizeInBytes;
        binfile.close();
        return false;
    }
    binfile.close();
    return true;
}

} //namespace
