// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "OperatorCache.hpp"

#include <iostream>

namespace vkcore {

OperatorCache::OperatorCache() {
}

OperatorCache::~OperatorCache() {
    delete reduce;
    delete spscan;
}

void OperatorCache::initialize(PVkDevice vd) {
    this->initFunctions(vd);
}

void *OperatorCache::getFunction(FunctionType type) {
    switch(type) {
    case FunctionType::SinglePassScan:
        return (void *)(spscan);
        break;

    case FunctionType::ReduceMax:
        return (void *)(reduce);
        break;

    default:
        std::cerr << "function not supported" << std::endl;
    }
    return NULL;
}

void OperatorCache::initFunctions(PVkDevice vd) {
    this->spscan = new SinglePassScan(vd);
    this->spscan->initialize();

    this->reduce = new ReduceMax(vd);
    this->reduce->initialize();
}

}

