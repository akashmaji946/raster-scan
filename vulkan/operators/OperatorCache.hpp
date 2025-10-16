// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ReduceMax.hpp"
#include "SinglePassScan.hpp"

#include <vector>

namespace vkcore {

enum class FunctionType : unsigned int {
    SinglePassScan = 0,
    ReduceMax
};

class OperatorCache
{
public:
    OperatorCache();
    ~OperatorCache();

public:
    void initialize(PVkDevice vd);
    void* getFunction(FunctionType type);

protected:
    void initFunctions(PVkDevice vd);

protected:
    SinglePassScan *spscan;
    ReduceMax *reduce;
};

}
