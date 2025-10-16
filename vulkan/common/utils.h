// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <iostream>
#include <vector>

struct X {
  ~X() { std::cerr << std::endl; }
};

#define LOG (X(), std::cerr)

namespace vkcore {

void validate(bool suc, std::string msg);
bool readShader(std::string fileName, std::vector<uint32_t> &code);

}
