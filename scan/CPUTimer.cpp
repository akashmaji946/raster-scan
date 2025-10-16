// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "CPUTimer.hpp"

#include <iostream>

CPUTimer::CPUTimer() {
    started = false;
}

void CPUTimer::start() {
    started = true;
    this->st = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
}

int64_t CPUTimer::stop() {
    if(!started) {
        return 0;
    }
    std::chrono::microseconds en = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
    started = false;
    return (en - st).count();
}
