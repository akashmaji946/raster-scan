// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <chrono>

class CPUTimer {
public:
    CPUTimer();

public:
    void start();
    int64_t stop();

protected:
    std::chrono::microseconds st;
    bool started;
};

