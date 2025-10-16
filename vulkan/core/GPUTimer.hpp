// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VulkanDevice.hpp>

namespace vkcore {


class GPUTimer {
public:
    GPUTimer();

public:
    void start(PVkDevice vd);
    void stop();
    int64_t getTime();

protected:
    bool started,used;
    PVkDevice vd;
    int timerIndex;
};

} // namespace

