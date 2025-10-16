// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "GPUTimer.hpp"

#include <common/utils.h>

namespace vkcore {

GPUTimer::GPUTimer() {
    started = false;
    used = false;
}

void GPUTimer::start(PVkDevice vd) {
    started = true;
    used = true;
    this->vd = vd;
    timerIndex = vd->getNextTimer();
    vd->commandBuffer->resetQueryPool(vd->timerPool.get(),timerIndex * 2,2);
    vd->commandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, vd->timerPool.get(), timerIndex * 2);
}

void GPUTimer::stop() {
    vd->commandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, vd->timerPool.get(), timerIndex * 2+1);
    started = false;
}

inline uint64_t toMicroSeconds(uint64_t gpuTime, float period) {
    return uint64_t((gpuTime * period) / 1000);
}

int64_t GPUTimer::getTime() {
    if(!used) {
        return 0;
    }
    if(started) {
        std::cerr << "GPU getTime Error!! should not come here" << std::endl;
        exit(-2);
    }
    uint64_t tbuf[2];
    int64_t ret = -1;
    vk::Result result = vd->device->getQueryPoolResults(vd->timerPool.get(), timerIndex * 2, 2, sizeof(uint64_t) * 2, tbuf, sizeof(uint64_t), vk::QueryResultFlagBits::e64);
    if (result == vk::Result::eNotReady) {
        ret = -1;
    } else if (result == vk::Result::eSuccess) {
        ret = tbuf[1] - tbuf[0];
    } else {
        std::cerr << "Error!! GPU time get result failed: " << result << std::endl;
        std::exit(-1);
    }
    vd->resetTimer(timerIndex);
    return toMicroSeconds(ret,vd->props.limits.timestampPeriod);
}

} // namespace
