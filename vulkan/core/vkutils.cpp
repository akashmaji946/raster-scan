// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "vkutils.h"

namespace vkcore {

void readUsingStagingBuf(char *data, size_t size, PBuffer buf, PBuffer staging, PVkDevice vd, int tid) {
    size_t rem = size;
    size_t offset = 0;
    while(rem > 0) {
        size_t csize = (std::min)(rem, (size_t)STAGING_BUFFER_SIZE);
        vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->transferCommandBuffer[tid].get());
        vd->transferCommandBuffer[tid]->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        staging->copyFrom(csize,offset,0,buf,tid);
        vd->transferCommandBuffer[tid]->end();
        vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
        vd->submit(submitInfo,drawFence.get(),true);
        vd->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);

        staging->readData(data + offset,csize,0);

        offset += csize;
        rem -= csize;
    }
}

void loadUsingStagingBuf(char *data, size_t size, PBuffer buf, PBuffer staging, PVkDevice vd, size_t dstOffset, int tid) {
    size_t rem = size;
    size_t offset = 0;
    while(rem > 0) {
        size_t csize = (std::min)(rem, (size_t)STAGING_BUFFER_SIZE);
        staging->loadData(data + offset,csize,0);

        vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &vd->transferCommandBuffer[tid].get());
        vd->transferCommandBuffer[tid]->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        buf->copyFrom(csize,0,offset+dstOffset,staging,tid);
        vd->transferCommandBuffer[tid]->end();
        vk::UniqueFence drawFence = vd->device->createFenceUnique(vk::FenceCreateInfo());
        vd->submit(submitInfo,drawFence.get(),true);
        vd->waitForFences(drawFence.get(), VK_TRUE, UINT64_MAX);

        offset += csize;
        rem -= csize;
    }
}

}
