// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <core/VkInclude.hpp>
#include <core/VulkanDevice.hpp>
#include <core/VkData.hpp>

namespace vkcore {

void loadUsingStagingBuf(char *data, size_t size, PBuffer buf, PBuffer staging, PVkDevice vd, size_t dstOffset, int tid=0);
void readUsingStagingBuf(char *data, size_t size, PBuffer buf, PBuffer staging, PVkDevice vd, int tid=0);

} // namespace

