#include "GPURadixSort.hpp"
#include <cstring>
#include <fstream>
#include <iostream>

const uint32_t WORKGROUP_SIZE = 256;
const uint32_t RADIX_BITS = 4;
const uint32_t NUM_BUCKETS = 16;
const uint32_t NUM_PASSES = 32 / RADIX_BITS; // 8

struct HistPushConstants {
  uint32_t shift;
  uint32_t numElements;
};

struct ScanPushConstants {
  uint32_t numGroups;
};

// Helper Functions
std::vector<uint32_t> loadData(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("Failed to open file");

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint32_t> buffer(size / sizeof(uint32_t));
  if (file.read((char *)buffer.data(), size))
    return buffer;
  throw std::runtime_error("Read error");
}

void print_usage(const char *progName) {
  std::cout << "Usage: " << progName << " -m <millions> -d <distribution_type>"
            << std::endl;
  std::cout << "  -m: Number of elements in millions (default 1)" << std::endl;
  std::cout
      << "  -d: Distribution type (0=Uniform, 1=Normal, 2=Zipf) (default 0)"
      << std::endl;
}

// RadixSort Implementation
RadixSort::RadixSort(VulkanContext &context, uint32_t n)
    : ctx(context), numElements(n) {
  numGroups = (numElements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

  // Allocate Buffers
  vk::DeviceSize dataSize = numElements * sizeof(uint32_t);
  vk::DeviceSize histogramSize = numGroups * NUM_BUCKETS * sizeof(uint32_t);
  vk::DeviceSize offsetsSize = NUM_BUCKETS * sizeof(uint32_t);

  buf_in = ctx.createBuffer(dataSize,
                            vk::BufferUsageFlagBits::eStorageBuffer |
                                vk::BufferUsageFlagBits::eTransferDst |
                                vk::BufferUsageFlagBits::eTransferSrc,
                            vk::MemoryPropertyFlagBits::eDeviceLocal);

  buf_out = ctx.createBuffer(dataSize,
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc,
                             vk::MemoryPropertyFlagBits::eDeviceLocal);

  buf_histograms =
      ctx.createBuffer(histogramSize, vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eDeviceLocal);

  // Bucket offsets (scan output), must be host visible for CPU prefix sum
  buf_bucket_offsets =
      ctx.createBuffer(offsetsSize,
                       vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferSrc,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent);

  // Global bucket offsets (scatter input), uploaded from CPU
  buf_global_offsets =
      ctx.createBuffer(offsetsSize,
                       vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent);

  // Calculate blocks for scan (256 items per block)
  numscanBlocks = (numGroups + 256 - 1) / 256;
  vk::DeviceSize partialSumsSize =
      numscanBlocks * NUM_BUCKETS * sizeof(uint32_t);

  buf_partial_sums =
      ctx.createBuffer(partialSumsSize, vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eDeviceLocal);

  // Payload Buffers
  buf_payload_in = ctx.createBuffer(dataSize,
                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                        vk::BufferUsageFlagBits::eTransferDst |
                                        vk::BufferUsageFlagBits::eTransferSrc,
                                    vk::MemoryPropertyFlagBits::eDeviceLocal);
  buf_payload_out = ctx.createBuffer(dataSize,
                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                         vk::BufferUsageFlagBits::eTransferSrc |
                                         vk::BufferUsageFlagBits::eTransferDst,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal);

  createPipelines();
}

RadixSort::~RadixSort() {
  ctx.destroyBuffer(buf_in);
  ctx.destroyBuffer(buf_out);
  ctx.destroyBuffer(buf_histograms);
  ctx.destroyBuffer(buf_bucket_offsets);
  ctx.destroyBuffer(buf_global_offsets);
  ctx.destroyBuffer(buf_partial_sums);
  ctx.destroyBuffer(buf_payload_in);
  ctx.destroyBuffer(buf_payload_out);

  for (auto &buf : columnBuffers)
    ctx.destroyBuffer(buf);
  ctx.destroyBuffer(rowIdBuffer);
  ctx.destroyBuffer(tempGatherBuffer);

  ctx.device.destroyPipeline(histPipeline);
  ctx.device.destroyPipeline(scatterPipeline);

  ctx.device.destroyPipeline(scanReducePipeline);
  ctx.device.destroyPipeline(scanGlobalPipeline);
  ctx.device.destroyPipeline(scanAddPipeline);

  ctx.device.destroyPipelineLayout(histLayout);
  ctx.device.destroyPipelineLayout(scatterLayout);

  ctx.device.destroyPipelineLayout(scanReduceLayout);
  ctx.device.destroyPipelineLayout(scanGlobalLayout);
  ctx.device.destroyPipelineLayout(scanAddLayout);

  ctx.device.destroyDescriptorSetLayout(histDescLayout);
  ctx.device.destroyDescriptorSetLayout(scatterDescLayout);

  ctx.device.destroyDescriptorSetLayout(scanReduceDescLayout);
  ctx.device.destroyDescriptorSetLayout(scanGlobalDescLayout);
  ctx.device.destroyDescriptorSetLayout(scanAddDescLayout);

  ctx.device.destroyPipeline(shufflePipeline);
  ctx.device.destroyPipelineLayout(shuffleLayout);
  ctx.device.destroyDescriptorSetLayout(shuffleDescLayout);

  ctx.device.destroyDescriptorPool(descriptorPool);
}

// Upload keys and initial payload (e.g. 0..N-1)
void RadixSort::uploadData(const std::vector<uint32_t> &keys,
                           const std::vector<uint32_t> &payloads) {
  vk::DeviceSize size = keys.size() * sizeof(uint32_t);
  BufferResource staging =
      ctx.createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent);

  // Upload Keys
  memcpy(staging.mapped, keys.data(), size);

  ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
    vk::BufferCopy copyRegion(0, 0, size);
    cmd.copyBuffer(staging.buffer, buf_in.buffer, 1, &copyRegion);
  });

  // Reuse staging for Payloads (assuming same size)
  memcpy(staging.mapped, payloads.data(), size);

  ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
    vk::BufferCopy copyRegion(0, 0, size);
    cmd.copyBuffer(staging.buffer, buf_payload_in.buffer, 1, &copyRegion);
  });

  ctx.destroyBuffer(staging);
}

std::vector<uint32_t> RadixSort::downloadKeys() {
  std::vector<uint32_t> result(numElements);
  vk::DeviceSize size = numElements * sizeof(uint32_t);
  BufferResource staging =
      ctx.createBuffer(size, vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent);

  ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
    vk::BufferCopy copyRegion(0, 0, size);
    cmd.copyBuffer(buf_in.buffer, staging.buffer, 1, &copyRegion);
  });

  memcpy(result.data(), staging.mapped, size);

  ctx.destroyBuffer(staging);
  return result;
}

std::vector<uint32_t> RadixSort::downloadPayload() {
  std::vector<uint32_t> result(numElements);
  vk::DeviceSize size = numElements * sizeof(uint32_t);
  BufferResource staging =
      ctx.createBuffer(size, vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent);

  ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
    vk::BufferCopy copyRegion(0, 0, size);
    cmd.copyBuffer(buf_payload_in.buffer, staging.buffer, 1, &copyRegion);
  });

  memcpy(result.data(), staging.mapped, size);

  ctx.destroyBuffer(staging);
  return result;
}

void RadixSort::run() {
  bool pingPong = false; // false = in->out, true = out->in

  for (uint32_t pass = 0; pass < 8; pass++) {
    uint32_t shift = pass * 4;

    BufferResource *currentIn = pingPong ? &buf_out : &buf_in;
    BufferResource *currentOut = pingPong ? &buf_in : &buf_out;

    BufferResource *currentPayloadIn =
        pingPong ? &buf_payload_out : &buf_payload_in;
    BufferResource *currentPayloadOut =
        pingPong ? &buf_payload_in : &buf_payload_out;

    // 1. Histogram
    updateHistogramDescriptors(*currentIn);
    ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
      cmd.bindPipeline(vk::PipelineBindPoint::eCompute, histPipeline);
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, histLayout, 0, 1,
                             &histSet, 0, nullptr);

      HistPushConstants push{shift, numElements};
      cmd.pushConstants(histLayout, vk::ShaderStageFlagBits::eCompute, 0,
                        sizeof(HistPushConstants), &push);

      cmd.dispatch(numGroups, 1, 1);

      vk::MemoryBarrier barrier(vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eShaderRead);
      cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eComputeShader, {}, 1,
                          &barrier, 0, nullptr, 0, nullptr);
    });

    // 2. Scan (Multi-pass)

    // Pass 1: Reduce (Local Scan + Block Sums)
    ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
      cmd.bindPipeline(vk::PipelineBindPoint::eCompute, scanReducePipeline);
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, scanReduceLayout,
                             0, 1, &scanReduceSet, 0, nullptr);

      uint32_t push = numGroups;
      cmd.pushConstants(scanReduceLayout, vk::ShaderStageFlagBits::eCompute, 0,
                        sizeof(uint32_t), &push);

      // Dispatch (numscanBlocks, 16, 1) -> One Z-slice per bucket
      cmd.dispatch(numscanBlocks, 16, 1);

      // Barrier: Partial sums written
      vk::MemoryBarrier barrier(vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eShaderRead);
      cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eComputeShader, {}, 1,
                          &barrier, 0, nullptr, 0, nullptr);
    });

    // Pass 2: Global Scan of Partial Sums
    ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
      cmd.bindPipeline(vk::PipelineBindPoint::eCompute, scanGlobalPipeline);
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, scanGlobalLayout,
                             0, 1, &scanGlobalSet, 0, nullptr);

      uint32_t push = numscanBlocks;
      cmd.pushConstants(scanGlobalLayout, vk::ShaderStageFlagBits::eCompute, 0,
                        sizeof(uint32_t), &push);

      // One workgroup containing 16 threads (one per bucket)
      cmd.dispatch(1, 1, 1);

      vk::MemoryBarrier barrier(vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eShaderRead |
                                    vk::AccessFlagBits::eHostRead);
      cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eComputeShader |
                              vk::PipelineStageFlagBits::eHost,
                          {}, 1, &barrier, 0, nullptr, 0, nullptr);
    });

    // Pass 3: Add Block Offsets
    ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
      cmd.bindPipeline(vk::PipelineBindPoint::eCompute, scanAddPipeline);
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, scanAddLayout, 0,
                             1, &scanAddSet, 0, nullptr);

      uint32_t push = numGroups;
      cmd.pushConstants(scanAddLayout, vk::ShaderStageFlagBits::eCompute, 0,
                        sizeof(uint32_t), &push);

      cmd.dispatch(numscanBlocks, 16, 1);

      vk::MemoryBarrier barrier(
          vk::AccessFlagBits::eShaderWrite,
          vk::AccessFlagBits::eHostRead); // Host read for debug check
      cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0,
                          nullptr, 0, nullptr);
    });

    // 3. CPU Prefix Sum (Calculation is in scan_global now, but
    // globalBaseOffsets setup is tricky) Wait, scan_global produces
    // 'bucketTotal', which are the total counts for each bucket. But
    // 'buf_global_offsets' are the START offsets for each bucket (Prefix Sum of
    // bucketTotal). scan_global only scanned the partial sums. We need to scan
    // 'bucketTotal' on CPU to get 'globalBaseOffsets'.

    uint32_t *bucketCounts = (uint32_t *)buf_bucket_offsets.mapped;
    uint32_t *globalOffsets = (uint32_t *)buf_global_offsets.mapped;

    uint32_t sum = 0;
    for (int i = 0; i < NUM_BUCKETS; i++) {
      globalOffsets[i] = sum;
      sum += bucketCounts[i];
    }

    // 4. Scatter
    updateScatterDescriptors(*currentIn, *currentOut, *currentPayloadIn,
                             *currentPayloadOut);
    ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
      cmd.bindPipeline(vk::PipelineBindPoint::eCompute, scatterPipeline);
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, scatterLayout, 0,
                             1, &scatterSet, 0, nullptr);

      HistPushConstants push{shift, numElements};
      cmd.pushConstants(scatterLayout, vk::ShaderStageFlagBits::eCompute, 0,
                        sizeof(HistPushConstants), &push);

      cmd.dispatch(numGroups, 1, 1);

      vk::MemoryBarrier barrier(vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eShaderRead);
      cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eComputeShader, {}, 1,
                          &barrier, 0, nullptr, 0, nullptr);
    });

    pingPong = !pingPong;
  }

  // If ended on pingPong=true (pass 7 done, pingpong becomes false? wait).
  // Pass 0 (false -> true). In=buf_in, Out=buf_out. Next=buf_out.
  // ...
  // Pass 7 (true -> false). In=buf_out, Out=buf_in. Next=buf_in.
  // Result is in buf_in. Correct.
}

void RadixSort::updateScatterDescriptors(const BufferResource &in,
                                         const BufferResource &out,
                                         const BufferResource &pIn,
                                         const BufferResource &pOut) {
  vk::DescriptorBufferInfo inInfo(in.buffer, 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo outInfo(out.buffer, 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo pInInfo(pIn.buffer, 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo pOutInfo(pOut.buffer, 0, VK_WHOLE_SIZE);

  std::vector<vk::WriteDescriptorSet> writes = {
      {scatterSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &inInfo},
      {scatterSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &outInfo},
      {scatterSet, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &pInInfo},
      {scatterSet, 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &pOutInfo}};
  ctx.device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);
}

void RadixSort::createPipelines() {
  // 1. Descriptor Pool
  std::vector<vk::DescriptorPoolSize> sizes = {
      {vk::DescriptorType::eStorageBuffer, 100}}; // Increased for safety
  vk::DescriptorPoolCreateInfo poolInfo({}, 50, sizes.size(),
                                        sizes.data()); // maxSets = 50
  descriptorPool = ctx.device.createDescriptorPool(poolInfo);

  // 2. Histogram Pipeline
  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}};
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings.size(),
                                                 bindings.data());
    histDescLayout = ctx.device.createDescriptorSetLayout(layoutInfo);

    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1, &histDescLayout);
    histSet = ctx.device.allocateDescriptorSets(allocInfo)[0];

    auto code = VulkanContext::readFile("../shaders/spv/histogram.spv");
    auto module = ctx.createShaderModule(code);

    vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0,
                                    sizeof(HistPushConstants));
    vk::PipelineLayoutCreateInfo plInfo({}, 1, &histDescLayout, 1, &pushRange);
    histLayout = ctx.device.createPipelineLayout(plInfo);

    vk::ComputePipelineCreateInfo pInfo(
        {}, {{}, vk::ShaderStageFlagBits::eCompute, module, "main"},
        histLayout);

    histPipeline = ctx.device.createComputePipeline(nullptr, pInfo).value;
    ctx.device.destroyShaderModule(module);

    vk::DescriptorBufferInfo histInfo(buf_histograms.buffer, 0, VK_WHOLE_SIZE);
    vk::WriteDescriptorSet write(histSet, 1, 0, 1,
                                 vk::DescriptorType::eStorageBuffer, nullptr,
                                 &histInfo);
    ctx.device.updateDescriptorSets(1, &write, 0, nullptr);
  }

  // 3. Scan Pipelines (Reduce, Global, Add)

  // --- Scan Reduce ---
  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // Hist
        {1, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute} // Partial Sums
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings.size(),
                                                 bindings.data());
    scanReduceDescLayout = ctx.device.createDescriptorSetLayout(layoutInfo);

    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1,
                                            &scanReduceDescLayout);
    scanReduceSet = ctx.device.allocateDescriptorSets(allocInfo)[0];

    vk::DescriptorBufferInfo histInfo(buf_histograms.buffer, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo partialInfo(buf_partial_sums.buffer, 0,
                                         VK_WHOLE_SIZE);
    std::vector<vk::WriteDescriptorSet> writes = {
        {scanReduceSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
         &histInfo},
        {scanReduceSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
         &partialInfo}};
    ctx.device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);

    auto code = VulkanContext::readFile("../shaders/spv/scan_reduce.spv");
    auto module = ctx.createShaderModule(code);
    vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0,
                                    sizeof(uint32_t));
    vk::PipelineLayoutCreateInfo plInfo({}, 1, &scanReduceDescLayout, 1,
                                        &pushRange);
    scanReduceLayout = ctx.device.createPipelineLayout(plInfo);
    vk::ComputePipelineCreateInfo pInfo(
        {}, {{}, vk::ShaderStageFlagBits::eCompute, module, "main"},
        scanReduceLayout);
    scanReducePipeline = ctx.device.createComputePipeline(nullptr, pInfo).value;
    ctx.device.destroyShaderModule(module);
  }

  // --- Scan Global ---
  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // Partial Sums (In/Out)
        {1, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute} // Global Offsets (Bucket Total)
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings.size(),
                                                 bindings.data());
    scanGlobalDescLayout = ctx.device.createDescriptorSetLayout(layoutInfo);

    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1,
                                            &scanGlobalDescLayout);
    scanGlobalSet = ctx.device.allocateDescriptorSets(allocInfo)[0];

    vk::DescriptorBufferInfo partialInfo(buf_partial_sums.buffer, 0,
                                         VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo offsetInfo(buf_bucket_offsets.buffer, 0,
                                        VK_WHOLE_SIZE);
    std::vector<vk::WriteDescriptorSet> writes = {
        {scanGlobalSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
         &partialInfo},
        {scanGlobalSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
         &offsetInfo}};
    ctx.device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);

    auto code = VulkanContext::readFile("../shaders/spv/scan_global.spv");
    auto module = ctx.createShaderModule(code);
    vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0,
                                    sizeof(uint32_t));
    vk::PipelineLayoutCreateInfo plInfo({}, 1, &scanGlobalDescLayout, 1,
                                        &pushRange);
    scanGlobalLayout = ctx.device.createPipelineLayout(plInfo);
    vk::ComputePipelineCreateInfo pInfo(
        {}, {{}, vk::ShaderStageFlagBits::eCompute, module, "main"},
        scanGlobalLayout);
    scanGlobalPipeline = ctx.device.createComputePipeline(nullptr, pInfo).value;
    ctx.device.destroyShaderModule(module);
  }

  // --- Scan Add ---
  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // Hist
        {1, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute} // Partial Sums
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings.size(),
                                                 bindings.data());
    scanAddDescLayout = ctx.device.createDescriptorSetLayout(layoutInfo);

    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1,
                                            &scanAddDescLayout);
    scanAddSet = ctx.device.allocateDescriptorSets(allocInfo)[0];

    vk::DescriptorBufferInfo histInfo(buf_histograms.buffer, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo partialInfo(buf_partial_sums.buffer, 0,
                                         VK_WHOLE_SIZE);
    std::vector<vk::WriteDescriptorSet> writes = {
        {scanAddSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
         &histInfo},
        {scanAddSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
         &partialInfo}};
    ctx.device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);

    auto code = VulkanContext::readFile("../shaders/spv/scan_add.spv");
    auto module = ctx.createShaderModule(code);
    vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0,
                                    sizeof(uint32_t));
    vk::PipelineLayoutCreateInfo plInfo({}, 1, &scanAddDescLayout, 1,
                                        &pushRange);
    scanAddLayout = ctx.device.createPipelineLayout(plInfo);
    vk::ComputePipelineCreateInfo pInfo(
        {}, {{}, vk::ShaderStageFlagBits::eCompute, module, "main"},
        scanAddLayout);
    scanAddPipeline = ctx.device.createComputePipeline(nullptr, pInfo).value;
    ctx.device.destroyShaderModule(module);
  }

  // 4. Scatter Pipeline
  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // In Keys
        {1, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // Out Keys
        {2, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // Hist
        {3, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // Global Offsets
        {4, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // In Payload
        {5, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute} // Out Payload
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings.size(),
                                                 bindings.data());
    scatterDescLayout = ctx.device.createDescriptorSetLayout(layoutInfo);

    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1,
                                            &scatterDescLayout);
    scatterSet = ctx.device.allocateDescriptorSets(allocInfo)[0];

    // Static bindings (2, 3)
    vk::DescriptorBufferInfo histInfo(buf_histograms.buffer, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo offInfo(buf_global_offsets.buffer, 0,
                                     VK_WHOLE_SIZE);

    std::vector<vk::WriteDescriptorSet> writes = {
        {scatterSet, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
         &histInfo},
        {scatterSet, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
         &offInfo}};
    ctx.device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);

    auto code = VulkanContext::readFile("../shaders/spv/scatter.spv");
    auto module = ctx.createShaderModule(code);

    vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0,
                                    sizeof(HistPushConstants));
    vk::PipelineLayoutCreateInfo plInfo({}, 1, &scatterDescLayout, 1,
                                        &pushRange);
    scatterLayout = ctx.device.createPipelineLayout(plInfo);

    vk::ComputePipelineCreateInfo pInfo(
        {}, {{}, vk::ShaderStageFlagBits::eCompute, module, "main"},
        scatterLayout);

    scatterPipeline = ctx.device.createComputePipeline(nullptr, pInfo).value;
    ctx.device.destroyShaderModule(module);
  }

  // 5. Shuffle Pipeline
  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // Input
        {1, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute}, // Indices
        {2, vk::DescriptorType::eStorageBuffer, 1,
         vk::ShaderStageFlagBits::eCompute} // Output
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings.size(),
                                                 bindings.data());
    shuffleDescLayout = ctx.device.createDescriptorSetLayout(layoutInfo);

    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1,
                                            &shuffleDescLayout);
    shuffleSet = ctx.device.allocateDescriptorSets(allocInfo)[0];

    auto code = VulkanContext::readFile("../shaders/spv/shuffle.spv");
    auto module = ctx.createShaderModule(code);

    vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0,
                                    sizeof(uint32_t));
    vk::PipelineLayoutCreateInfo plInfo({}, 1, &shuffleDescLayout, 1,
                                        &pushRange);
    shuffleLayout = ctx.device.createPipelineLayout(plInfo);

    vk::ComputePipelineCreateInfo pInfo(
        {}, {{}, vk::ShaderStageFlagBits::eCompute, module, "main"},
        shuffleLayout);
    shufflePipeline = ctx.device.createComputePipeline(nullptr, pInfo).value;
    ctx.device.destroyShaderModule(module);
  }
}

void RadixSort::updateHistogramDescriptors(const BufferResource &in) {
  vk::DescriptorBufferInfo info(in.buffer, 0, VK_WHOLE_SIZE);
  vk::WriteDescriptorSet write(
      histSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &info);
  ctx.device.updateDescriptorSets(1, &write, 0, nullptr);
}

void RadixSort::setInputBuffers(const BufferResource &keys,
                                const BufferResource &payloads) {
  vk::DeviceSize size = numElements * sizeof(uint32_t);
  ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
    vk::BufferCopy copyRegion(0, 0, size);
    if (keys.buffer != buf_in.buffer) {
      cmd.copyBuffer(keys.buffer, buf_in.buffer, 1, &copyRegion);
    }
    if (payloads.buffer != buf_payload_in.buffer) {
      cmd.copyBuffer(payloads.buffer, buf_payload_in.buffer, 1, &copyRegion);
    }
  });
}

void RadixSort::performShuffle(const BufferResource &src,
                               const BufferResource &dst,
                               const BufferResource &indices) {
  // Use member shuffleSet instead of allocating new one

  vk::DescriptorBufferInfo srcInfo(src.buffer, 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo indicesInfo(indices.buffer, 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo dstInfo(dst.buffer, 0, VK_WHOLE_SIZE);

  std::vector<vk::WriteDescriptorSet> writes = {
      {shuffleSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &srcInfo},
      {shuffleSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &indicesInfo},
      {shuffleSet, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &dstInfo}};
  ctx.device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);

  ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, shufflePipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, shuffleLayout, 0, 1,
                           &shuffleSet, 0, nullptr);

    uint32_t push = numElements;
    cmd.pushConstants(shuffleLayout, vk::ShaderStageFlagBits::eCompute, 0,
                      sizeof(uint32_t), &push);

    cmd.dispatch((numElements + 255) / 256, 1, 1);

    vk::MemoryBarrier barrier(vk::AccessFlagBits::eShaderWrite,
                              vk::AccessFlagBits::eShaderRead |
                                  vk::AccessFlagBits::eHostRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader |
                            vk::PipelineStageFlagBits::eHost,
                        {}, 1, &barrier, 0, nullptr, 0, nullptr);
  });
}

// --- Generic N-Column API Implementation ---

void RadixSort::allocateColumns(uint32_t numCols) {
  vk::DeviceSize dataSize = numElements * sizeof(uint32_t);

  // Allocate Column Buffers (Device Local)
  columnBuffers.resize(numCols);
  for (uint32_t i = 0; i < numCols; i++) {
    columnBuffers[i] =
        ctx.createBuffer(dataSize,
                         vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eTransferDst |
                             vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlagBits::eDeviceLocal);
  }

  // Allocate Row ID Buffer
  rowIdBuffer = ctx.createBuffer(dataSize,
                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                     vk::BufferUsageFlagBits::eTransferDst |
                                     vk::BufferUsageFlagBits::eTransferSrc,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal);

  // Allocate Temp Gather Buffer
  tempGatherBuffer =
      ctx.createBuffer(dataSize,
                       vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferDst |
                           vk::BufferUsageFlagBits::eTransferSrc,
                       vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void RadixSort::uploadColumnData(
    const std::vector<std::vector<uint32_t>> &columns,
    const std::vector<uint32_t> &rowIDs) {
  if (columns.size() != columnBuffers.size()) {
    throw std::runtime_error(
        "Column data size mismatch with allocated buffers");
  }

  vk::DeviceSize dataSize = numElements * sizeof(uint32_t);
  vk::DeviceSize totalSize =
      (columns.size() + 1) * dataSize; // Columns + RowIDs

  BufferResource staging =
      ctx.createBuffer(totalSize, vk::BufferUsageFlagBits::eTransferSrc,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent);

  uint8_t *ptr = (uint8_t *)staging.mapped;

  // Copy Columns to Staging
  for (size_t i = 0; i < columns.size(); i++) {
    memcpy(ptr + (i * dataSize), columns[i].data(), dataSize);
  }
  // Copy RowIDs
  memcpy(ptr + (columns.size() * dataSize), rowIDs.data(), dataSize);

  ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
    vk::BufferCopy copyRegion(0, 0, dataSize);

    // Upload Columns
    for (size_t i = 0; i < columns.size(); i++) {
      copyRegion.srcOffset = i * dataSize;
      cmd.copyBuffer(staging.buffer, columnBuffers[i].buffer, 1, &copyRegion);
    }

    // Upload RowIDs
    copyRegion.srcOffset = columns.size() * dataSize;
    cmd.copyBuffer(staging.buffer, rowIdBuffer.buffer, 1, &copyRegion);
  });

  ctx.destroyBuffer(staging);
}

void RadixSort::sort(uint32_t numCols) {
  if (numCols < 1)
    return;

  // 1. Sort the last column (Secondary Key) first
  // Input: Last Column, Payload: RowIDs
  setInputBuffers(columnBuffers[numCols - 1], rowIdBuffer);
  run();

  // Note: run() leaves the Sorted Keys in buf_in (or buf_out depending on
  // pingpong) and Sorted Payloads in buf_payload_in (or out). Ideally, we want
  // the Sorted Payloads (RowIDs) to be in a known state. The run() method logic
  // doesn't explicitly guarantee where the final result is without checking the
  // internal ping-pong state, but based on the code: "Result is in buf_in.
  // Correct." -> So Keys in buf_in. And Payloads should be in buf_payload_in.

  // Update: We NEED the sorted RowIDs for the next shuffle.
  // Since run() finishes with results in buf_payload_in, we can use that.

  // 2. Loop for remaining columns (k = numCols - 2 down to 0)
  for (int k = (int)numCols - 2; k >= 0; k--) {
    // Current Sorted RowIDs are in buf_payload_in (from previous sort).

    // Shuffle Column k: temp = Column[k][SortedRowIDs]
    performShuffle(columnBuffers[k], tempGatherBuffer, buf_payload_in);

    // Now Sort Column k
    // Input: Temp Gathered Column k
    // Payload: The CURRENT Sorted RowIDs (which were used for gathering)
    setInputBuffers(tempGatherBuffer, buf_payload_in);

    run();

    // After run():
    // buf_in contains Sorted Column k (we don't strictly need this anymore)
    // buf_payload_in contains the New Sorted RowIDs (stable sorted by Col k)
  }
}

std::vector<uint32_t> RadixSort::downloadRowIDs() {
  std::vector<uint32_t> result(numElements);
  vk::DeviceSize size = numElements * sizeof(uint32_t);
  BufferResource staging =
      ctx.createBuffer(size, vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent |
                           vk::MemoryPropertyFlagBits::eHostCached);

  ctx.runCommandBuffer([&](vk::CommandBuffer cmd) {
    vk::BufferCopy copyRegion(0, 0, size);
    // Result of last sort is in buf_payload_in
    cmd.copyBuffer(buf_payload_in.buffer, staging.buffer, 1, &copyRegion);
  });

  memcpy(result.data(), staging.mapped, size);
  ctx.destroyBuffer(staging);
  return result;
}
