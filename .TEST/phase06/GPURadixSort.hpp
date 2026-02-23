#pragma once
#include "vulkan_common.hpp"
#include <string>
#include <vector>

// Helper functions (formerly in main.cpp)
std::vector<uint32_t> loadData(const std::string &path);
void print_usage(const char *progName);

using namespace vk_raster;

class RadixSort {
public:
  RadixSort(VulkanContext &context, uint32_t n);
  ~RadixSort();

  void uploadData(const std::vector<uint32_t> &data);
  // Upload keys and initial payload (e.g. 0..N-1)
  void uploadData(const std::vector<uint32_t> &keys,
                  const std::vector<uint32_t> &payloads);
  // Keys are in buf_in, Payloads in buf_payload_in. Result in same buffers (or
  // swapped).
  void run();

  std::vector<uint32_t> downloadKeys();
  std::vector<uint32_t> downloadPayload();

  // --- Generic N-Column API ---
  void allocateColumns(uint32_t numCols);
  void uploadColumnData(const std::vector<std::vector<uint32_t>> &columns,
                        const std::vector<uint32_t> &rowIDs);
  void sort(uint32_t numCols);
  std::vector<uint32_t> downloadRowIDs();

private:
  VulkanContext &ctx;
  uint32_t numElements;
  uint32_t numGroups;

  // Buffer Management for N-Columns
  std::vector<BufferResource> columnBuffers;
  BufferResource rowIdBuffer;
  BufferResource tempGatherBuffer; // For intermediate shuffle results

  // Internal Buffers for Single-Column Radix Sort
  BufferResource buf_in, buf_out;
  BufferResource buf_histograms;
  BufferResource buf_bucket_offsets; // Output of scan: counts per bucket
  BufferResource buf_global_offsets; // Input to scatter: prefix sum of counts

  // Payloads
  BufferResource buf_payload_in, buf_payload_out;

  vk::DescriptorPool descriptorPool;

  // Pipelines
  vk::Pipeline histPipeline, scatterPipeline;
  vk::PipelineLayout histLayout, scatterLayout;
  vk::DescriptorSetLayout histDescLayout, scatterDescLayout;
  vk::DescriptorSet histSet, scatterSet;

  // Scan Pipelines (Multi-pass)
  vk::Pipeline scanReducePipeline, scanGlobalPipeline, scanAddPipeline;
  vk::PipelineLayout scanReduceLayout, scanGlobalLayout, scanAddLayout;
  vk::DescriptorSetLayout scanReduceDescLayout, scanGlobalDescLayout,
      scanAddDescLayout;
  vk::DescriptorSet scanReduceSet, scanGlobalSet, scanAddSet;

  // Intermediate Buffers
  BufferResource buf_partial_sums;
  uint32_t numscanBlocks;

  // Shuffle Pipeline
  vk::Pipeline shufflePipeline;
  vk::PipelineLayout shuffleLayout;
  vk::DescriptorSetLayout shuffleDescLayout;
  vk::DescriptorSet shuffleSet; // Added member to reuse descriptor set

  void createPipelines();
  void updateHistogramDescriptors(const BufferResource &in);
  void updateScatterDescriptors(const BufferResource &in,
                                const BufferResource &out,
                                const BufferResource &pIn,
                                const BufferResource &pOut);

  // Helper methods
  void performShuffle(const BufferResource &src, const BufferResource &dst,
                      const BufferResource &indices);
  void setInputBuffers(const BufferResource &keys,
                       const BufferResource &payloads);
};
