#include "radix.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// Helper to get thread count
static unsigned int get_thread_count() {
  unsigned int t = std::thread::hardware_concurrency();
  return t == 0 ? 4 : t;
}

void cpu_parallel_radix_sort(std::vector<uint32_t> &data) {
  const int NUM_PASSES = 4; // 32 bits / 8 bits
  const int RADIX_BITS = 8;
  const int NUM_BUCKETS = 256;
  const uint32_t MASK = 0xFF;

  size_t n = data.size();
  if (n == 0)
    return;

  unsigned int numThreads = get_thread_count();

  std::vector<uint32_t> buffer(n);
  std::vector<uint32_t> *src = &data;
  std::vector<uint32_t> *dst = &buffer;

  for (int pass = 0; pass < NUM_PASSES; pass++) {
    int shift = pass * RADIX_BITS;

    // 1. Parallel Histogram
    std::vector<std::array<size_t, NUM_BUCKETS>> histograms(numThreads, {0});
    std::vector<std::thread> threads;
    size_t chunkSize = (n + numThreads - 1) / numThreads;

    for (unsigned int t = 0; t < numThreads; t++) {
      threads.emplace_back([&, t]() {
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, n);
        for (size_t i = start; i < end; i++) {
          uint32_t val = (*src)[i];
          uint8_t bucket = (val >> shift) & MASK;
          histograms[t][bucket]++;
        }
      });
    }
    for (auto &t : threads)
      t.join();
    threads.clear();

    // 2. Prefix Sum (Global)
    std::vector<std::array<size_t, NUM_BUCKETS>> globalOffsets(numThreads);
    size_t sum = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
      for (unsigned int t = 0; t < numThreads; t++) {
        globalOffsets[t][b] = sum;
        sum += histograms[t][b];
      }
    }

    // 3. Parallel Scatter
    for (unsigned int t = 0; t < numThreads; t++) {
      threads.emplace_back([&, t]() {
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, n);
        std::array<size_t, NUM_BUCKETS> myOffsets = globalOffsets[t];

        for (size_t i = start; i < end; i++) {
          uint32_t val = (*src)[i];
          uint8_t bucket = (val >> shift) & MASK;
          (*dst)[myOffsets[bucket]++] = val;
        }
      });
    }
    for (auto &t : threads)
      t.join();
    threads.clear();

    // Swap
    std::swap(src, dst);
  }

  if (src != &data) {
    data = *src;
  }
}

void cpu_parallel_radix_sort_pairs(std::vector<uint32_t> &keys,
                                   std::vector<uint32_t> &payloads) {
  const int NUM_PASSES = 4; // 32 bits / 8 bits
  const int RADIX_BITS = 8;
  const int NUM_BUCKETS = 256;
  const uint32_t MASK = 0xFF;

  size_t n = keys.size();
  if (n == 0)
    return;

  unsigned int numThreads = get_thread_count();

  std::vector<uint32_t> key_buffer(n);
  std::vector<uint32_t> payload_buffer(n);

  std::vector<uint32_t> *key_src = &keys;
  std::vector<uint32_t> *key_dst = &key_buffer;

  std::vector<uint32_t> *payload_src = &payloads;
  std::vector<uint32_t> *payload_dst = &payload_buffer;

  for (int pass = 0; pass < NUM_PASSES; pass++) {
    int shift = pass * RADIX_BITS;

    // 1. Parallel Histogram
    std::vector<std::array<size_t, NUM_BUCKETS>> histograms(numThreads, {0});
    std::vector<std::thread> threads;
    size_t chunkSize = (n + numThreads - 1) / numThreads;

    for (unsigned int t = 0; t < numThreads; t++) {
      threads.emplace_back([&, t]() {
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, n);
        for (size_t i = start; i < end; i++) {
          uint32_t val = (*key_src)[i];
          uint8_t bucket = (val >> shift) & MASK;
          histograms[t][bucket]++;
        }
      });
    }
    for (auto &t : threads)
      t.join();
    threads.clear();

    // 2. Prefix Sum
    std::vector<std::array<size_t, NUM_BUCKETS>> globalOffsets(numThreads);
    size_t sum = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
      for (unsigned int t = 0; t < numThreads; t++) {
        globalOffsets[t][b] = sum;
        sum += histograms[t][b];
      }
    }

    // 3. Parallel Scatter
    for (unsigned int t = 0; t < numThreads; t++) {
      threads.emplace_back([&, t]() {
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, n);
        std::array<size_t, NUM_BUCKETS> myOffsets = globalOffsets[t];

        for (size_t i = start; i < end; i++) {
          uint32_t val = (*key_src)[i];
          uint32_t payload = (*payload_src)[i];
          uint8_t bucket = (val >> shift) & MASK;

          size_t pos = myOffsets[bucket]++;
          (*key_dst)[pos] = val;
          (*payload_dst)[pos] = payload;
        }
      });
    }
    for (auto &t : threads)
      t.join();
    threads.clear();

    // Swap
    std::swap(key_src, key_dst);
    std::swap(payload_src, payload_dst);
  }

  // Copy back if needed
  if (key_src != &keys) {
    keys = *key_src;
    payloads = *payload_src;
  }
}
