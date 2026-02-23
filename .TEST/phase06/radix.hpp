#pragma once
#include <cstdint>
#include <vector>

// Parallel LSD Radix Sort for uint32_t
void cpu_parallel_radix_sort(std::vector<uint32_t> &data);
void cpu_parallel_radix_sort_pairs(std::vector<uint32_t> &keys,
                                   std::vector<uint32_t> &payloads);
