#include "GPURadixSort.hpp"
#include "radix.hpp"
#include <algorithm>
#include <chrono>
#include <fstream> // Added for std::ifstream
#include <iostream>
#include <numeric> // Added for std::iota
#include <vector>

// --- CONFIGURATION FLAGS ---
#define CPU_INBUILT_SORT 0
#define CPU_PARALLEL_RADIX 1
#define GPU_PARALLEL_RADIX 1
// ---------------------------

// Helper to read N columns from binary file
std::vector<std::vector<uint32_t>>
readColumns(const std::string &filename, int numCols, uint32_t numElements) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open " + filename +
                             ". Did you run data_gen.py?");
  }

  std::vector<std::vector<uint32_t>> columns(numCols);
  for (int i = 0; i < numCols; i++) {
    columns[i].resize(numElements);
    file.read(reinterpret_cast<char *>(columns[i].data()),
              numElements * sizeof(uint32_t));
    if (!file) {
      throw std::runtime_error("Error reading column " + std::to_string(i) +
                               " from " + filename);
    }
  }
  file.close(); // Close the file after reading all columns
  return columns;
}

// Main for Phase 06 - Generic N-Column Sort
int main(int argc, char **argv) {
  try {
    uint32_t numElements = 1000000;
    int m = 1;
    int distId = 0;
    int numCols = 2; // Default

    // Parse args
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "-m")
        m = std::stoi(argv[++i]);
      else if (arg == "-d")
        distId = std::stoi(argv[++i]);
      else if (arg == "-c")
        numCols = std::stoi(argv[++i]);
    }
    numElements = m * 1000000;

    std::cout << "Generating Multi-Column Data (" << numElements
              << " elements, " << numCols << " columns)..." << std::endl;

    // Construct filename: data_c2_m1_d1.bin
    // Note: User said "python3 data_gen.py -c 2 -m 1 -d 1 will generate
    // data_c2_m1_d1.bin" And "./main -c 2 -m 1 -d 1 to read two columns from
    // .bin file data_c2_m1_d1.bin" Assuming we run from build/, the data is in
    // ../data/
    std::string filename = "../data/data_c" + std::to_string(numCols) + "_m" +
                           std::to_string(m) + "_d" + std::to_string(distId) +
                           ".bin";

    std::cout << "Reading from " << filename << std::endl;

    // Generic Read
    std::vector<std::vector<uint32_t>> columns =
        readColumns(filename, numCols, numElements);

    // Create Row IDs
    std::vector<uint32_t> rowIDs(numElements);
    std::iota(rowIDs.begin(), rowIDs.end(), 0);

    // --- 1. Ground Truth (std::stable_sort) ---
    std::cout << "Generating Ground Truth (CPU std::stable_sort)..."
              << std::endl;
    std::vector<uint32_t> cpuRowIDs_Ref = rowIDs;

    auto startRef = std::chrono::high_resolution_clock::now();
    std::stable_sort(cpuRowIDs_Ref.begin(), cpuRowIDs_Ref.end(),
                     [&](uint32_t i, uint32_t j) {
                       for (int c = 0; c < numCols; c++) {
                         if (columns[c][i] != columns[c][j]) {
                           return columns[c][i] < columns[c][j];
                         }
                       }
                       return false; // i < j implicit for stable sort
                     });
    auto endRef = std::chrono::high_resolution_clock::now();
    double timeRef =
        std::chrono::duration<double, std::milli>(endRef - startRef).count();
    std::cout << "Ground Truth Time: " << timeRef << " ms" << std::endl;

    // --- 2. CPU Parallel Radix Sort ---
    std::cout << "Running CPU Parallel Radix Sort..." << std::endl;
    std::vector<uint32_t> cpuRowIDs_Par = rowIDs;

    auto startPar = std::chrono::high_resolution_clock::now();
    {
      // Generic Multi-Column Sort on CPU (Parallel)
      std::vector<uint32_t> keys(numElements);

      // Initial Sort: Last Column
      keys = columns[numCols - 1];
      cpu_parallel_radix_sort_pairs(keys, cpuRowIDs_Par);

      // Remaining Columns
      for (int c = numCols - 2; c >= 0; c--) {
        // Gather keys
        for (size_t i = 0; i < numElements; i++) {
          keys[i] = columns[c][cpuRowIDs_Par[i]];
        }
        // Sort
        cpu_parallel_radix_sort_pairs(keys, cpuRowIDs_Par);
      }
    }
    auto endPar = std::chrono::high_resolution_clock::now();
    double timePar =
        std::chrono::duration<double, std::milli>(endPar - startPar).count();
    std::cout << "CPU Parallel Radix Sort Time: " << timePar << " ms"
              << std::endl;

    // Verify CPU Parallel against Ground Truth
    bool cpu_par_passed = true;
    for (size_t i = 0; i < numElements; i++) {
      if (cpuRowIDs_Par[i] != cpuRowIDs_Ref[i]) {
        std::cout << "CPU Parallel verification FAILED at index " << i
                  << std::endl;
        cpu_par_passed = false;
        break;
      }
    }
    if (cpu_par_passed)
      std::cout << "CPU Parallel Verification PASSED!" << std::endl;
    else
      std::cout << "CPU Parallel Verification FAILED!" << std::endl;

    // --- GPU Setup ---
    std::cout << "Initializing Vulkan..." << std::endl;
    VulkanContext ctx(true); // Validation enabled

    std::cout << "GPU Multi-Column Sort..." << std::endl;
    // Note: We'll time different phases.

    RadixSort sorter(ctx, numElements);

    // 1. Allocate Buffers
    sorter.allocateColumns(numCols);

    // 2. Upload Data (Timer starts here for Transfer)
    std::cout << "Transferring data to GPU..." << std::endl;
    auto transferStart = std::chrono::high_resolution_clock::now();

    sorter.uploadColumnData(columns, rowIDs);

    auto transferEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> transferTime =
        transferEnd - transferStart;

    // 3. Sort (Timer starts here for Compute)
    std::cout << "GPU Sort (Generic Loop)..." << std::endl;
    auto sortStart = std::chrono::high_resolution_clock::now();

    sorter.sort(numCols);

    auto sortEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> sortTime = sortEnd - sortStart;

    // 4. Download Result
    auto downloadStart = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t> finalRowIDs = sorter.downloadRowIDs();

    auto downloadEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> downloadTime =
        downloadEnd - downloadStart;

    double totalTime =
        transferTime.count() + sortTime.count() + downloadTime.count();

    std::cout << "------------------------------------------------"
              << std::endl;
    std::cout << "Data Transfer (Host->GPU): " << transferTime.count() << " ms"
              << std::endl;
    std::cout << "GPU Sort (Compute):        " << sortTime.count() << " ms"
              << std::endl;
    std::cout << "Data Transfer (GPU->Host): " << downloadTime.count() << " ms"
              << std::endl;
    std::cout << "Total Time:                " << totalTime << " ms"
              << std::endl;
    std::cout << "------------------------------------------------"
              << std::endl;

    // Verification
    bool passed = true;
    for (size_t i = 0; i < numElements; i++) {
      // finalRowIDs[i] is the original index of the element at sorted position
      // i. We compare this against cpuRowIDs_Ref[i] which is the golden truth
      // index.

      // Direct index comparison might fail if there are duplicate keys since
      // stable sort guarantees relative order but duplicate keys are
      // equivalent. However, we used std::stable_sort on indices starting with
      // 0..N, so if keys are equal, the one with smaller original index comes
      // first. Thus, finalRowIDs[i] should EXACTLY match cpuRowIDs_Ref[i].

      if (finalRowIDs[i] != cpuRowIDs_Ref[i]) {
        // Let's print details
        std::cerr << "Mismatch at index " << i << ":\n";
        std::cerr << "  GPU RowID=" << finalRowIDs[i] << " (";
        for (int c = 0; c < std::min(numCols, 3); c++)
          std::cerr << columns[c][finalRowIDs[i]]
                    << (c < std::min(numCols, 3) - 1 ? ", " : "");
        std::cerr << ")\n";

        std::cerr << "  REF RowID=" << cpuRowIDs_Ref[i] << " (";
        for (int c = 0; c < std::min(numCols, 3); c++)
          std::cerr << columns[c][cpuRowIDs_Ref[i]]
                    << (c < std::min(numCols, 3) - 1 ? ", " : "");
        std::cerr << ")\n";

        passed = false;
        break;
      }
    }

    if (passed) {
      std::cout << "GPU Verification PASSED!" << std::endl;
    } else {
      std::cout << "GPU Verification FAILED!" << std::endl;
    }

    std::cout << "\n=== FINAL BENCHMARK RESULTS (" << numElements / 1e6
              << "M rows, " << numCols << " cols) ===" << std::endl;
    std::cout << "CPU Ground Truth (std::stable_sort): " << timeRef << " ms"
              << std::endl;
    std::cout << "CPU Parallel Radix Sort:             " << timePar << " ms"
              << std::endl;
    std::cout << "GPU Total Time (inc. transfer):      " << totalTime << " ms"
              << std::endl;
    std::cout << "GPU Compute Time (sort only):        " << sortTime.count()
              << " ms" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Speedup vs CPU Ground Truth:" << std::endl;
    std::cout << "  Total (End-to-End): " << timeRef / totalTime << "x"
              << std::endl;
    std::cout << "  Compute Only:       " << timeRef / sortTime.count() << "x"
              << std::endl;
    std::cout << "Speedup vs CPU Parallel:" << std::endl;
    std::cout << "  Total (End-to-End): " << timePar / totalTime << "x"
              << std::endl;
    std::cout << "  Compute Only:       " << timePar / sortTime.count() << "x"
              << std::endl;
    std::cout << "============================================" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Fatal Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
