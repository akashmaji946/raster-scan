// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <cstring>

#include "modes/ModeUtils.hpp"
#include "modes/RunModes.hpp"

#include <core/VkEngine.hpp>
#include <core/vkutils.h>

using namespace vkcore;

// Flag to select pipeline:
// 0 = Original RasterScan2D pipeline
// 1 = New RasterScanIndexUpdate pipeline (naive linked list)
// 2 = Compare both pipelines above

// 3 = Test dynamic operations (rowId-based delete)
// 4 = Test dynamic updated (range delete operations)

// 5 = GPU query execution (output results to text files)
// 6 = Bitmap-based free space management stress test (insert/delete cycles)

// 7 = Delete-by-data performance test (random data)
// 8 = Delete-by-data performance test with varying distributions (uses encodedData)
// 9 = Robustness test: delete/insert cycles + compact + reverse order cycles
// 10 = CPU Verification test: runs cycles on GPU and verifies results against CPU ground truth

// 11 = Same as Mode 8 but with VARYING batch sizes
// 12 = Same as Mode 9 but with VARYING batch sizes
// 13 = Same as Mode 10 but with VARYING batch sizes

// 21 = Compact Index Mode (Contiguous memory per bin)
// 22 = CompactIndex + RasterScan2D Comparison
// 23 = Batch Delete/Insert Cycles with CPU Verification
// 24 = TPC-C Customer Table Benchmark
// 25 = TPC-C with Mode 22 Structure
// 42 = Mode 22 + Query Performance After Each Update Cycle
// 44 = TPC-C with Query Performance After Each Update Cycle

// 51 = Equi-Depth Index (per-axis equi-depth binning for balanced bins)
// 52 = RasterScan2D + EquiDepth Index Comparison
// 53 = 

// 60 = BruteForceIndex + RasterScan2D Comparison (TPC-C)
// 61 = BruteForceIndex + RasterScan2D Comparison

// 62 = BruteForceIndex + EquiDepth Index Comparison
// 63 = EquiDepthIndex + EquiDepth Index Comparison (TPC-C)

// 65 = BruteForceIndex + RasterScan2D Comparison (100 queries, 1%-100% selectivity)
// 66 = BruteForceIndex + RasterScan2D Comparison (100 point queries)
// 67 = BruteForceIndex + RasterScan2D Comparison (100 queries, 1% selectivity)

// 64 = BruteForceIndex + RasterScan2D Comparison (TPC-C, 100 queries, 1%-100% selectivity)
// 68 = BruteForceIndex + RasterScan2D Comparison (TPC-C, 100 point queries)
// 69 = BruteForceIndex + RasterScan2D Comparison (TPC-C, 100 queries, 1% selectivity)

// 70 = BruteForce Scan with Update Support (TPC-C) - BRUTE_SCALE_FACTOR controls buffer size

// 80 = CompactBruteScan - builds like CompactScan (binned), queries like BruteForce (all bins), with aux buffer for updates
// 81 = CompactBruteScan - builds like CompactScan (binned), queries like BruteForce (all bins), with aux buffer pushAuxToMain for updates
#define USE_INDEX_UPDATE_PIPELINE 81

int main(int argc, char* argv[]) {
    // Default values
    int m = 100;  // millions of rows
    int c = 3;   // columns
    int d = 0;   // dataId: 0=uniform, 1=normal, 2=zipf1.1, 3=zipf1.3, 4=zipf1.5

    std::string testFolder = "test";
    char gpuVendor = 'D';  // Default
    bool useSkewedPipeline = false;  // -s flag: use indexed delete for skewed distributions
    int runMode = 0;  // -r flag: 0=both, 1=first index only, 2=second index only

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            c = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            testFolder = argv[++i];
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            gpuVendor = argv[++i][0];
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            d = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0) {
            useSkewedPipeline = true;  // Use indexed delete for skewed distributions
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            runMode = atoi(argv[++i]);  // 0=both, 1=first index only, 2=second index only
        } else {
            printUsage(argv[0]);
            return 1;
        }
    }

    // Set global variables based on arguments
    g_dim = c;
    g_npoints = uint32_t(m) * 1000000;
    g_opfolder = PROJECT_DIR + "encodedData/data_" + std::to_string(m) + "m_" + std::to_string(c) + "c/";
    g_qfolder = PROJECT_DIR + "tests/" + testFolder + "/";
    g_useSkewedPipeline = useSkewedPipeline;
    g_runmode = runMode;

    std::cerr << "[Configuration] m=" << m << ", c=" << c << ", d=" << d << ", testFolder=" << testFolder << ", gpuVendor=" << gpuVendor << ", skewedPipeline=" << (useSkewedPipeline ? "ON" : "OFF") << "\n";
    std::cerr << "[Configuration] Data folder: " << g_opfolder << "\n";
    std::cerr << "[Configuration] Test folder: " << g_qfolder << "\n";

    // Parse command-line arguments for GPU selection
    int devId = -1;
    std::cerr << "[GPU Selection] Using -g " << gpuVendor << "\n";
    std::cerr.flush();
    devId = selectGPUByVendor(gpuVendor);
    
    // If no GPU specified, use default
    if(devId == -1) {
        devId = VkEngine::getEngine()->getDefaultDeviceId();
    }
    
    std::cerr << "[GPU Selection] Selected device ID: " << devId << "\n";
    std::cerr.flush();
    
    PVkDevice vd = VkEngine::getEngine()->getDevice(devId);

    std::cerr << "\n\n**** using device " << vd->props.deviceName << " with ID = " << devId << " ****" << std::endl;
    
    // Print GPU memory info (calculated)
    uint64_t gpuMemory = vd->heapSize;
    std::cerr << "[GPU Info] Device: " << vd->props.deviceName << std::endl;
    std::cerr << "[GPU Info] VRAM:   " << GPUMemoryTool::formatBytes(gpuMemory) << std::endl;
    std::cerr << "[GPU Info] MAX_PAGES: " << MAX_PAGES << " (" << GPUMemoryTool::formatBytes(4ll * MAX_PAGES * PAGE_SIZE_UINTS ) << " page buffer)" << std::endl;
    
    // Query actual GPU memory usage from VMA
    GPUMemoryTool::printGPUMemoryUsage(vd);
    GPUMemoryTool::printGPUMemoryStatus(vd, "Initial state");
    
    PBuffer staging(new Buffer(vd));
    staging->create(STAGING_BUFFER_SIZE,vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst,MemoryType::ReadWrite);
    GPU_MEM_PRINT("Staging Buffer", STAGING_BUFFER_SIZE);
    GPUMemoryTool::printGPUMemoryStatus(vd, "After staging buffer");
    
    OperatorCache op;
    op.initialize(vd);
    GPUMemoryTool::printGPUMemoryStatus(vd, "After OperatorCache init");

    
#if USE_INDEX_UPDATE_PIPELINE == 81
    testCompactBruteScanAuxOverflow(vd, d, staging, testFolder);
#elif USE_INDEX_UPDATE_PIPELINE == 80
    testCompactBruteScan(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 70
    testBruteForceUpdateTPCC(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 69
    testBruteForceVsRasterScanTPCCRandom1Pct(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 68
    testBruteForceVsRasterScanTPCCPointQueries(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 67
    testBruteForceVsRasterScanRandom1Pct(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 66
    testBruteForceVsRasterScanPointQueries(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 65
    testBruteForceVsRasterScan100(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 64
    testBruteForceVsRasterScanTPCC100(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 63
    testBruteForceVsEquiDepthTPCC(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 62
    testBruteForceVsEquiDepthPlain(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 61
    testBruteForceVsRasterScanPlain(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 60
    testBruteForceVsRasterScan(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 53
    testTPCCEquiDepthVsRasterScan(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 52
    testEquiDepthVsRasterScan(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 51
    testEquiDepthIndex(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 44
    testTPCCWithQueryAfterUpdate(d, vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 42
    testCompactIndexWithQueryAfterUpdate(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 13
    testCPUVerificationVarying(vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 25
    testTPCCMode25(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 24
    testTPCCBenchmark(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 23
    testCompactIndexBatchCycles(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 21
    testCompactIndex(0, vd, staging, op); // Hardcoded dataId 0 (normal) for test
#elif USE_INDEX_UPDATE_PIPELINE == 22
    testCompactIndexAndCompare(d, vd, staging, op);
#elif USE_INDEX_UPDATE_PIPELINE == 12
    std::cerr << "\n*** Robustness Test: Delete/Insert + Compact + Reverse Cycles (Varying Batch Sizes) ***\n";
    testRobustnessWithReverseCyclesVarying(vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 11
    std::cerr << "\n*** Delete-by-Data Performance Test with Varying Distributions (Varying Batch Sizes) ***\n";
    testDeleteByDataWithDistributionsVarying(vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 10
    testCPUVerification(vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 9
    std::cerr << "\n*** Robustness Test: Delete/Insert + Compact + Reverse Cycles ***\n";
    testRobustnessWithReverseCycles(vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 8
    std::cerr << "\n*** Delete-by-Data Performance Test with Varying Distributions ***\n";
    testDeleteByDataWithDistributions(vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 7
    std::cerr << "\n*** Delete-by-Data Performance Test ***\n";
    testDeleteByData(vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 6
    std::cerr << "\n*** Bitmap-based Free Space Management Stress Test ***\n";
    testBitmapFreeSpaceManagement(vd, staging);
#elif USE_INDEX_UPDATE_PIPELINE == 5
    std::cerr << "\n*** GPU Query Execution with Output to Files ***\n";
    for(int i = 0; i < nDataset; i++) {
        queriesGPUWithOutput(i, vd, staging, op, true);
    }
#elif USE_INDEX_UPDATE_PIPELINE == 4
    std::cerr << "\n*** Testing Range Delete Operations ***\n";
    testRangeDelete(vd, staging, "1M");
    testRangeDelete(vd, staging, "10M");
    testRangeDelete(vd, staging, "100M");
#elif USE_INDEX_UPDATE_PIPELINE == 3
    std::cerr << "\n*** Testing Dynamic Index Operations ***\n";
    testDynamicOperations(vd, staging, "1M");
    testDynamicOperations(vd, staging, "10M");
    testDynamicOperations(vd, staging, "100M");
#elif USE_INDEX_UPDATE_PIPELINE == 2
    std::cerr << "\n*** COMPARING Both Pipelines ***\n";
    for(int i = 0; i < nDataset; i++) {
        compareResults(i, vd, staging, op);
    }
#elif USE_INDEX_UPDATE_PIPELINE == 1
    std::cerr << "\n*** Running NEW RasterScanIndexUpdate (Linked List) Pipeline ***\n";
    for(int i = 0; i < nDataset; i++) {
        queries3dIndexUpdate(i, vd, staging, true);
    }
#else
    std::cerr << "\n*** Running ORIGINAL RasterScan2D Pipeline ***\n";
    for(int i = 0; i < nDataset; i++) {
        queries3d(i, vd, staging, op, true);
    }
#endif

    return 0;
}
