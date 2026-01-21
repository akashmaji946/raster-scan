#pragma once

#include "ModeUtils.hpp"

// Mode 0: Original RasterScan2D Pipeline
void queries3d(int dataId, PVkDevice vd, PBuffer staging, OperatorCache &op, bool test = false);

// Mode 1: New RasterScanIndexUpdate Pipeline
void queries3dIndexUpdate(int dataId, PVkDevice vd, PBuffer staging, bool test = false);

// Mode 2: Compare Pipelines
void compareResults(int dataId, PVkDevice vd, PBuffer staging, OperatorCache &op);

// Mode 3: Dynamic Operations
void testDynamicOperations(PVkDevice vd, PBuffer staging, std::string label);

// Mode 4: Range Delete
void testRangeDelete(PVkDevice vd, PBuffer staging, std::string label);

// Mode 5: GPU Query Output
void queriesGPUWithOutput(int dataId, PVkDevice vd, PBuffer staging, OperatorCache &op, bool test);

// Mode 6: Bitmap Stress Test
void testBitmapFreeSpaceManagement(PVkDevice vd, PBuffer staging);

// Mode 7: Delete by Data
void testDeleteByData(PVkDevice vd, PBuffer staging);

// Mode 8: Delete by Data with Distributions
void testDeleteByDataWithDistributions(PVkDevice vd, PBuffer staging);

// Mode 9: Robustness Test
void testRobustnessWithReverseCycles(PVkDevice vd, PBuffer staging);

// Mode 10: CPU Verification
void testCPUVerification(PVkDevice vd, PBuffer staging);

// Mode 11: Varying Batch Sizes (Mode 8 variant)
void testDeleteByDataWithDistributionsVarying(PVkDevice vd, PBuffer staging);

// Mode 12: Varying Batch Sizes (Mode 9 variant)
void testRobustnessWithReverseCyclesVarying(PVkDevice vd, PBuffer staging);

// Mode 13: Varying Batch Sizes (Mode 10 variant)
void testCPUVerificationVarying(vkcore::PVkDevice vd, vkcore::PBuffer staging);

// Mode 21
void testCompactIndex(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op);

// Mode 22: Mode 21 + RasterScan2D Comparison
void testCompactIndexAndCompare(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op);

// Mode 23: Batch Delete/Insert Cycles with CPU Verification
void testCompactIndexBatchCycles(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op);

// Mode 24: TPC-C Customer Table Benchmark
void testTPCCBenchmark(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op);

// Mode 42: Mode 22 + Query Performance After Each Update Cycle
void testCompactIndexWithQueryAfterUpdate(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op);

// Mode 44: TPC-C with Query Performance After Each Update Cycle
void testTPCCWithQueryAfterUpdate(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging);

// Mode 25: TPC-C with Mode 22 Structure
void testTPCCMode25(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op);

// Mode 51: Equi-Depth Index Test
void testEquiDepthIndex(int dataId, vkcore::PVkDevice vd, vkcore::PBuffer staging, OperatorCache &op);
