// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// CPU vs GPU Results Comparison Tool
// Compares query results from CPU and GPU implementations

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <set>
#include <cstdint>

const std::string PROJECT_DIR = "/home/akashmaji/Device/IMPORTANT/raster-scan/";

std::vector<std::string> datasets = {
    "normal",
    "zipf1.5",
    "zipf1.3",
    "zipf1.1",
    "uniform",
};

// Parse rowIds from a comma-separated string
std::set<uint32_t> parseRowIds(const std::string& line) {
    std::set<uint32_t> rowIds;
    if (line.empty()) return rowIds;
    
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            try {
                rowIds.insert(std::stoul(token));
            } catch (...) {
                // Skip invalid tokens
            }
        }
    }
    return rowIds;
}

// Extract rowIds from a results file
std::vector<std::set<uint32_t>> extractRowIds(const std::string& filePath) {
    std::vector<std::set<uint32_t>> results;
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file: " << filePath << "\n";
        return results;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("RowIds: ") == 0) {
            std::string rowIdStr = line.substr(8);  // Skip "RowIds: "
            results.push_back(parseRowIds(rowIdStr));
        }
    }
    
    file.close();
    return results;
}

// Compare two sets of rowIds
struct ComparisonResult {
    uint32_t cpuCount;
    uint32_t gpuCount;
    uint32_t matchCount;
    uint32_t cpuOnlyCount;
    uint32_t gpuOnlyCount;
    bool isMatch;
    double accuracy;
};

ComparisonResult compareResults(const std::set<uint32_t>& cpuRowIds, 
                                const std::set<uint32_t>& gpuRowIds) {
    ComparisonResult result;
    result.cpuCount = cpuRowIds.size();
    result.gpuCount = gpuRowIds.size();
    
    // Count matches
    result.matchCount = 0;
    for (const auto& id : cpuRowIds) {
        if (gpuRowIds.count(id) > 0) {
            result.matchCount++;
        }
    }
    
    // Count CPU-only and GPU-only
    result.cpuOnlyCount = result.cpuCount - result.matchCount;
    result.gpuOnlyCount = result.gpuCount - result.matchCount;
    
    // Check if results match exactly
    result.isMatch = (cpuRowIds == gpuRowIds);
    
    // Calculate accuracy (Jaccard similarity)
    uint32_t unionSize = result.cpuCount + result.gpuCount - result.matchCount;
    result.accuracy = (unionSize > 0) ? (double)result.matchCount / unionSize : 1.0;
    
    return result;
}

void compareDataset(const std::string& datasetName) {
    std::string cpuFile = PROJECT_DIR + "compare/cpu_files/" + datasetName + "_results.txt";
    std::string gpuFile = PROJECT_DIR + "compare/gpu_files/" + datasetName + "_results.txt";
    
    std::cerr << "\n========================================\n";
    std::cerr << "Comparing: " << datasetName << "\n";
    std::cerr << "========================================\n";
    
    // Extract rowIds from both files
    auto cpuResults = extractRowIds(cpuFile);
    auto gpuResults = extractRowIds(gpuFile);
    
    if (cpuResults.empty()) {
        std::cerr << "ERROR: No CPU results found in " << cpuFile << "\n";
        return;
    }
    
    if (gpuResults.empty()) {
        std::cerr << "ERROR: No GPU results found in " << gpuFile << "\n";
        return;
    }
    
    if (cpuResults.size() != gpuResults.size()) {
        std::cerr << "WARNING: Different number of queries. CPU: " << cpuResults.size() 
                  << ", GPU: " << gpuResults.size() << "\n";
    }
    
    // Compare results
    std::cout << "\n--- Query Results Comparison ---\n";
    std::cout << std::setw(8) << "Query"
              << std::setw(12) << "CPU_Count"
              << std::setw(12) << "GPU_Count"
              << std::setw(12) << "Matches"
              << std::setw(12) << "CPU_Only"
              << std::setw(12) << "GPU_Only"
              << std::setw(12) << "Accuracy"
              << std::setw(10) << "Match?\n";
    std::cout << std::string(90, '-') << "\n";
    
    int totalQueries = std::min(cpuResults.size(), gpuResults.size());
    int allMatch = 1;
    double totalAccuracy = 0.0;
    uint32_t totalCpuCount = 0, totalGpuCount = 0, totalMatches = 0;
    
    for (int i = 0; i < totalQueries; i++) {
        auto cmp = compareResults(cpuResults[i], gpuResults[i]);
        totalAccuracy += cmp.accuracy;
        totalCpuCount += cmp.cpuCount;
        totalGpuCount += cmp.gpuCount;
        totalMatches += cmp.matchCount;
        
        if (!cmp.isMatch) allMatch = 0;
        
        std::cout << std::setw(8) << (i + 1)
                  << std::setw(12) << cmp.cpuCount
                  << std::setw(12) << cmp.gpuCount
                  << std::setw(12) << cmp.matchCount
                  << std::setw(12) << cmp.cpuOnlyCount
                  << std::setw(12) << cmp.gpuOnlyCount
                  << std::setw(12) << std::fixed << std::setprecision(4) << cmp.accuracy
                  << std::setw(10) << (cmp.isMatch ? "YES" : "NO") << "\n";
    }
    
    std::cout << std::string(90, '-') << "\n";
    std::cout << "\n--- Summary ---\n";
    std::cout << "Total queries compared: " << totalQueries << "\n";
    std::cout << "Total CPU matches: " << totalCpuCount << "\n";
    std::cout << "Total GPU matches: " << totalGpuCount << "\n";
    std::cout << "Total matching rowIds: " << totalMatches << "\n";
    std::cout << "Average accuracy (Jaccard): " << std::fixed << std::setprecision(4) 
              << (totalAccuracy / totalQueries) << "\n";
    std::cout << "All queries match: " << (allMatch ? "YES" : "NO") << "\n";
    std::cout << "========================================\n";
}

void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " [dataset_name]\n";
    std::cerr << "  dataset_name: Optional. If provided, compares only that dataset.\n";
    std::cerr << "                If not provided, compares all datasets.\n";
    std::cerr << "  Available datasets: normal, zipf1.5, zipf1.3, zipf1.1, uniform\n";
}

int main(int argc, char* argv[]) {
    std::cerr << "\n*** CPU vs GPU Results Comparison Tool ***\n";
    
    if (argc > 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    if (argc == 2) {
        // Compare specific dataset
        std::string datasetName = argv[1];
        compareDataset(datasetName);
    } else {
        // Compare all datasets
        std::cerr << "\nComparing all datasets...\n";
        for (const auto& dataset : datasets) {
            compareDataset(dataset);
        }
    }
    
    std::cerr << "\n*** Comparison Complete ***\n";
    return 0;
}
