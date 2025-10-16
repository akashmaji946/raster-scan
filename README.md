# Rasterization-based Database Index Scan

The goal of this project is to inform the broader database community that the way researchers are modeling database operations as ray tracing (RT) operations is inefficient. To accomplish this, we  present a rasterization-based indexing approach that adheres to the same data model employed by RT-based methods, and demonstrate that the raster-based method consistently and substantively outperforms its RT counterparts, often achieving order-of-magnitude speedups.

## Getting Started

### Dependencies

1. Requires the VulkanSDK installed. Also select to install the VMA library when installing VulkanSDK.
2. Compatible version of CMake

### Build 

Run cmake, and then build the project

### Test

Two executables are generated.

1. **EncodeData.exe** -- this is used to encode the input data into the same format used by RTScan[1]. This was done to make sure we have an apples-to-apples comparison with the RT-based approach. Generate the datasets using the script in the RTScan repo, and use the generated datasets as input to encode.exe to produce data in the format that is consumed by the Raster-based index.

2. **RasterScan.exe** -- creates the index and runs queries on the input data. The queries for the experiments in the paper uses the same predicates that was used in [1].

[1] Y. Lv, K. Zhang, Z. Wang, X. Zhang, R. Lee, Z. He, Y. Jing, and X. Wang. RTScan: Efficient Scan with Ray Tracing Cores. Proc. VLDB Endow. 17, 6 (2024).


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Citation

```bibtex
@inproceedings{doraiswamy2026rasterscan,
author = {Doraiswamy, Harish and Haritsa, Jayant},
title = {Raster is Faster: Rethinking Ray Tracing in Database Indexing},
booktitle = {16th Annual Conference on Innovative Data Systems Research (CIDR '26)},
year = {2026}
}
```