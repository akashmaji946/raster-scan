# Code Refactoring Summary

## Overview
Refactored the data generation, encoding, and scanning pipelines to accept command-line arguments and use global variables for flexible configuration.

## Changes Made

### 1. Created `scripts/gen_data.py`
**Purpose**: Generate data with multiple distributions (normal, uniform, zipf variants)

**Usage**:
```bash
python3 scripts/gen_data.py -m 128 -c 3
```

**Arguments**:
- `-m`: Number of millions of rows (default: 50)
- `-c`: Number of columns (default: 3)

**Output**: Creates binary files in `data/data_{m}m_{c}c/` folder:
- `normal.bin`
- `uniform.bin`
- `zipf1.1.bin`
- `zipf1.3.bin`
- `zipf1.5.bin`

---

### 2. Refactored `scan/encode.cpp`
**Purpose**: Encode raw data files to RasterScan format

**Usage**:
```bash
./EncodeData -m 128 -c 3
```

**Arguments**:
- `-m`: Number of millions of rows (default: 50)
- `-c`: Number of columns (default: 3)

**Global Variables**:
- `PROJECT_DIR`: Base project directory
- `g_dim`: Number of dimensions/columns
- `g_npoints`: Number of points
- `g_folder`: Input data folder path
- `g_opfolder`: Output encoded data folder path
- `g_datasets`: List of dataset names

**Input**: Reads from `data/data_{m}m_{c}c/` folder

**Output**: Generates encoded files in `encodedData/data_{m}m_{c}c/` folder:
- `{dataset}-data.bin`
- `{dataset}-map.bin`

---

### 3. Refactored `scan/scan.cpp`
**Purpose**: Run RasterScan queries on encoded data

**Usage**:
```bash
./RasterScan -m 128 -c 3 -t testfolder -g A
```

**Arguments**:
- `-m`: Number of millions of rows (default: 50)
- `-c`: Number of columns (default: 3)
- `-t`: Test folder name (default: test)
- `-g`: GPU vendor (A=AMD, N=NVIDIA, D=Default, default: D)

**Global Variables**:
- `PROJECT_DIR`: Base project directory
- `g_dim`: Number of dimensions/columns
- `g_npoints`: Number of points
- `g_opfolder`: Encoded data folder path
- `g_qfolder`: Test queries folder path
- `datasets`: List of dataset names (global)
- `querysets`: List of query file names (global)
- `qct`: Query counts per dataset (global)

**Input**: 
- Reads encoded data from `encodedData/data_{m}m_{c}c/` folder
- Reads test queries from `tests/{testfolder}/` folder

**Output**: Query execution results and performance metrics

---

## Default Values

All programs use the following defaults:
- **m (millions)**: 50
- **c (columns)**: 3
- **testFolder**: "test"
- **GPU**: Default device

## Folder Structure

```
/home/akashmaji/Documents/RasterDB/raster-scan/
├── data/
│   └── data_{m}m_{c}c/          # Generated raw data
│       ├── normal.bin
│       ├── uniform.bin
│       ├── zipf1.1.bin
│       ├── zipf1.3.bin
│       └── zipf1.5.bin
├── encodedData/
│   └── data_{m}m_{c}c/          # Encoded data
│       ├── normal-data.bin
│       ├── normal-map.bin
│       ├── uniform-data.bin
│       ├── uniform-map.bin
│       └── ... (other datasets)
├── tests/
│   └── {testfolder}/            # Test queries
│       ├── normal.txt
│       ├── uniform.txt
│       ├── zipf1.1.txt
│       ├── zipf1.3.txt
│       └── zipf1.5.txt
└── scripts/
    └── gen_data.py              # Data generation script
```

## Example Workflow

```bash
# Step 1: Generate data (128 million rows, 3 columns)
python3 scripts/gen_data.py -m 128 -c 3

# Step 2: Encode data
./EncodeData -m 128 -c 3

# Step 3: Run queries on AMD GPU
./RasterScan -m 128 -c 3 -t testfolder -g A
```

## Notes

- All paths use `PROJECT_DIR` global variable for flexibility
- Command-line arguments override default values
- GPU selection supports: A (AMD), N (NVIDIA), D (Default)
- The refactoring maintains backward compatibility with existing code logic
- No new functionality was added, only restructuring and parameterization
