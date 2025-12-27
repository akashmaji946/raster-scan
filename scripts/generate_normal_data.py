#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate normal distribution data')
parser.add_argument('-n', '--num-rows', type=int, default=100000000, help='Number of rows')
parser.add_argument('-c', '--num-cols', type=int, default=3, help='Number of columns')
args = parser.parse_args()

n = args.num_rows
c = args.num_cols

# Create data folder
folder = f"data/data_n{n}_c{c}"
os.makedirs(folder, exist_ok=True)

# Generate single file with normal data
print(f"[INFO] Generating normal data: {folder}/normal.bin")
normal_data = np.random.normal(2**31, 1, (c, n)).astype(np.uint32)
normal_data.tofile(f"{folder}/normal.bin")
