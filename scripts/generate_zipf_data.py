#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate Zipf distribution data')
parser.add_argument('-n', '--num-rows', type=int, default=100000000, help='Number of rows')
parser.add_argument('-c', '--num-cols', type=int, default=3, help='Number of columns')
args = parser.parse_args()

n = args.num_rows
c = args.num_cols

# Create data folder
folder = f"data/data_n{n}_c{c}"
os.makedirs(folder, exist_ok=True)

# Generate zipf data for all 3 variants
zipf_params = [1.1, 1.3, 1.5]
for param in zipf_params:
    print(f"[INFO] Generating zipf{param} data: {folder}/zipf{param}.bin")
    zipf_data = np.random.zipf(param, (c, n))
    zipf_data = zipf_data % np.iinfo(np.uint32).max
    zipf_data.astype("uint32").tofile(f"{folder}/zipf{param}.bin")
