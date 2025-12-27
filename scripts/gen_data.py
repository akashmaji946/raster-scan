#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import subprocess

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate data with multiple distributions')
parser.add_argument('-m', '--num-millions', type=int, default=50, help='Number of millions of rows (default: 50)')
parser.add_argument('-c', '--num-cols', type=int, default=3, help='Number of columns (default: 3)')
args = parser.parse_args()

# Convert millions to actual number of rows
n = args.num_millions * 1000000
c = args.num_cols

# Create data folder
folder = f"data/data_{args.num_millions}m_{c}c"
os.makedirs(folder, exist_ok=True)

print(f"[INFO] Generating data with m={args.num_millions} (n={n}), c={c}")
print(f"[INFO] Output folder: {folder}/")

# Generate uniform data
print(f"[INFO] Generating uniform data: {folder}/uniform.bin")
uniform_data = np.random.uniform(0, 2**32 - 1, (c, n)).astype(np.uint32)
uniform_data.tofile(f"{folder}/uniform.bin")

# Generate normal data
print(f"[INFO] Generating normal data: {folder}/normal.bin")
normal_data = np.random.normal(2**31, 1, (c, n)).astype(np.uint32)
normal_data.tofile(f"{folder}/normal.bin")

# Generate zipf data for all 3 variants
zipf_params = [1.1, 1.3, 1.5]
for param in zipf_params:
    print(f"[INFO] Generating zipf{param} data: {folder}/zipf{param}.bin")
    zipf_data = np.random.zipf(param, (c, n))
    zipf_data = zipf_data % np.iinfo(np.uint32).max
    zipf_data.astype("uint32").tofile(f"{folder}/zipf{param}.bin")

print(f"[INFO] Generated datasets in {folder}/ :")
os.system(f"ls -lash {folder}/")
