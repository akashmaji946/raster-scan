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

# Generate TPC-C Customer table data (shuffled)
# TPC-C Constants
DISTRICTS_PER_WAREHOUSE = 10
CUSTOMERS_PER_DISTRICT = 3000
CUSTOMERS_PER_WAREHOUSE = DISTRICTS_PER_WAREHOUSE * CUSTOMERS_PER_DISTRICT

print(f"[INFO] Generating TPC-C data (shuffled): {folder}/tpcc.bin")
target_customers = n
warehouse_count = (target_customers + CUSTOMERS_PER_WAREHOUSE - 1) // CUSTOMERS_PER_WAREHOUSE
print(f"[TPC-C] Target customers: {target_customers}, Warehouses needed: {warehouse_count}")

# Generate W, D, C columns
W = np.zeros(target_customers, dtype=np.uint32)
D = np.zeros(target_customers, dtype=np.uint32)
C = np.zeros(target_customers, dtype=np.uint32)

count = 0
for c_w_id in range(1, warehouse_count + 1):
    if count >= target_customers:
        break
    for c_d_id in range(1, DISTRICTS_PER_WAREHOUSE + 1):
        if count >= target_customers:
            break
        for c_id in range(1, CUSTOMERS_PER_DISTRICT + 1):
            if count >= target_customers:
                break
            W[count] = c_w_id
            D[count] = c_d_id
            C[count] = c_id
            count += 1

print(f"[TPC-C] Generated {count} customers")

# Shuffle rows randomly (Fisher-Yates shuffle with fixed seed for reproducibility)
print("[TPC-C] Shuffling data rows...")
np.random.seed(42)
indices = np.random.permutation(target_customers)
W = W[indices]
D = D[indices]
C = C[indices]

# Stack into column-major format (same as other distributions)
tpcc_data = np.vstack([W, D, C]).astype(np.uint32)
tpcc_data.tofile(f"{folder}/tpcc.bin")
print(f"[TPC-C] W range: [{W.min()}, {W.max()}], D range: [{D.min()}, {D.max()}], C range: [{C.min()}, {C.max()}]")

print(f"[INFO] Generated datasets in {folder}/ :")
os.system(f"ls -lash {folder}/")
