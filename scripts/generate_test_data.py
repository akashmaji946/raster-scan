#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate test query files for different distributions')
parser.add_argument('-n', '--num-rows', type=int, default=100000000, help='Number of rows')
parser.add_argument('-c', '--num-cols', type=int, default=3, help='Number of columns')
args = parser.parse_args()

n = args.num_rows
c = args.num_cols

# Create test folder
test_folder = f"test/test_n{n}_c{c}"
os.makedirs(test_folder, exist_ok=True)

# Generate test files with simple queries
# Each test file contains 3 "lt" queries (one per column) followed by "exit"
# The threshold values are set to 90% of the max uint32 value

max_val = np.iinfo(np.uint32).max
threshold_90_percent = int(max_val * 0.9)

test_files = {
    'normal.txt': 'normal 90%',
    'uniform.txt': 'uniform 90%',
    'zipf1.1.txt': 'zipf 1.1 90%',
    'zipf1.3.txt': 'zipf 1.3 90%',
    'zipf1.5.txt': 'zipf 1.5 90%',
}

for filename, description in test_files.items():
    filepath = os.path.join(test_folder, filename)
    print(f"[INFO] Generating test file: {filepath}")
    
    with open(filepath, 'w') as f:
        # Write 3 "lt" queries (one per column)
        for col in range(c):
            f.write(f"lt {threshold_90_percent}\n")
        
        # Write exit command
        f.write("exit\n")
        
        # Write description comment
        f.write(f"{description}\n")

print(f"[INFO] Generated test files in {test_folder}/")
