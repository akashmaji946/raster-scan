#!/bin/bash

# Default values
n=100000000
c=3

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            n="$2"
            shift 2
            ;;
        -c)
            c="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-n <number of rows>] [-c <number of columns>]"
            exit 1
            ;;
    esac
done

echo "[INFO] Generating data with n=$n, c=$c"

# Create data folder
mkdir -p "data/data_n${n}_c${c}"

# Compile and run uniform data generator
g++ scripts/generate_uniform_data_n_col.cpp -o generate_uniform_data_n_col

echo "[INFO] Generating uniform data with n=$n, c=$c"
./generate_uniform_data_n_col -n "$n" -c "$c"
rm generate_uniform_data_n_col

# Run Python data generators
echo "[INFO] Generating zipf data with n=$n, c=$c"
python scripts/generate_zipf_data.py -n "$n" -c "$c"

echo "[INFO] Generating normal data with n=$n, c=$c"
python scripts/generate_normal_data.py -n "$n" -c "$c"

echo "[INFO] Generated datasets in ./data/data_n${n}_c${c}/ :" 

ls -lash "data/data_n${n}_c${c}/"