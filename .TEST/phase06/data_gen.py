import numpy as np
import argparse
import struct
import os

def generate_data(n, distribution_type, output_file_unused):
    print(f"Generating {n} elements with distribution type '{distribution_type}'...")
    
    if distribution_type == 0: # Uniform
        # Generate random uint32
        data = np.random.randint(0, np.iinfo(np.uint32).max, size=n, dtype=np.uint32)
    elif distribution_type == 1: # Normal
        # Normal distribution centered at INT_MAX/2, scaled to fit mostly within range
        mean = np.iinfo(np.uint32).max / 2
        std_dev = np.iinfo(np.uint32).max / 6 # 99.7% within range
        data = np.random.normal(mean, std_dev, size=n)
        data = np.clip(data, 0, np.iinfo(np.uint32).max).astype(np.uint32)
    elif distribution_type == 2: # Zipf 1.01
        # Zipf is heavy-tailed. numpy.random.zipf produces values >= 1.
        # We need to map these to uint32 range.
        # Zipf(a) parameter a > 1.
        a = 1.01
        print(f"Generating Zipf({a}) distribution... this might take a moment.")
        data = np.random.zipf(a, size=n).astype(np.uint32)
        # Zipf can generate very large numbers, but we want them scattered or concentrated?
        # Usually Zipf implies skew towards small numbers. 
        # Let's keep them as is, but ensuring they fit in uint32 is tricky if they explode.
        # However, for a=1.01, large values are common? No, probability P(k) ~ 1/k^a.
        # So 1 is most common.
        # We might want to scramble them if we want to test sorting effectively, 
        # but user just asked for "zipf1.01".
        # Let's just clip to uint32 max just in case.
        # Note: np.random.zipf can yield very large values, potentially overflowing uint64 even.
        # But for n=1M, max value likely won't exceed uint32 range too often?
        # Actually with a=1.01, tail is very fat.
        # Let's simple mod by UINT32_MAX to keep in range, or clip.
        data = np.bitwise_and(data, np.iinfo(np.uint32).max) 
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

    return data

def main():
    parser = argparse.ArgumentParser(description="Generate multi-column test data for sorting.")
    parser.add_argument("-c", type=int, default=2, help="Number of columns to generate")
    parser.add_argument("-m", type=int, default=1, help="Number of elements in millions")
    parser.add_argument("-d", type=str, default="0", help="Distribution type (0=Uniform, 1=Normal, 2=Zipf)")
    parser.add_argument("--out", type=str, default="phase05/data", help="Output directory")
    
    args = parser.parse_args()
    
    n_elements = args.m * 1000000
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        
    try:
        dist = int(args.d)
    except ValueError:
        dist = args.d

    print(f"Generating {n_elements} elements, {args.c} columns, distribution {dist}...")
    
    all_data = []
    
    # Generate columns independently
    for i in range(args.c):
        print(f"Generating column {i}...")
        col_data = generate_data(n_elements, dist, "")
        all_data.append(col_data)
        
    # Concatenate in column-major order (Col 0, then Col 1, ...)
    # simple concatenation of arrays
    final_data = np.concatenate(all_data)
    
    # Construct filename: data_c{c}_m{m}_d{d}.bin
    filename = f"data_c{args.c}_m{args.m}_d{args.d}.bin"
    output_path = os.path.join(args.out, filename)
    
    print(f"Saving to {output_path}...")
    final_data.tofile(output_path)
    print("Done.")

if __name__ == "__main__":
    main()
