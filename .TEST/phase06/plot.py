import matplotlib.pyplot as plt
import numpy as np

# =============================
# Benchmark Data (seconds)
# =============================

# 3 Columns
cpu_parallel_3 = 17.0807
gpu_total_3 = 3.14754
gpu_compute_3 = 1.57375

# 2 Columns
cpu_parallel_2 = 9.55478
gpu_total_2 = 2.32196
gpu_compute_2 = 1.01917

# 1 Column
cpu_parallel_1 = 2.20499
gpu_total_1 = 1.55923
gpu_compute_1 = 0.513741

# X-axis groups
column_labels = ["1 Column", "2 Columns", "3 Columns"]
x = np.arange(len(column_labels))
width = 0.20

plt.figure()

bars1 = plt.bar(x - width,
                [cpu_parallel_1, cpu_parallel_2, cpu_parallel_3],
                width, label="CPU Parallel Radix Time")

bars2 = plt.bar(x,
                [gpu_total_1, gpu_total_2, gpu_total_3],
                width, label="GPU Total Time (Sort + Data Transfer)")

bars3 = plt.bar(x + width,
                [gpu_compute_1, gpu_compute_2, gpu_compute_3],
                width, label="GPU Parallel Radix Time")

# Add labels (1 decimal place)
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height,
                 f"{height:.1f}",
                 ha='center',
                 va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.xticks(x, column_labels)
plt.yticks([i for i in range(0, 21, 2)])
plt.ylabel("Time (seconds)")
plt.title("100M Rows Sorting Benchmark (GTX 1650)")
plt.legend()

plt.tight_layout()
plt.savefig("name.png")
