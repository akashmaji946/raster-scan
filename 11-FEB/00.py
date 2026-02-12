import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Configuration
# ---------------------------
zipf_param = 1.01
UINT32_MAX = np.iinfo(np.uint32).max

# Different n values to visualize
n_values = [10_000, 100_000, 1_000_000]

# ---------------------------
# Helper function
# ---------------------------
def analyze_zipf(zipf_data, n):
    """
    Prints percentage of values less than
    10%, 20%, 30% of [0, UINT32_MAX]
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"\n[STATS] n = {n}")
    for t in thresholds:
        threshold_value = int(t * UINT32_MAX)
        percentage = np.mean(zipf_data < threshold_value) * 100
        print(f"  < {int(t*100)}% of range: {percentage:.4f}%")

# ---------------------------
# Plotting
# ---------------------------
plt.figure(figsize=(10, 6))

for n in n_values:
    # Generate 1-column Zipf data
    data = np.random.zipf(zipf_param, size=n)

    # Map into uint32 range (same as your pipeline)
    data = (data % UINT32_MAX).astype(np.uint32)

    # Analyze skew percentages
    analyze_zipf(data, n)

    # Plot histogram (log-log to reveal skew)
    counts, bins = np.histogram(data, bins=1000)
    plt.loglog(bins[:-1], counts, label=f"n={n}")

# ---------------------------
# Plot formatting
# ---------------------------
plt.xlabel("Value (uint32)")
plt.ylabel("Frequency")
plt.title("Zipf(1.01) Skewed Distribution (1 Column)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
