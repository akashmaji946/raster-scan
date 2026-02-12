import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
queries = np.arange(10, 101, 10)  # Query percentages (10, 20, ..., 100)

# Compact Index Query Times (ms)
compact_1024 = np.array([1.385, 1.616, 1.837, 2.082, 2.282, 2.442, 2.703, 3.254, 3.118, 3.437])
compact_2048 = np.array([1.490, 1.668, 1.888, 2.098, 2.342, 2.564, 2.821, 2.942, 3.152, 3.646])
compact_4096 = np.array([1.321, 1.533, 1.778, 2.007, 2.211, 2.514, 2.732, 2.964, 3.415, 3.481])
compact_8192 = np.array([1.483, 1.542, 1.759, 1.953, 2.353, 2.555, 2.693, 3.140, 3.146, 3.462])

# RasterScan2D Query Times (ms)
raster_1024 = np.array([2.836, 2.872, 3.231, 3.506, 3.879, 4.008, 3.661, 4.466, 4.467, 4.692])
raster_2048 = np.array([2.690, 2.832, 3.138, 3.633, 3.473, 4.040, 4.117, 4.420, 4.666, 4.949])
raster_4096 = np.array([2.779, 2.856, 3.113, 3.959, 3.433, 3.504, 3.803, 4.282, 4.642, 4.831])
raster_8192 = np.array([2.771, 3.079, 3.021, 3.358, 3.859, 3.863, 4.076, 4.364, 4.581, 4.875])

# --- 2. Plot Configuration ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.8,
    'grid.linestyle': '--'
})

# Standard figure size since no external table is needed
fig, ax = plt.subplots(figsize=(10, 10))

# Define colors for each resolution
colors = ['blue', 'green', 'red', 'purple']

# --- 3. Plotting Compact Lines (Solid) ---
ax.plot(queries, compact_1024, label='Compact 1024', linestyle='-', marker='o', color=colors[0], linewidth=1.5)
ax.plot(queries, compact_2048, label='Compact 2048', linestyle='-', marker='s', color=colors[1], linewidth=1.5)
ax.plot(queries, compact_4096, label='Compact 4096', linestyle='-', marker='^', color=colors[2], linewidth=1.5)
ax.plot(queries, compact_8192, label='Compact 8192', linestyle='-', marker='d', color=colors[3], linewidth=1.5)

# --- 4. Plotting Raster Lines (Dashed) ---
ax.plot(queries, raster_1024, label='Raster 1024', linestyle='--', marker='o', color=colors[0], markerfacecolor='none', linewidth=1.5)
ax.plot(queries, raster_2048, label='Raster 2048', linestyle='--', marker='s', color=colors[1], markerfacecolor='none', linewidth=1.5)
ax.plot(queries, raster_4096, label='Raster 4096', linestyle='--', marker='^', color=colors[2], markerfacecolor='none', linewidth=1.5)
ax.plot(queries, raster_8192, label='Raster 8192', linestyle='--', marker='d', color=colors[3], markerfacecolor='none', linewidth=1.5)

# --- 5. Labels and Axes ---
ax.set_title('Query Performance: Compact vs RasterScan2D')
ax.set_xlabel('Query Selectivity (%)')
ax.set_ylabel('Query Time (ms)')

# Customize Y-axis ticks
ax.set_ylim(0, 5.0)
ax.set_yticks(np.arange(0, 5.5, 0.5))
ax.set_xticks(queries)

# --- Legend Positioned at Bottom Right ---
# ncol=2 makes the legend wider and shorter, fitting nicely in the corner
ax.legend(loc='lower right', fontsize=10, ncol=2, framealpha=0.9, edgecolor='black')

plt.tight_layout()

# --- 6. Save ---
output_file = 'query_performance_simple.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.savefig('query_performance_simple.pdf', bbox_inches='tight')
print(f"Graph saved as {output_file}")