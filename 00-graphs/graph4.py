import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
factors = np.arange(1, 9)
labels = [f'{i}' for i in factors]

# Data provided
# Build Times (ms)
build_times = np.array([41.0, 46.9, 48.3, 55.6, 98.7, 130.3, 153.2, 182.3])

# Avg Query Times (ms)
query_times = np.array([4.45, 4.49, 4.49, 4.38, 4.42, 4.52, 4.54, 4.48])

# --- 2. Plot Configuration ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True 
})

fig, ax = plt.subplots()

x = np.arange(len(labels))
width = 0.6 

# --- 3. Creating Bars ---
# Using the CompactScan color (Light Orange) from previous plots
rects = ax.bar(x, build_times, width,
               label='Compact Index Build Time',
               color='#FFD1A4',
               edgecolor='black',
               linewidth=1.2)

# --- 4. Annotations ---

# A. Build Time (Inside the bar)
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height/2,
            f'{height:.1f}',
            ha='center', va='center', rotation=45, 
            fontsize=10, fontweight='bold', color='black')

# B. Avg Query Time (On top of the bar)
for i, rect in enumerate(rects):
    height = rect.get_height()
    q_time = query_times[i]
    # Format: Q: 2.519 ms
    ax.annotate(f'{q_time:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),rotation=45,
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='black')

# --- 5. Labels and Titles ---
ax.set_ylabel('Build Time (ms)')
ax.set_xlabel('Scaling Factor')
ax.set_title('Compact Index Build Time vs Scaling Factor [100M Uniform Dataset]')
ax.set_xticks(x)
ax.set_xticklabels(['SF' + str(i) for i in factors])

# Increase Y-limit to make room for annotations
ax.set_ylim(0, max(build_times) * 1.2)

# Legend (Optional, since there's only one series)
ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)

# Borders
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

plt.tight_layout()

# --- 6. Save ---
output_file = 'scaling_build_query.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.savefig('scaling_build_query.pdf', bbox_inches='tight')
print(f"Graph saved as {output_file}")