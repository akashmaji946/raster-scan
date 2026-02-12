import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
queries = np.arange(1, 11)
labels = [f'Q{i}' for i in queries]

# Query Times (ms)
t_raster = np.array([20.697, 30.826, 38.722, 46.103, 53.308, 59.920, 66.011, 71.699, 77.277, 82.608])
t_compact = np.array([5.284, 6.240, 6.909, 7.688, 8.226, 8.558, 8.922, 9.563, 9.736, 9.907])

# Build Times (ms)
build_raster = 87.5
build_compact = 109.6

# Calculate Average Query Times
avg_raster = np.mean(t_raster)
avg_compact = np.mean(t_compact)

# --- 2. Plot Configuration ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True 
})

fig, ax = plt.subplots()

x = np.arange(len(labels))
width = 0.3

# --- 3. Creating Shaded Bars with Borders ---
rects1 = ax.bar(x - width/2, t_raster, width,
                color='#CCEBFF',      # Light Blue
                edgecolor='black',    
                linewidth=1.2,        
                hatch='///')          

rects2 = ax.bar(x + width/2, t_compact, width,
                color='#FFD1A4',      # Light Orange
                edgecolor='black',    
                linewidth=1.2,        
                hatch='...')          

# --- 4. Annotations (Speedup) ---
speedups = t_raster / t_compact

for i, speedup in enumerate(speedups):
    height = max(t_raster[i], t_compact[i])
    ax.annotate(f'{speedup:.2f}x',
                xy=(x[i], height),
                xytext=(0, 4), 
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='normal', color='black')

# --- 5. Labels and Titles ---
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Queries (Varying Selectivity)')
ax.set_title('100M Zipf1.1 Dataset [8192 Resolution]')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Set Y-axis limit based on the max of BOTH arrays to avoid clipping
ax.set_ylim(0, max(np.max(t_raster), np.max(t_compact)) * 1.5)

# --- 6. Custom Legends (Stacked) ---
handles = [rects1, rects2]

# Legend 1: Build Times
labels_build = [f'RasterScan2D: {build_raster} ms', f'CompactScan: {build_compact} ms']
legend_build = ax.legend(handles, labels_build, 
                         title="Build Times", 
                         loc='upper left',
                         bbox_to_anchor=(0, 1),  # Top-left corner
                         frameon=True, framealpha=1, edgecolor='black', fancybox=False)
plt.setp(legend_build.get_title(), fontsize=10, fontweight='bold')

# Add the first legend manually so the second doesn't overwrite it
ax.add_artist(legend_build)

# Legend 2: Avg Query Times
labels_query = [f'RasterScan2D: {avg_raster:.2f} ms', f'CompactScan: {avg_compact:.2f} ms']
legend_query = ax.legend(handles, labels_query, 
                         title="Avg Query Time", 
                         loc='upper left',
                         bbox_to_anchor=(0, 0.78),  # Positioned below the first legend
                         frameon=True, framealpha=1, edgecolor='black', fancybox=False)
plt.setp(legend_query.get_title(), fontsize=10, fontweight='bold')

# Ensure the plot frame (spines) are visible and black
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

plt.tight_layout()

# --- 7. Save to File ---
print("Saving charts to file...")
plt.savefig('res.png', dpi=300, bbox_inches='tight')
plt.savefig('res.pdf', bbox_inches='tight')
print("Done. Saved as 'res.png' and 'res.pdf'")