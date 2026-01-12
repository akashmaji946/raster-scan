import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
queries = np.arange(1, 11)
labels = [f'Q{i}' for i in queries]

# Query Times (ms)
t_raster = np.array([19.512, 34.846, 50.278, 64.768, 79.632, 94.306, 108.936, 123.442, 137.910, 152.325])
t_compact = np.array([18.047, 33.920, 49.205, 64.077, 78.982, 93.924, 108.687, 123.288, 137.860, 152.359])

# Build Times (ms)
build_raster = 409.2
build_compact = 399.7

# Calculate Average Query Times
avg_raster = np.mean(t_raster)
avg_compact = np.mean(t_compact)

# Calculate Speedups (Raster / Compact)
speedup_build = build_raster / build_compact
speedup_avg_query = avg_raster / avg_compact

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

# --- 4. Annotations (Speedup above bars) ---
speedups = t_raster / t_compact
for i, speedup in enumerate(speedups):
    height = max(t_raster[i], t_compact[i])
    ax.annotate(f'{speedup:.2f}x',
                xy=(x[i], height),
                xytext=(0, 4), 
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='normal', color='black')

# --- 5. Labels and Titles ---
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Queries (Varying Selectivity)')
ax.set_title('750M Uniform Dataset [2048 Resolution]')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(t_raster) * 1.15)

# --- 6. Creating TWO Legends with Speedups in Title ---

# Define the handles
handles = [rects1, rects2]

# --- Legend 1: Build Times ---
labels_build = [f'RasterScan2D: {build_raster} ms', f'CompactScan: {build_compact} ms']
# Dynamically add speedup to title
title_build = f"Index Build Time ({speedup_build:.2f}x)"

legend_build = ax.legend(handles, labels_build, 
                         title=title_build, 
                         loc='upper left',
                         bbox_to_anchor=(0, 1), 
                         frameon=True, framealpha=1, edgecolor='black', fancybox=False)
plt.setp(legend_build.get_title(), fontsize=10, fontweight='normal') # Bold title looks better

ax.add_artist(legend_build)

# --- Legend 2: Mean Query Times ---
labels_query = [f'RasterScan2D: {avg_raster:.2f} ms', f'CompactScan: {avg_compact:.2f} ms']
# Dynamically add speedup to title
title_query = f"Mean Query Time ({speedup_avg_query:.2f}x)"

legend_query = ax.legend(handles, labels_query, 
                         title=title_query, 
                         loc='upper left',
                         bbox_to_anchor=(0, 0.78), 
                         frameon=True, framealpha=1, edgecolor='black', fancybox=False)
plt.setp(legend_query.get_title(), fontsize=10, fontweight='normal')

# Ensure plot frame is visible
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

plt.tight_layout()

# --- 7. Save to File ---
print("Saving charts to file...")
plt.savefig('res.png', dpi=300, bbox_inches='tight')
plt.savefig('res.pdf', bbox_inches='tight')
print("Done. Saved as 'res.png' and 'res.pdf'")