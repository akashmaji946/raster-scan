import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
# resolutions = ['1024', '2048', '4096', '8192']
# x_labels = [f'{res}' for res in resolutions]

resolutions = ['1024', '2048', '4096', '8192']
x_labels = [f'{res}' for res in resolutions]

# Build Times (ms)
build_raster = np.array([2.6, 2.6, 3.5, 7.8])
build_compact = np.array([4.6, 5.0, 7.7, 22.7])

# --- 2. Plot Configuration ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True 
})

fig, ax = plt.subplots()

x = np.arange(len(resolutions))
# CHANGE: Increased width to make bars wider
width = 0.40  

# --- 3. Creating Bars ---
# RasterScan2D (Light Blue)
rects1 = ax.bar(x - width/2, build_raster, width,
                label='RasterScan2D',
                color='#CCEBFF',
                edgecolor='black',
                linewidth=1.2,
                hatch='')

# Compact Index (Light Orange)
rects2 = ax.bar(x + width/2, build_compact, width,
                label='Compact Index',
                color='#FFD1A4',
                edgecolor='black',
                linewidth=1.2,
                hatch='')

# --- 4. Writing Values Inside Bars ---
def add_value_labels(rects):
    for rect in rects:
        height = rect.get_height()
        # Place text in the center of the bar
        # Rotation=90 ensures it fits nicely in narrow/short bars
        ax.text(rect.get_x() + rect.get_width()/2., height/2,
                f'{height:.1f}',
                ha='center', va='center', rotation=40, 
                fontsize=10, fontweight='bold', color='black')

add_value_labels(rects1)
add_value_labels(rects2)

# --- 5. Annotations (Speedup on Top) ---
# Speedup = Raster / Compact
ratios = build_raster / build_compact

for i, ratio in enumerate(ratios):
    height = max(build_raster[i], build_compact[i])
    # Place speedup text slightly above the bar
    ax.annotate(f'{ratio:.2f}x',
                xy=(x[i], height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='black')

# --- 6. Labels and Titles ---
ax.set_ylabel('Build Time (ms)')
ax.set_xlabel('Resolution')
ax.set_title('Index Build Time vs Resolution (1M Zipf1.1 Dataset)')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)

# Set Y-axis limit to accommodate annotations
ax.set_ylim(0, max(build_compact) * 1.2)

# --- 7. Legend and Save ---
legend = ax.legend(loc='upper left', frameon=True, framealpha=1, 
                   edgecolor='black', fancybox=False)

# Ensure borders are visible
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

plt.tight_layout()

output_file = 'build_time_resolution_wide.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.savefig('build_time_resolution_wide.pdf', bbox_inches='tight')
print(f"Graph saved as {output_file}")