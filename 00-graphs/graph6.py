import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
# Labels using K notation for readability
batch_labels = ['5', '50', '500', '5K', '50K', '500K']

# Data from prompt
delete_times = np.array([1.362, 1.995, 9.968, 29.879, 33.002, 328.381])
insert_times = np.array([1.250, 1.179, 1.642, 1.396, 1.149, 1.594])
build_time = 489.1

# Calculate Total Time for annotations
total_times = delete_times + insert_times

# --- 2. Plot Configuration ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'axes.axisbelow': True
})

fig, ax = plt.subplots(figsize=(10, 6))

# --- 3. Creating Stacked Bars ---
# Bar 1: Delete (Bottom) - Light Red
p1 = ax.bar(batch_labels, delete_times, 
            label='Avg Delete Time', 
            color='#FF9999',      
            edgecolor='black', 
            hatch='///',          
            linewidth=1.0)

# Bar 2: Insert (Top) - Light Green
# Stacked on top of delete_times
p2 = ax.bar(batch_labels, insert_times, 
            bottom=delete_times, 
            label='Avg Insert Time', 
            color='#99FF99',      
            edgecolor='black', 
            hatch='...',          
            linewidth=1.0)

# --- 4. Log Scale Configuration ---
ax.set_yscale('log')

# --- 5. Annotations ---
# A. Total Time on top of bars
for i, total in enumerate(total_times):
    ax.annotate(f'{total:.1f}',
                xy=(i, total),
                xytext=(0, 5),    
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='black')

# B. Horizontal Line for Build Time
ax.axhline(y=build_time, color='red', linestyle='--', linewidth=1.5, label=f'Build Time: {build_time} ms')

# Text label for the Build Time line
ax.text(len(batch_labels)-1, build_time * 1.1, 
        f'Index Build Time: {build_time} ms', 
        color='red', fontsize=10, ha='right', va='bottom', fontweight='bold')

# --- 6. Labels and Legends ---
ax.set_xlabel('Batch Size')
ax.set_ylabel('Time (ms) [Log Scale]')
ax.set_title('Update Performance: Delete vs Insert Time (Log Scale) [500M Normal]')

# Custom Y-ticks to make log scale easier to read
# Ticks chosen to cover the range (1ms to ~500ms)
ax.set_yticks([1, 10, 100, 489.1, 1000])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

# Add legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')

# Set Y-limits to ensure everything fits (from slightly below 1ms to above 1000ms)
ax.set_ylim(0.8, 1200)

plt.tight_layout()

# --- 7. Save ---
output_file = 'batch_update_performance_log_final.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.savefig('batch_update_performance_log_final.pdf', bbox_inches='tight')
print(f"Graph saved as {output_file}")