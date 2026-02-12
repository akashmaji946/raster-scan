import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
queries = np.arange(1, 11)
labels = [f'Q{i}' for i in queries]

# Data from prompt (10% updates)
t_before = np.array([1.400, 1.722, 1.815, 1.985, 2.136, 2.265, 2.385, 2.511, 2.628, 2.739])
t_after = np.array([2.180, 2.172, 2.252, 2.337, 2.317, 2.388, 2.482, 2.588, 2.690, 2.734])


# Calculate Averages
avg_before = np.mean(t_before)
avg_after = np.mean(t_after)

# --- 2. Plot Configuration ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True 
})

fig, ax = plt.subplots()

x = np.arange(len(labels))
width = 0.35

# --- 3. Creating Bars ---
# Before Updates (Light Blue)
rects1 = ax.bar(x - width/2, t_before, width,
                label=f'Before Updates (Avg: {avg_before:.2f} ms)',
                color='#CCEBFF',
                edgecolor='black',
                linewidth=1.2,
                hatch='///')

# After Updates (Light Orange)
rects2 = ax.bar(x + width/2, t_after, width,
                label=f'After Updates (Avg: {avg_after:.2f} ms)',
                color='#FFD1A4',
                edgecolor='black',
                linewidth=1.2,
                hatch='...')

# --- 4. Annotations (Ratio: After / Before) ---
# Shows performance stability (1.0x = No change)
ratios = t_after / t_before

for i, ratio in enumerate(ratios):
    height = max(t_before[i], t_after[i])
    ax.annotate(f'{ratio:.2f}x',
                xy=(x[i], height),
                xytext=(0, 4), 
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, fontweight='normal', color='black')

# --- 5. Labels and Titles ---
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Queries (Varying Selectivity)')
ax.set_title('10M Zipf1.1 dataset [10% updates]')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(t_after) * 1.3)

# --- 6. Legend ---
legend = ax.legend(loc='upper left', frameon=True, framealpha=1, 
                   edgecolor='black', fancybox=False)
plt.setp(legend.get_title(), fontsize=10, fontweight='bold')

# Ensure plot frame is visible
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

plt.tight_layout()

# --- 7. Save ---
print("Saving chart...")
plt.savefig('update_performance.png', dpi=300, bbox_inches='tight')
plt.savefig('update_performance.pdf', bbox_inches='tight')
print("Done. Saved as 'update_performance.png' and 'update_performance.pdf'")