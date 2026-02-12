import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ----------------------------
# Data
# ----------------------------
queries = [f"Q{i}" for i in range(1, 11)]

import numpy as np

import numpy as np

import numpy as np

import numpy as np

import numpy as np



import numpy as np

# RasterScan timings
rasterscan_times = np.array([
    2.131, 2.333, 2.522, 2.754, 2.874,
    3.098, 3.171, 3.279, 3.783, 3.878
])

# EquiDepth timings
equidepth_times = np.array([
    2.970, 3.016, 3.112, 3.151, 3.338,
    3.485, 3.708, 3.817, 4.078, 4.382
])

# Build times
rasterscan_build = 14.53
equidepth_build = 40.29






# ----------------------------
# Speedups (RasterScan / EquiDepth)
# ----------------------------
speedup = rasterscan_times / equidepth_times
avg_query_speedup = np.mean(speedup)
build_speedup = rasterscan_build / equidepth_build

# ----------------------------
# Plot setup
# ----------------------------
x = np.arange(len(queries))
width = 0.40 # thin bars

fig, ax = plt.subplots(figsize=(14, 5))

bars_raster = ax.bar(
    x - width / 2,
    rasterscan_times,
    width,
    hatch="///",
    edgecolor="black",
    color="orange",
    label="RasterScan2D"
)

bars_equidepth = ax.bar(
    x + width / 2,
    equidepth_times,
    width,
    hatch="...",
    edgecolor="black",
    color="steelblue",
    label="EquiDepthIndex"
)

# ----------------------------
# Y-axis headroom
# ----------------------------
ymax_all = max(np.max(rasterscan_times), np.max(equidepth_times))
ax.set_ylim(0, ymax_all * 1.5)

# ----------------------------
# Execution time annotations (inside bars)
# ----------------------------
def annotate_bars(bars, values):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.5,
            f"{val:.1f}",
            ha="center",
            va="center",
            rotation=45,
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="black",
                lw=0.5,
                alpha=0.85
            )
        )

annotate_bars(bars_raster, rasterscan_times)
annotate_bars(bars_equidepth, equidepth_times)

# ----------------------------
# Per-query speedup annotations
# ----------------------------
for i in range(len(queries)):
    ymax = max(rasterscan_times[i], equidepth_times[i])
    ax.text(
        x[i],
        ymax * 1.10,
        f"{speedup[i]:.2f}×",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# ----------------------------
# Axes, title, grid
# ----------------------------
ax.set_xlabel("Queries")
ax.set_ylabel("Time (ms)")
ax.set_xticks(x)
ax.set_xticklabels(queries)
ax.set_title("Query Performance (500M TPC-C) — RasterScan vs EquiDepth")
ax.grid(axis="y", linestyle="--", alpha=0.6)

# ----------------------------
# Legends
# ----------------------------

# Query legend (with avg speedup)
legend_query_handles = [
    Patch(facecolor="orange", edgecolor="black", hatch="///", label="RasterScan2D"),
    Patch(facecolor="steelblue", edgecolor="black", hatch="...", label="EquiDepthIndex"),
    Patch(facecolor="none", edgecolor="none",
          label=f"Avg Query Speedup: {avg_query_speedup:.2f}×")
]

legend_query = ax.legend(
    handles=legend_query_handles,
    loc="upper left",
    title="Query Execution Time"
)

# Build-time legend
build_legend_handles = [
    Patch(facecolor="orange", edgecolor="black", hatch="///",
          label=f"RasterScan Build: {rasterscan_build:.2f} ms"),
    Patch(facecolor="steelblue", edgecolor="black", hatch="...",
          label=f"EquiDepth Build: {equidepth_build:.2f} ms"),
    Patch(facecolor="none", edgecolor="none",
          label=f"Build Speedup: {build_speedup:.2f}×")
]

legend_build = ax.legend(
    handles=build_legend_handles,
    loc="upper center",
    title="Index Build Time"
)

ax.add_artist(legend_query)

# ----------------------------
# Save figure
# ----------------------------
plt.tight_layout()
plt.savefig("img.png", dpi=300, bbox_inches="tight")
# plt.show()
