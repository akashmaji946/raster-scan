import matplotlib.pyplot as plt

# -----------------------------
# Helper: draw table as image
# -----------------------------
def save_table_image(data, col_labels, title, filename):
    fig, ax = plt.subplots(figsize=(len(col_labels) * 1.2, len(data) * 0.5))
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# -----------------------------
# PointsBuffer
# -----------------------------
points_buffer = [
    [0,  0,  0,  0],
    [1,  1,  2,  3],
    [2,  2,  3,  4],
    [3,  1,  2,  7],
    [4,  4,  5,  6],
    [5,  5,  6,  7],
    [6,  6,  7,  8],
    [7,  7,  8,  9],
    [8,  8,  9, 10],
    [9,  9, 10, 11],
    [10, 10, 11, 12],
    [11, 11, 12, 13],
    [12, 12, 13, 14],
    [13, 14, 15, 15],
]

save_table_image(
    points_buffer,
    ["RID", "X", "Y", "Z"],
    "PointsBuffer",
    "points_buffer.png"
)

# -----------------------------
# Count (RID % 4)
# -----------------------------
NUM_BUCKETS = 4
count = [0] * NUM_BUCKETS

for row in points_buffer:
    count[row[0] % NUM_BUCKETS] += 1

count_table = [[i, count[i]] for i in range(NUM_BUCKETS)]

save_table_image(
    count_table,
    ["Bucket", "Count"],
    "Count (RID % 4)",
    "count.png"
)

# -----------------------------
# Prefix Sum (exclusive)
# -----------------------------
prefix_sum = []
running = 0
for c in count:
    prefix_sum.append(running)
    running += c

prefix_table = [[i, prefix_sum[i]] for i in range(NUM_BUCKETS)]

save_table_image(
    prefix_table,
    ["Bucket", "PrefixSum"],
    "Prefix Sum (Exclusive)",
    "prefix_sum.png"
)

# -----------------------------
# dataBuffer (scatter)
# -----------------------------
write_pos = prefix_sum[:]
data_buffer = [None] * len(points_buffer)

for rid, x, y, z in points_buffer:
    bucket = rid % NUM_BUCKETS
    idx = write_pos[bucket]
    data_buffer[idx] = [x, y, z, rid]
    write_pos[bucket] += 1

save_table_image(
    data_buffer,
    ["X", "Y", "Z", "rowId"],
    "dataBuffer",
    "data_buffer.png"
)

print("PNG tables generated successfully.")
