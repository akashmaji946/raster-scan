import matplotlib.pyplot as plt

# -----------------------------
# Slim table renderer
# -----------------------------
def save_slim_table(data, col_labels, title, filename,
                    col_width=0.8, row_height=0.35, font_size=9):

    fig_width = len(col_labels) * col_width
    fig_height = len(data) * row_height + 0.6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 0.9)

    ax.set_title(title, fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
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

save_slim_table(
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

save_slim_table(
    count_table,
    ["Bucket", "Count"],
    "Count",
    "count.png",
    col_width=1.0
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

save_slim_table(
    prefix_table,
    ["Bucket", "PrefixSum"],
    "PrefixSum",
    "prefix_sum.png",
    col_width=1.1
)

# -----------------------------
# dataBuffer
# -----------------------------
write_pos = prefix_sum[:]
data_buffer = [None] * len(points_buffer)

for rid, x, y, z in points_buffer:
    bucket = rid % NUM_BUCKETS
    idx = write_pos[bucket]
    data_buffer[idx] = [x, y, z, rid]
    write_pos[bucket] += 1

save_slim_table(
    data_buffer,
    ["X", "Y", "Z", "rowId"],
    "dataBuffer",
    "data_buffer.png"
)

print("Slim PNG tables generated.")
