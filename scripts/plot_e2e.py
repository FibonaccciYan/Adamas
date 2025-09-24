import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",   # 论文常用字体
    "figure.dpi": 300
})

# 构造 DataFrame
data = {
    "page_size": [1]*18,
    "token_budget": [102400,102400,102400,256,256,256,
                     512,512,512,1024,1024,1024,
                     2048,2048,2048,4096,4096,4096],
    "context_len": [8192,16384,32768]*6,
    "decode_len": [256]*18,
    "avg_prefill_latency": [1.2158,2.7430,7.5718,1.2017,2.7177,7.5333,
                            1.2122,2.7382,7.5481,1.2161,2.7424,7.5647,
                            1.2142,2.7391,7.5582,1.2184,2.7414,7.5763],
    "avg_decode_latency": [30.2,38.0,53.7,26.0,29.3,34.9,
                           26.3,29.5,35.2,26.7,30.0,35.8,
                           27.5,30.9,36.9,28.8,32.3,38.6],
}
df = pd.DataFrame(data)

# 选择要画的指标
metric = "avg_decode_latency"   # 或者 "avg_prefill_latency"

# 所有 context_len
contexts = df["context_len"].unique()
# 所有 token_budget
budgets = df["token_budget"].unique()

# 设置柱状图参数
x = np.arange(len(contexts))  # 横轴位置
bar_width = 0.12              # 每根柱子宽度
# colors = ['#E7483D', '#F49E39', '#DF8D44', '#A6519E', '#918AC2', '#6565AE']
# colors = ['#E7483D', '#F5A94D', '#E9E955', '#A1D741', '#51C382', '#439DBD']
colors = ['#E7483D', '#E89838', '#94CE3E', '#45B0BB', '#3367A1', '#6565AE']

fig, ax = plt.subplots(figsize=(12, 6))

bars = {}
for i, b in enumerate(budgets):
    vals = []
    for ctx in contexts:
        row = df[(df["context_len"] == ctx) & (df["token_budget"] == b)]
        if not row.empty:
            vals.append(row[metric].values[0])
        else:
            vals.append(np.nan)
    label = str(b) if b != 102400 else "Full"
    rects = ax.bar(x + i*bar_width, vals, width=bar_width, label=f"{label}", color=colors[i % len(colors)], edgecolor="black", linewidth=1)
    bars[b] = rects


# === 在 32768 分组上加箭头 ===
# 找到 256 和 Full 对应的柱子
bar_256 = bars[256][2]
bar_full = bars[102400][2]

y1 = bar_256.get_height()
y2 = bar_full.get_height()
x1 = bar_256.get_x() + bar_256.get_width()/2
# x2 = bar_full.get_x() + bar_full.get_width()/2
x2 = x1

# 在箭头中间留出间隙
gap = (y2 - y1) * 0.2   # 空隙占总长度 20%
y_mid1 = y1 + (y2 - y1) * 0.4
y_mid2 = y2 - (y2 - y1) * 0.4

# 画下半段箭头（向上）
ax.annotate(
    "", 
    xy=(x1, y1), xytext=(x1, y_mid1),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5)
)

# 画上半段箭头（向下）
ax.annotate(
    "", 
    xy=(x1, y2), xytext=(x1, y_mid2),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5)
)

# 在中间写上倍数
ratio = y2 / y1
ax.text(x1, (y_mid1 + y_mid2) / 2, f"{ratio:.1f}x",
        color="black", ha="center", va="center", fontsize=12,
        bbox=dict(facecolor="none", edgecolor="none", pad=1.0))  # 白底，避免和柱子重叠

# 设置横轴
ax.set_xticks(x + bar_width*(len(budgets)-1)/2)
ax.set_xticklabels([str(c) for c in contexts])

# 标签 & 图例
ax.set_xlabel("Context Length")
ax.set_ylabel("Latency (ms)")
ax.legend()

plt.tight_layout()
plt.savefig("/data0/ysy/Adamas/results/e2e.png")
plt.savefig("/data0/ysy/Adamas/results/e2e.pdf")
