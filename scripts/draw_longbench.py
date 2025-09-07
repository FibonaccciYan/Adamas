import matplotlib.pyplot as plt

# Matplotlib 全局风格设置（适合论文）
plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",   # 论文常用字体
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300
})

# 模拟数据
x = [64, 128, 256, 512, 1024, 2048, 4096]
methods = ["HSA", "Quest"]
datasets = ["Qasper", "HotpotQA", "GovReport", "TriviaQA", "NarrativeQA", "MultifieldQA"]

data = {
    "GovReport": {
        "HSA": [30.41, 30.44, 30.37, 31.18, 30.77, 31, 31.07],
        "Quest": [3.3, 11.93, 22.84, 27.27, 29.89, 31.19, 31.23],
    },
    "HotpotQA": {
        "HSA": [32.18, 32.89, 32.3, 31.45, 33.06, 32.56, 31.41],
        "Quest": [5.31, 15.49, 21.54, 26.64, 30.29, 32.21, 32.93],
    },
    "MultifieldQA": {
        "HSA": [36.6, 40.67, 42.49, 42.93, 42.22, 40.77, 41.83],
        "Quest": [6.99, 20.87, 31.12, 39.8, 42.03, 44.09, 43.25],
    },
    "NarrativeQA": {
        "HSA": [17.81, 18.52, 17.36, 18.59, 18.74, 20.14, 20.22],
        "Quest": [1.86, 5.98, 14.2, 16.28, 17.91, 19.66, 19.88],
    },
    "Qasper": {
        "HSA": [28.33, 29.38, 30.68, 30.93, 29.94, 29.26, 28.85],
        "Quest": [8.67, 18.54, 26.72, 30.89, 31.01, 31.86, 29.79],
    },
    "TriviaQA": {
        "HSA": [78.91, 82.95, 84.67, 83.99, 83.36, 83.75, 83.95],
        "Quest": [12.29, 44.68, 68.92, 81.32, 83.95, 85.84, 84.94],
    },
}


# Full baseline
full_line = {
    "Qasper": 28.89, "HotpotQA": 31.07, "GovReport": 31.12,
    "TriviaQA": 84.25, "NarrativeQA": 21.23, "MultifieldQA": 41.64
}

# 颜色和样式
colors = {
    "HSA": "#d62728",
    "Quest": "#1f77b4",
}

# 画 2x3 子图
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
axes = axes.flatten()

for i, (dataset, results) in enumerate(data.items()):
    ax = axes[i]

    for method, values in results.items():
        ax.plot(x[:len(values)], values, marker=".", label=method, color=colors[method])

    # baseline
    ax.axhline(full_line[dataset], ls="--", color="green", label="Full")

    # 设置标题
    ax.set_title(dataset)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_xlabel("KV Cache Budget", fontsize=12)

    # 只显示逆时针旋转 45° 的 X 轴刻度
    ax.set_xscale("log", base=2)
    ax.set_xticks(x[:len(values)])
    ax.set_xticklabels(x[:len(values)], rotation=45, ha="center", fontsize=12)

    # 右边框和上边框保留实线
    for spine in ax.spines.values():
        spine.set_visible(True)

    # 网格
    ax.grid(True, linestyle="--", alpha=0.5)

    # 每个子图单独 legend
    ax.legend(fontsize=9, loc="best", frameon=True)

# # 图例只放一个
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 1])  # 给 legend 腾空间
plt.savefig("/data0/ysy/HSA/results/longbench.png")
plt.savefig("/data0/ysy/HSA/results/longbench.pdf")
