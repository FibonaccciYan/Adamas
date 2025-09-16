import matplotlib.pyplot as plt

# Matplotlib 全局风格设置（适合论文）
plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",   # 论文常用字体
    "figure.dpi": 300
})

# 模拟数据
x = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
methods = ["HSA", "HSA w/o Hadamard", "HSA w/o Bucketization", "HSA w/ L2"]

data = {
    "GovReport": {
        "HSA": [22.08, 28.31, 30.41, 30.44, 30.37, 31.18, 30.77, 31.00, 31.07],
        "HSA w/o Hadamard": [0.60, 0.59, 0.59, 0.64, 2.06, 6.32, 14.12, 23.53, 29.14],
        "HSA w/o Bucketization": [17.17, 23.67, 27.88, 30.09, 30.72, 31.16, 31.18, 30.84, 30.70],
        "HSA w/ L2": [18.27, 27.19, 30.09, 30.45, 30.86, 30.93, 30.96, 30.50, 30.56],
    },
    "HotpotQA": {
        "HSA": [24.05, 31.65, 32.18, 32.89, 32.30, 31.45, 33.06, 32.56, 31.41],
        "HSA w/o Hadamard": [1.01, 1.04, 0.68, 0.48, 0.77, 2.07, 5.38, 13.47, 25.13],
        "HSA w/o Bucketization": [19.22, 24.70, 29.04, 31.55, 33.79, 31.63, 32.14, 32.47, 31.58],
        "HSA w/ L2": [18.73, 28.83, 32.58, 32.91, 31.15, 30.70, 32.66, 31.86, 31.10],
    },
    "MultifieldQA": {
        "HSA": [21.53, 33.38, 36.60, 40.67, 42.49, 42.93, 42.22, 40.77, 41.83],
        "HSA w/o Hadamard": [2.97, 2.84, 2.51, 5.27, 8.28, 18.26, 27.45, 39.24, 41.29],
        "HSA w/o Bucketization": [23.21, 29.51, 35.64, 38.98, 40.53, 41.99, 42.43, 41.62, 42.01],
        "HSA w/ L2": [23.70, 34.11, 39.81, 41.41, 43.08, 41.60, 41.28, 41.65, 42.33],
    },
    "NarrativeQA": {
        "HSA": [11.98, 15.13, 17.81, 18.52, 17.36, 18.59, 18.74, 20.14, 20.22],
        "HSA w/o Hadamard": [1.68, 1.00, 1.12, 0.91, 0.38, 1.34, 3.29, 6.00, 13.32],
        "HSA w/o Bucketization": [5.38, 9.16, 14.79, 15.18, 17.34, 19.39, 19.94, 20.00, 20.01],
        "HSA w/ L2": [8.83, 14.50, 16.06, 18.39, 18.37, 19.20, 18.54, 19.74, 20.33],
    },
    "Qasper": {
        "HSA": [18.83, 24.52, 28.33, 29.38, 30.68, 30.93, 29.94, 29.26, 28.85],
        "HSA w/o Hadamard": [1.73, 2.40, 2.29, 4.93, 9.24, 18.27, 26.17, 31.08, 30.14],
        "HSA w/o Bucketization": [18.04, 20.81, 24.16, 28.73, 29.50, 30.04, 30.18, 30.52, 28.73],
        "HSA w/ L2": [22.41, 27.04, 28.70, 31.36, 30.35, 31.23, 30.80, 29.36, 28.86],
    },
    "TriviaQA": {
        "HSA": [56.77, 75.13, 78.91, 82.95, 84.67, 83.99, 83.36, 83.75, 83.95],
        "HSA w/o Hadamard": [2.63, 2.08, 1.21, 3.52, 9.73, 20.14, 33.15, 56.62, 79.13],
        "HSA w/o Bucketization": [52.26, 70.62, 80.08, 80.88, 82.59, 83.90, 84.11, 84.22, 83.63],
        "HSA w/ L2": [37.82, 66.80, 80.97, 83.35, 84.21, 84.29, 83.60, 84.03, 84.25],
    },
}


# Full baseline
full_line = {
    "GovReport": 31.12, "HotpotQA": 31.07, "MultifieldQA": 41.64,
    "NarrativeQA": 21.23, "Qasper": 28.89, "TriviaQA": 84.25, 
}

# 颜色和样式
colors = {
    "HSA": "#d62728",
    "HSA w/o Hadamard": "#1f77b4",
    "HSA w/o Bucketization": "#f49e39",
    "HSA w/ L2": "#6565ae",
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

    if dataset == "NarrativeQA":
        # 调整 y limit
        ax.set_ylim(None, 22)

    # 网格
    ax.grid(True, linestyle="--", alpha=0.5)

# 图例只放一个
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # 给 legend 腾空间
plt.savefig("/data0/ysy/HSA/results/ablation.png")
plt.savefig("/data0/ysy/HSA/results/ablation.pdf")
