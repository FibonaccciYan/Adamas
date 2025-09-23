import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",   # 论文常用字体
    "figure.dpi": 300
})

data = {
    "context_len": [8192, 16384, 32768] * 6,
    "token_budget": [256, 256, 256, 512, 512, 512,
                     1024, 1024, 1024, 2048, 2048, 2048,
                     4096, 4096, 4096, 102400, 102400, 102400],  # 102400 表示 Full
    "approx_attn": [17.645, 20.407, 17.874, 27.906, 25.204, 27.823,
                    39.478, 39.679, 39.545, 64.452, 65.045, 64.571,
                    112.313, 112.316, 113.318, 208.226, 399.745, 778.815],
    "topk": [12.672, 18.649, 28.343, 14.105, 18.477, 28.409, 
             14.235, 19.011, 28.705, 14.826, 19.594, 28.376, 
             16.578, 21.157, 31.745, 0, 0, 0],
    "estimate": [38.996, 70.707, 128.819] * 5 + [0, 0, 0],

}

df = pd.DataFrame(data)

colors = {
    "Estimation": "#56B1E4",
    "Top-K Selection": "#1379DF",      
    "Approximate Attention": "#2049BE"
}

# 显示顺序的 y 轴刻度
y_ticks = [256, 512, 1024, 2048, 4096, 102400]
y_labels = ["256", "512", "1024", "2048", "4096", "Full Attn"]

context_lens = sorted(df["context_len"].unique())

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for i, (ax, cl) in enumerate(zip(axes, context_lens)):
    subdf = df[df["context_len"] == cl].sort_values("token_budget")
    
    y_pos = np.arange(len(subdf))
    est = subdf["estimate"].values
    topk = subdf["topk"].values
    attn = subdf["approx_attn"].values
    
    # 堆叠柱状图
    ax.barh(y_pos, est, color=colors["Estimation"], label="Estimation" if i==0 else "")
    ax.barh(y_pos, topk, left=est, color=colors["Top-K Selection"], label="Top-K Selection" if i==0 else "")
    ax.barh(y_pos, attn, left=est+topk, color=colors["Approximate Attention"], label="Approximate Attention" if i==0 else "")
    
    # y 轴顺序：从上往下递增
    ax.set_ylim(-0.5, len(y_pos)-0.5)
    ax.invert_yaxis()
    
    if i == 0:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])  # 去掉中间和右侧的 y 轴刻度
    
    # context_len 和 latency 放在 x 轴下面
    ax.set_xlabel(f"Latency (us)\n\nSequence Length: {cl}")

    # 在 x 轴刻度处画竖线
    ax.xaxis.grid(True, which="major", linestyle="-", color="gray", alpha=0.6)
    ax.set_axisbelow(True)  # 让网格线在柱子下方

# 图例放上面
fig.legend(["Estimation", "Top-K Selection", "Approximate Attention"], loc="upper center", ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("/data0/ysy/HSA/results/kernels.png")
plt.savefig("/data0/ysy/HSA/results/kernels.pdf")