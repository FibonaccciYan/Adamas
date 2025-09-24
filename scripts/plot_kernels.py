import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
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
    "Similarity estimation": "#56B1E4",
    "Top-$k$ selection": "#1379DF",      
    "Approximate attention": "#2049BE"
}

y_ticks = [256, 512, 1024, 2048, 4096, 102400]
y_labels = ["256", "512", "1024", "2048", "4096", "Full"]
ratio_pad = [20, 40, 80]

context_lens = sorted(df["context_len"].unique())

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for i, (ax, cl) in enumerate(zip(axes, context_lens)):
    subdf = df[df["context_len"] == cl].sort_values("token_budget").reset_index(drop=True) 
    
    y_pos = np.arange(len(subdf))
    est = subdf["estimate"].values
    topk = subdf["topk"].values
    attn = subdf["approx_attn"].values

    total = est + topk + attn
    
    ax.barh(y_pos, est, color=colors["Similarity estimation"], label="Similarity estimation" if i==0 else "")
    ax.barh(y_pos, topk, left=est, color=colors["Top-$k$ selection"], label="Top-$k$ selection" if i==0 else "")
    ax.barh(y_pos, attn, left=est+topk, color=colors["Approximate attention"], label="Approximate attention" if i==0 else "")
    
    ax.set_ylim(-0.5, len(y_pos)-0.5)
    ax.invert_yaxis()
    
    if i == 0:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([]) 
    
    ax.set_xlabel(f"Latency (us)\n\nSequence length: {cl}")

    ax.xaxis.grid(True, which="major", linestyle="-", color="gray", alpha=0.6)
    ax.set_axisbelow(True) 

    row_256 = subdf[subdf["token_budget"] == 256].index[0]   # 第一行
    row_full = subdf[subdf["token_budget"] == 102400].index[0]  # Full 行
    
    latency_256 = total[row_256]
    latency_full = total[row_full]
    ratio = latency_full / latency_256 if latency_256 > 0 else np.nan

    y_256 = y_pos[row_256]
    
    # 中间数字位置（偏离上下约 0.25 保证不被柱子遮挡）
    mid_x = (latency_full + latency_256) / 2
    ax.text(mid_x, y_256, f"×{ratio:.1f}",
            ha="center", va="center", fontsize=18, fontweight="bold")

    # 左侧箭头（从256柱子末端 -> 中间数字）
    ax.annotate(
        "", 
        xy=(latency_256, y_256), xycoords="data",     # -5 是为了让箭头不要顶到文字
        xytext=(mid_x - ratio_pad[i], y_256), textcoords="data",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2)
    )
    
    # 右侧箭头（从Full柱子末端 -> 中间数字）
    ax.annotate(
        "", 
        xy=(latency_full, y_256), xycoords="data",     # +5 同理
        xytext=(mid_x + ratio_pad[i], y_256), textcoords="data",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2)
    )

fig.legend(["Similarity estimation", "Top-$k$ selection", "Approximate attention"], loc="upper center", ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("/data0/ysy/Adamas/results/kernels.png", bbox_inches="tight", pad_inches=0.02)
plt.savefig("/data0/ysy/Adamas/results/kernels.pdf", bbox_inches="tight", pad_inches=0.02)