import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "figure.dpi": 300
})

x = [64, 128, 256, 512, 1024, 2048, 4096]
methods = ["Adamas (Ours)", "Quest", "StreamingLLM"]
datasets = ["Qasper", "HotpotQA", "GovReport", "TriviaQA", "NarrativeQA", "MultifieldQA"]

data = {
    "GovReport": {
        "StreamingLLM": [1.21, 7.13, 14.52, 18.86, 21.72, 24.08, 27.01],
        "Quest": [3.3, 11.93, 22.84, 27.27, 29.89, 31.19, 31.23],
        "Adamas (Ours)": [30.41, 30.44, 30.37, 31.18, 30.77, 31, 31.07],
    },
    "HotpotQA": {
        "StreamingLLM": [3.63, 9.68, 14.51, 16.25, 18.62, 21.83, 24.14],
        "Quest": [5.31, 15.49, 21.54, 26.64, 30.29, 32.21, 32.93],
        "Adamas (Ours)": [32.18, 32.89, 32.3, 31.45, 33.06, 32.56, 31.41],
    },
    "MultifieldQA": {
        "StreamingLLM": [2.70, 11.67, 17.93, 20.81, 21.62, 27.07, 34.21],
        "Quest": [6.99, 20.87, 31.12, 39.8, 42.03, 44.09, 43.25],
        "Adamas (Ours)": [36.6, 40.67, 42.49, 42.93, 42.22, 40.77, 41.83],
    },
    "NarrativeQA": {
        "StreamingLLM": [0.92, 4.47, 9.17, 10.61, 12.46, 16.85, 17.40],
        "Quest": [1.86, 5.98, 14.2, 16.28, 17.91, 19.66, 19.88],
        "Adamas (Ours)": [17.81, 18.52, 17.36, 18.59, 18.74, 20.14, 20.22],
    },
    "Qasper": {
        "StreamingLLM": [3.15, 8.72, 10.87, 11.74, 14.79, 17.50, 25.33],
        "Quest": [8.67, 18.54, 26.72, 30.89, 31.01, 31.86, 29.79],
        "Adamas (Ours)": [28.33, 29.38, 30.68, 30.93, 29.94, 29.26, 28.85],
    },
    "TriviaQA": {
        "StreamingLLM": [8.73, 33.14, 51.20, 60.37, 68.48, 75.67, 80.21],
        "Quest": [12.29, 44.68, 68.92, 81.32, 83.95, 85.84, 84.94],
        "Adamas (Ours)": [78.91, 82.95, 84.67, 83.99, 83.36, 83.75, 83.95],
    },
}


# Full baseline
full_line = {
    "Qasper": 28.89, "HotpotQA": 31.07, "GovReport": 31.12,
    "TriviaQA": 84.25, "NarrativeQA": 21.23, "MultifieldQA": 41.64
}

colors = {
    "StreamingLLM": "#f49e39",
    "Quest": "#1f77b4",
    "Adamas (Ours)": "#d62728",
}

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
axes = axes.flatten()

for i, (dataset, results) in enumerate(data.items()):
    ax = axes[i]

    for method, values in results.items():
        ax.plot(x[:len(values)], values, marker=".", label=method, color=colors[method])

    ax.axhline(full_line[dataset], ls="--", color="green", label="Full attention")

    ax.set_title(dataset)
    ax.set_ylabel("F1 score", fontsize=12)
    ax.set_xlabel("Token budget", fontsize=12)

    ax.set_xscale("log", base=2)
    ax.set_xticks(x[:len(values)])
    ax.set_xticklabels(x[:len(values)], rotation=45, ha="center", fontsize=12)

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="best", frameon=True)

plt.tight_layout(rect=[0, 0, 1, 1])  # 给 legend 腾空间
plt.savefig("/data0/ysy/Adamas/results/longbench.png", bbox_inches="tight", pad_inches=0.02)
plt.savefig("/data0/ysy/Adamas/results/longbench.pdf", bbox_inches="tight", pad_inches=0.02)
