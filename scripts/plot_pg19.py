import torch
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",   # 论文常用字体
    "figure.dpi": 300
})

HSA_path = "/data0/ysy/HSA/"
HSA_data_path = HSA_path + "evaluation/pg19/results/longchat-7b-v1.5-32k/"

Quest_path = "/data0/ysy/quest/"
Quest_data_path = Quest_path + "evaluation/pg19/results/longchat-7b-v1.5-32k/"

token_budgets = [256, 512, 1024, 2048]
y_upper_lims = [16, 10, 9, 9]


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
axes = axes.flatten()

for i, (token_budget, ax) in enumerate(zip(token_budgets, axes)):
    # ==== HSA ====
    with open(HSA_data_path + f"log_HSA_{token_budget}.txt", "r") as f:
        nlls_HSA = [float(line.strip()) for line in f if line.strip()]
    nlls_HSA = torch.tensor(nlls_HSA)
    cumsum_HSA = torch.cumsum(nlls_HSA, dim=0)
    lengths = torch.arange(1, len(nlls_HSA) + 1)
    mean_nll_HSA = cumsum_HSA / lengths
    ppl_curve_HSA = torch.exp(mean_nll_HSA)

    # ==== Quest ====
    with open(Quest_data_path + f"log_quest_{token_budget}.txt", "r") as f:
        nlls_Quest = [float(line.strip()) for line in f if line.strip()]
    nlls_Quest = torch.tensor(nlls_Quest)
    cumsum_Quest = torch.cumsum(nlls_Quest, dim=0)
    lengths = torch.arange(1, len(nlls_Quest) + 1)
    mean_nll_Quest = cumsum_Quest / lengths
    ppl_curve_Quest = torch.exp(mean_nll_Quest)

    # ==== StreamingLLM ====
    with open(HSA_data_path + f"log_streamingLLM_{token_budget}.txt", "r") as f:
        nlls_streamingLLM = [float(line.strip()) for line in f if line.strip()]
    nlls_streamingLLM = torch.tensor(nlls_streamingLLM)
    cumsum_streamingLLM = torch.cumsum(nlls_streamingLLM, dim=0)
    lengths = torch.arange(1, len(nlls_streamingLLM) + 1)
    mean_nll_streamingLLM = cumsum_streamingLLM / lengths
    ppl_curve_streamingLLM = torch.exp(mean_nll_streamingLLM)

    # ==== Full ====
    with open(HSA_data_path + f"log_full.txt", "r") as f:
        nlls_full = [float(line.strip()) for line in f if line.strip()]
    nlls_full = torch.tensor(nlls_full)
    cumsum_full = torch.cumsum(nlls_full, dim=0)
    lengths = torch.arange(1, len(nlls_full) + 1)
    mean_nll_full = cumsum_full / lengths
    ppl_curve_full = torch.exp(mean_nll_full)

    # ==== Plot ====
    max_len = min(32000, len(ppl_curve_HSA), len(ppl_curve_Quest))

    ax.plot(range(1, max_len + 1), ppl_curve_streamingLLM[:max_len].numpy(), label="StreamingLLM", color="#ff7f0e")
    ax.plot(range(1, max_len + 1), ppl_curve_Quest[:max_len].numpy(), label="Quest", color="#1f77b4")
    ax.plot(range(1, max_len + 1), ppl_curve_HSA[:max_len].numpy(), label="HSA", color="#d62728")
    ax.plot(range(1, max_len + 1), ppl_curve_full[:max_len].numpy(), label="Full Attn", color="#2ca02c")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Perplexity (PPL)")
    ax.set_title(f"Token budget: {token_budget}")
    ax.set_ylim(6.5, y_upper_lims[i])
    ax.grid(True)

# 图例只放一个
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # 给 legend 腾空间
plt.savefig(HSA_path + f"results/ppl_curve_HSA_Quest_256-2048.png")
plt.savefig(HSA_path + f"results/ppl_curve_HSA_Quest_256-2048.pdf")
