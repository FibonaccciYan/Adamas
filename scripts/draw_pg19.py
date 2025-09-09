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

token_budget = 256

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

# ==== Plot ====
max_len = min(32000, len(ppl_curve_HSA), len(ppl_curve_Quest))

plt.figure(figsize=(8, 5))
plt.plot(range(1, max_len + 1), ppl_curve_HSA[:max_len].numpy(), label="HSA")
plt.plot(range(1, max_len + 1), ppl_curve_Quest[:max_len].numpy(), label="Quest")
plt.xlabel("Sequence Length")
plt.ylabel("Perplexity (PPL)")
plt.title(f"PPL vs Sequence Length (token budget={token_budget})")
plt.ylim(4, 16)
plt.grid(True)
plt.legend()
plt.savefig(HSA_path + f"results/ppl_curve_HSA_Quest_{token_budget}.png")
plt.close()
