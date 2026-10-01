# RULER Reproduction

This directory adapts [NVIDIA RULER](https://github.com/NVIDIA/RULER) for
Adamas and full-attention evaluation with Hugging Face models.
The upstream Apache-2.0 license and source copyright notices are retained.

## Data Preparation

Install this repository using the root README, then download the source data
from this directory:

```bash
python -m nltk.downloader punkt punkt_tab
(cd scripts/data/synthetic/json && python download_paulgraham_essay.py)
(cd scripts/data/synthetic/json && bash download_qa_dataset.sh)
```

These scripts download Paul Graham essays, SQuAD, and HotpotQA. Generated data
is excluded from Git. The RULER `niah_*` tasks are part of this benchmark.

## Evaluation

From the repository root:

```bash
bash scripts/ruler.sh
MODEL=qwen3-8b PYTHON_BIN=/path/to/qwen3/python bash scripts/ruler.sh
MODEL_PATH=/path/to/Llama-3.1-8B-Instruct bash scripts/ruler.sh
```

The script runs Full and Adamas with budgets 256, 512, 1024, 2048, and 4096.
Default sequence lengths are 4K, 8K, 16K, 32K, 64K, and 128K.
Llama-3.1 uses Transformers 4.45.2; Qwen3 uses 4.51.0 and enables YaRN
when the evaluation length exceeds 32K.

Controls: `GPU`, `BUDGETS`, `RULER_SEQ_LENGTHS` (comma-separated),
`RULER_TASKS` (comma-separated), `RULER_NUM_SAMPLES`, and `ROOT_DIR`.
Task definitions are in `scripts/config_tasks.sh` and `scripts/synthetic.yaml`.
Predictions and summaries are saved under `scripts/benchmark_root` by default.

## Upstream Citation

```bibtex
@article{hsieh2024ruler,
  title={RULER: What's the Real Context Size of Your Long-Context Language Models?},
  author={Cheng-Ping Hsieh and Simeng Sun and Samuel Kriman and Shantanu Acharya and Dima Rekesh and Fei Jia and Yang Zhang and Boris Ginsburg},
  year={2024},
  journal={arXiv preprint arXiv:2404.06654},
}
```
