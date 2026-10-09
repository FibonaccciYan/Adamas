#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-/home/ysy/anaconda3/envs/qwen3/bin/python}
MODEL_PATH=${MODEL_PATH:-/data1/model/qwen/Qwen/Qwen3-8B}
GPU=${GPU:-0}

BUDGET_POOL=(256 512 1024 2048 4096 133120)
CONTEXT_POOL=(102400 131072)
OUTPUT_PATH="${ROOT_DIR}/test_results/e2e-qwen3/Qwen3-8B"
mkdir -p "${OUTPUT_PATH}"

for budget in "${BUDGET_POOL[@]}"; do
    for context in "${CONTEXT_POOL[@]}"; do
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${ROOT_DIR}/e2e-qwen3/bench_textgen.py" \
            --model_path "${MODEL_PATH}" \
            --context_len "${context}" \
            --decode_len 256 \
            --token_budget "${budget}" \
            --page_size 1 \
            --output_file "${OUTPUT_PATH}/e2e_Adamas.txt" \
            > "${OUTPUT_PATH}/log_${budget}_${context}.log" 2>&1
    done
done
