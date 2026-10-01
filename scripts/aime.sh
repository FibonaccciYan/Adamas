#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN=${PYTHON_BIN:-python}
model_args=()
[[ -z ${MODEL_PATH:-} ]] || model_args=(--model_path "$MODEL_PATH")
cd "$REPO_DIR/evaluation/AIME"
for benchmark in ${BENCHMARKS:-aime2024 aime2025}; do
    for ((repeat=1; repeat<=${REPEATS:-3}; repeat++)); do
        "$PYTHON_BIN" -u pred.py --benchmark "$benchmark" --model Qwen3-8b "${model_args[@]}" \
            --max_new_tokens "${MAX_NEW_TOKENS:-38912}" --thinking \
            --output_dir "pred/${benchmark}/run_${repeat}"
        for budget in ${BUDGETS_AIME:-1024 2048 4096}; do
            "$PYTHON_BIN" -u pred.py --benchmark "$benchmark" --model Qwen3-8b "${model_args[@]}" \
                --max_new_tokens "${MAX_NEW_TOKENS:-38912}" --thinking --Adamas \
                --token_budget "$budget" --chunk_size 1 --output_dir "pred/${benchmark}/run_${repeat}"
        done
    done
done
