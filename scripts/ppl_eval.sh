#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN=${PYTHON_BIN:-python}
cd "$REPO_DIR/evaluation/pg19"
MODEL_PATH=${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}
"$PYTHON_BIN" -u ppl_eval.py --model_name_or_path "$MODEL_PATH" \
    --output_dir "results/${MODEL_PATH##*/}" --num_eval_tokens "${NUM_EVAL_TOKENS:-32000}"
for budget in ${BUDGETS:-256 512 1024 2048 4096}; do
    "$PYTHON_BIN" -u ppl_eval.py --model_name_or_path "$MODEL_PATH" \
        --output_dir "results/${MODEL_PATH##*/}" --num_eval_tokens "${NUM_EVAL_TOKENS:-32000}" \
        --Adamas --token_budget "$budget"
done
