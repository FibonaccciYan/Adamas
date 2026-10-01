#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN=${PYTHON_BIN:-python}
cd "$REPO_DIR/evaluation/passkey"
MODEL_PATH=${MODEL_PATH:-NousResearch/Yarn-Llama-2-7b-128k}
output="results/${MODEL_PATH##*/}"
mkdir -p "$output"
for budget in ${BUDGETS_PASSKEY:-16 32 64 128 256 512 1024 2048 4096}; do
    "$PYTHON_BIN" -u passkey.py -m "$MODEL_PATH" --iterations "${ITERATIONS:-100}" \
        --fixed-length "${CONTEXT_LENGTH:-100000}" --Adamas --token_budget "$budget" \
        --chunk_size 1 --output-file "$output/Adamas-${budget}.jsonl"
done
