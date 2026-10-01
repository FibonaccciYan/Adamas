#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN=${PYTHON_BIN:-python}
model_args=()
[[ -z ${MODEL_PATH:-} ]] || model_args=(--model_path "$MODEL_PATH")
cd "$REPO_DIR"
MODEL=${MODEL:-Llama-3.1-8B-Instruct}
output="test_results/${MODEL}"
mkdir -p "$output"
for context in ${CONTEXT_LENGTHS:-32768}; do
    # A budget covering the allocated cache selects the existing full-attention path.
    full_budget=$((context + ${DECODE_LENGTH:-256} + 512))
    "$PYTHON_BIN" -u scripts/bench_textgen.py --model "$MODEL" "${model_args[@]}" \
        --context_len "$context" --decode_len "${DECODE_LENGTH:-256}" \
        --token_budget "$full_budget" --page_size 1 --iteration "${ITERATIONS:-10}" \
        > "$output/log_full_${context}.log" 2>&1
done
for budget in ${BUDGETS:-256 512 1024 2048 4096}; do
    for context in ${CONTEXT_LENGTHS:-32768}; do
        "$PYTHON_BIN" -u scripts/bench_textgen.py --model "$MODEL" "${model_args[@]}" \
            --context_len "$context" --decode_len "${DECODE_LENGTH:-256}" \
            --token_budget "$budget" --page_size 1 --iteration "${ITERATIONS:-10}" \
            > "$output/log_${budget}_${context}.log" 2>&1
    done
done
