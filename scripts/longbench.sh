#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN=${PYTHON_BIN:-python}
model_args=()
[[ -z ${MODEL_PATH:-} ]] || model_args=(--model_path "$MODEL_PATH")
cd "$REPO_DIR/evaluation/LongBench"
MODEL=${MODEL:-Meta-Llama-3.1-8B-Instruct}
for task in ${TASKS:-qasper narrativeqa hotpotqa multifieldqa_en triviaqa gov_report}; do
    "$PYTHON_BIN" -u pred.py --model "$MODEL" "${model_args[@]}" --task "$task"
    for budget in ${BUDGETS:-256 512 1024 2048 4096}; do
        "$PYTHON_BIN" -u pred.py --model "$MODEL" "${model_args[@]}" \
            --task "$task" --Adamas --token_budget "$budget" --chunk_size 1
    done
done
"$PYTHON_BIN" -u eval.py --model "$MODEL"
