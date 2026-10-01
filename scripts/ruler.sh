#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN=${PYTHON_BIN:-python}
cd "$REPO_DIR/evaluation/RULER/scripts"
export PYTHON_BIN METHOD=adamas
METHOD=full METHOD_TAG=full METHOD_BUDGET= \
    CUDA_VISIBLE_DEVICES="${GPU:-0}" bash run.sh "${MODEL:-llama3.1-8b-instruct}" synthetic
for budget in ${BUDGETS:-256 512 1024 2048 4096}; do
    export ADAMAS_TOKEN_BUDGET=$budget METHOD_TAG="adamas_${budget}"
    CUDA_VISIBLE_DEVICES="${GPU:-0}" bash run.sh "${MODEL:-llama3.1-8b-instruct}" synthetic
done
