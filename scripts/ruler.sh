#!/bin/bash
set -euo pipefail

GPU=${GPU:-0}
MODEL_NAME=${MODEL_NAME:-llama3.1-8b-instruct}
BENCHMARK=${BENCHMARK:-synthetic}
PYTHONPATH_ROOT=${PYTHONPATH_ROOT:-/data0/ysy/Adamas}
PYTHON_BIN=${PYTHON_BIN:-/home/ysy/anaconda3/envs/hsa/bin/python}
ADAMAS_BUDGETS=${ADAMAS_BUDGETS:-"256 512 1024 2048 4096"}
ROOT_DIR=${ROOT_DIR:-benchmark_root}

export METHOD=adamas
export PYTHONPATH=${PYTHONPATH_ROOT}
export PYTHON_BIN
export ROOT_DIR

cd /data0/ysy/Adamas/evaluation/RULER/scripts

for BUDGET in ${ADAMAS_BUDGETS}; do
    export ADAMAS_TOKEN_BUDGET=${BUDGET}
    export METHOD_TAG="adamas_${BUDGET}"

    echo "Launching RULER with METHOD=adamas, METHOD_TAG=${METHOD_TAG}, GPU=${GPU}, TOKEN_BUDGET=${BUDGET}, MODEL=${MODEL_NAME}, BENCHMARK=${BENCHMARK}"
    CUDA_VISIBLE_DEVICES="${GPU}" bash run.sh "${MODEL_NAME}" "${BENCHMARK}"
done
