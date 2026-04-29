#!/bin/bash
set -euo pipefail

GPU=${GPU:-4}
MODEL_NAME=${MODEL_NAME:-llama3.1-8b-instruct}
BENCHMARK=${BENCHMARK:-synthetic}
PYTHONPATH_ROOT=${PYTHONPATH_ROOT:-/data0/ysy/Adamas}
PYTHON_BIN=${PYTHON_BIN:-/home/ysy/anaconda3/envs/hsa/bin/python}
ADAMAS_BUCKET_THRESHOLD=${ADAMAS_BUCKET_THRESHOLD:-10}
ADAMAS_BUDGETS=${ADAMAS_BUDGETS:-"64"}
ROOT_DIR_PREFIX=${ROOT_DIR_PREFIX:-benchmark_root_adamas_budget}

export METHOD=adamas
export PYTHONPATH=${PYTHONPATH_ROOT}
export PYTHON_BIN
export ADAMAS_BUCKET_THRESHOLD

cd /data0/ysy/Adamas/evaluation/RULER/scripts

for BUDGET in ${ADAMAS_BUDGETS}; do
    export ADAMAS_TOKEN_BUDGET=${BUDGET}
    export ROOT_DIR=${ROOT_DIR_PREFIX}_${BUDGET}

    echo "Launching RULER with METHOD=adamas, GPU=${GPU}, TOKEN_BUDGET=${BUDGET}, MODEL=${MODEL_NAME}, BENCHMARK=${BENCHMARK}"
    CUDA_VISIBLE_DEVICES="${GPU}" bash run.sh "${MODEL_NAME}" "${BENCHMARK}"
done
