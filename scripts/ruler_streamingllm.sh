#!/bin/bash
set -euo pipefail

GPU=${GPU:-5}
MODEL_NAME=${MODEL_NAME:-llama3.1-8b-instruct}
BENCHMARK=${BENCHMARK:-synthetic}
PYTHONPATH_ROOT=${PYTHONPATH_ROOT:-/data0/ysy/Adamas}
PYTHON_BIN=${PYTHON_BIN:-/home/ysy/anaconda3/envs/hsa/bin/python}
STREAMINGLLM_SINK_SIZE=${STREAMINGLLM_SINK_SIZE:-4}
STREAMINGLLM_BUDGETS=${STREAMINGLLM_BUDGETS:-"64"}
ROOT_DIR_PREFIX=${ROOT_DIR_PREFIX:-benchmark_root_streamingllm_budget}

export METHOD=streamingllm
export PYTHONPATH=${PYTHONPATH_ROOT}
export PYTHON_BIN
export STREAMINGLLM_SINK_SIZE

cd /data0/ysy/Adamas/evaluation/RULER/scripts

for BUDGET in ${STREAMINGLLM_BUDGETS}; do
    export STREAMINGLLM_WINDOW_SIZE=${BUDGET}
    export ROOT_DIR=${ROOT_DIR_PREFIX}_${BUDGET}

    echo "Launching RULER with METHOD=streamingllm, GPU=${GPU}, WINDOW_SIZE=${BUDGET}, MODEL=${MODEL_NAME}, BENCHMARK=${BENCHMARK}"
    CUDA_VISIBLE_DEVICES="${GPU}" bash run.sh "${MODEL_NAME}" "${BENCHMARK}"
done
