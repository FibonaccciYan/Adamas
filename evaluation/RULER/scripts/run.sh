#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <benchmark_name>"
    exit 1
fi

METHOD=${METHOD:-full}
case "$METHOD" in
    full|adamas) ;;
    *) echo "Unsupported METHOD: $METHOD; use adamas or full." >&2; exit 1 ;;
esac
METHOD_BUDGET=${METHOD_BUDGET:-}
if [ -z "${METHOD_BUDGET}" ]; then
    case "${METHOD}" in
        adamas)
            METHOD_BUDGET=${ADAMAS_TOKEN_BUDGET:-}
            ;;
    esac
fi
if [ -z "${METHOD_TAG:-}" ]; then
    if [ -n "${METHOD_BUDGET}" ]; then
        METHOD_TAG="${METHOD}_${METHOD_BUDGET}"
    else
        METHOD_TAG="${METHOD}"
    fi
fi
ROOT_DIR=${ROOT_DIR:-benchmark_root}
BATCH_SIZE=${BATCH_SIZE:-1}

source config_models.sh
MODEL_NAME=${1}
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_CONFIG=$(MODEL_SELECT "${MODEL_NAME}")
IFS=":" read -r MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

source config_tasks.sh
BENCHMARK=${2}
declare -n DEFAULT_TASKS=$BENCHMARK
if [ -z "${DEFAULT_TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

if [ -n "${RULER_NUM_SAMPLES:-}" ]; then
    NUM_SAMPLES=${RULER_NUM_SAMPLES}
fi

SEQ_LENGTH_LIST=("${SEQ_LENGTHS[@]}")
if [ -n "${RULER_SEQ_LENGTHS:-}" ]; then
    IFS=',' read -r -a SEQ_LENGTH_LIST <<< "${RULER_SEQ_LENGTHS}"
fi

TASK_LIST=("${DEFAULT_TASKS[@]}")
if [ -n "${RULER_TASKS:-}" ]; then
    IFS=',' read -r -a TASK_LIST <<< "${RULER_TASKS}"
fi

echo "METHOD=${METHOD}"
echo "METHOD_TAG=${METHOD_TAG}"
echo "METHOD_BUDGET=${METHOD_BUDGET}"
echo "MODEL=${MODEL_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BENCHMARK=${BENCHMARK}"
echo "SEQ_LENGTHS=${SEQ_LENGTH_LIST[*]}"
echo "TASKS=${TASK_LIST[*]}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"

total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTH_LIST[@]}"; do
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${METHOD_TAG}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p "${DATA_DIR}" "${PRED_DIR}"

    for TASK in "${TASK_LIST[@]}"; do
        export RULER_CURRENT_SEQ_LENGTH=${MAX_SEQ_LENGTH}

        "${PYTHON_BIN}" data/prepare.py \
            --save_dir "${DATA_DIR}" --benchmark "${BENCHMARK}" --task "${TASK}" \
            --tokenizer_path "${TOKENIZER_PATH}" --tokenizer_type "${TOKENIZER_TYPE}" \
            --max_seq_length "${MAX_SEQ_LENGTH}" --model_template_type "${MODEL_TEMPLATE_TYPE}" \
            --num_samples "${NUM_SAMPLES}" ${REMOVE_NEWLINE_TAB}

        start_time=$(date +%s)
        "${PYTHON_BIN}" pred/call_api.py \
            --data_dir "${DATA_DIR}" --save_dir "${PRED_DIR}" --benchmark "${BENCHMARK}" \
            --task "${TASK}" --server_type "${MODEL_FRAMEWORK}" \
            --model_name_or_path "${MODEL_PATH}" --temperature "${TEMPERATURE}" \
            --top_k "${TOP_K}" --top_p "${TOP_P}" --batch_size "${BATCH_SIZE}" ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done

    "${PYTHON_BIN}" eval/evaluate.py --data_dir "${PRED_DIR}" --benchmark "${BENCHMARK}"
done

echo "Total time spent on call_api: $total_time seconds"
