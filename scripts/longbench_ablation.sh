#!/usr/bin/env bash
set -euo pipefail

cd evaluation/LongBench

model="${MODEL:-Meta-Llama-3.1-8B-Instruct}"
tasks="${TASKS:-qasper narrativeqa hotpotqa multifieldqa_en}"
budgets="${BUDGETS:-64 128 256 512 1024 2048 4096}"
variants="${VARIANTS:-adamas no_hadamard l2_distance thresholds_5sigma}"

if [ "$#" -gt 0 ]; then
    variants="$*"
fi

if [ "${RUN_FULL:-0}" = "1" ]; then
    for task in $tasks; do
        python -u pred.py --model "$model" --task "$task"
    done
fi

for variant in $variants; do
    for task in $tasks; do
        for budget in $budgets; do
            python -u pred.py \
                --model "$model" \
                --task "$task" \
                --Adamas \
                --adamas_variant "$variant" \
                --token_budget "$budget" \
                --chunk_size 1
        done
    done
done

python -u eval.py --model "$model"
