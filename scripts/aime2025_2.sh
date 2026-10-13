#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../evaluation/AIME"

model="Qwen3-8b"

for budget in 1024 4096
do
    python -u pred.py \
        --benchmark aime2025 \
        --model "$model" --max_new_tokens 38912 \
        --Adamas --token_budget "$budget" --chunk_size 1 \
        --thinking
done
