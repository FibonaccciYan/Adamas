cd evaluation/AIME

model="Qwen3-8b"

for budget in 1024 128
do
    python -u pred.py \
        --model $model --max_new_tokens 38912 \
        --Adamas --token_budget $budget --chunk_size 1 \
        --thinking
done