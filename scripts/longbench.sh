cd evaluation/LongBench

model="longchat-v1.5-7b-32k"
# model="Meta-Llama-3.1-8B-Instruct"
# model="Qwen3-8b"

for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "triviaqa" "gov_report"
do
    python -u pred.py \
        --model $model --task $task

    for budget in 256 512 1024 2048 4096
    do
        python -u pred.py \
            --model $model --task $task \
            --Adamas --token_budget $budget --chunk_size 1 \
            # --thinking
    done
done

python -u eval.py --model $model