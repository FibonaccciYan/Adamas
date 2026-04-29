cd evaluation/LongBench

model="Qwen3-8b"

for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
do
    python -u pred.py \
        --model $model --task $task

    for budget in 64 128 256 512 1024 2048 4096
    do
        python -u pred.py \
            --model $model --task $task \
            --Adamas --token_budget $budget --chunk_size 1 \
            # --thinking
    done
done

python -u eval.py --model $model