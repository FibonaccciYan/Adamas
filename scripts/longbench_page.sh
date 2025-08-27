cd evaluation/LongBench

model="longchat-v1.5-7b-32k"
out_path="pred/${model}_lsh_paged_3bit"

# for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
# do
#     python -u pred.py \
#         --model $model --task $task

#     for budget in 512 1024 2048 4096
#     do
#         python -u pred.py \
#             --model $model --task $task \
#             --HSA --token_budget $budget --chunk_size 16
#     done
# done

for task in triviaqa
do
    # python -u pred.py \
    #     --model $model --task $task --path $out_path
        
    for budget in 512 1024
    do
        python -u pred.py \
            --model $model --task $task \
            --attn lsh_paged_3bit --token_budget $budget --chunk_size 16 --path $out_path
    done
done

python -u eval.py --model $model --path $out_path