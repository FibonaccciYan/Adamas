cd scripts

BUDGET_POOL=('64' '128' '256' '512' '1024' '2048' '4096' '102400') # 102400 is full cache version
CONTEXT_POOL=('8192' '16384' '32768')

for budget in "${BUDGET_POOL[@]}"
do
    for context in "${CONTEXT_POOL[@]}"
    do
        CUDA_VISIBLE_DEVICES="1" python3 profile_textgen.py --context_len $context --decode_len 16 --token_budget $budget --iteration 5 --page_size 1 > "../test_results/profile_log_${budget}_${context}.log" 2>&1
    done
done