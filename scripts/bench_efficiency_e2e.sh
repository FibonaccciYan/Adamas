cd scripts

BUDGET_POOL=('64' '128' '256' '512' '1024' '2048' '4096' '102400') # 102400 is full cache version
CONTEXT_POOL=('8192' '16384' '32768')

# BUDGET_POOL=('128')
# CONTEXT_POOL=('32768')

# for page_size in 1 8 16 32
for page_size in 1
do
    for budget in "${BUDGET_POOL[@]}"
    do
        for context in "${CONTEXT_POOL[@]}"
        do
            # CUDA_LAUNCH_BLOCKING=1 compute-sanitizer --tool memcheck --leak-check full 
            python3 bench_textgen.py --context_len $context --decode_len 256 --token_budget $budget --iteration 1 --page_size $page_size > "../test_results/log_${budget}_${context}_164.log" 2>&1
        done
    done
done