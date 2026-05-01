MODEL=Llama-3.1-8B-Instruct

# BUDGET_POOL=('64' '128' '256' '512' '1024' '2048' '4096' '102400') # 102400 is full cache version
BUDGET_POOL=('512' '1024' '2048' '4096' '102400') # 102400 is full cache version
CONTEXT_POOL=('8192' '16384' '32768')
# CONTEXT_POOL=('40960' '49152' '57344')

for budget in "${BUDGET_POOL[@]}"
do
    for context in "${CONTEXT_POOL[@]}"
    do
        python3 scripts/bench_textgen.py --model $MODEL --context_len $context --decode_len 256 --token_budget $budget --iteration 1 --page_size 1 > "test_results/${MODEL}/log_${budget}_${context}.log" 2>&1
    done
done