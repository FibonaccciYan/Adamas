MODEL=Llama-3.1-8B-Instruct

BUDGET_POOL=('256' '512' '1024' '2048' '4096' '102400') # 102400 is full cache version
CONTEXT_POOL=('8192' '16384' '24576' '32768' '40960' '49152' '57344' '65565')

OUTPUT_PATH="test_results/${MODEL}"
mkdir -p $OUTPUT_PATH

for budget in "${BUDGET_POOL[@]}"
do
    for context in "${CONTEXT_POOL[@]}"
    do
        python3 scripts/bench_textgen.py --model $MODEL --context_len $context --decode_len 256 --token_budget $budget --page_size 1 > "${OUTPUT_PATH}/log_${budget}_${context}.log" 2>&1
    done
done