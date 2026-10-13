MODEL=Llama-3.1-8B-Instruct

BUDGET_POOL=('256' '512' '1024' '2048' '4096' '133120')
CONTEXT_POOL=('102400' '131072')

MINFERENCE_PATH=${MINFERENCE_PATH:-/data0/ysy/MInference}
MINFERENCE_CONFIG=${MINFERENCE_CONFIG:-${MINFERENCE_PATH}/minference/configs/Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern_v2.json}
OUTPUT_PATH="test_results/e2e-minference/${MODEL}"
mkdir -p "$OUTPUT_PATH"

for budget in "${BUDGET_POOL[@]}"
do
    for context in "${CONTEXT_POOL[@]}"
    do
        python3 scripts/bench_textgen_minference.py \
            --model "$MODEL" \
            --context_len "$context" \
            --decode_len 256 \
            --token_budget "$budget" \
            --page_size 1 \
            --minference_path "$MINFERENCE_PATH" \
            --minference_config "$MINFERENCE_CONFIG" \
            > "${OUTPUT_PATH}/log_${budget}_${context}.log" 2>&1
    done
done
