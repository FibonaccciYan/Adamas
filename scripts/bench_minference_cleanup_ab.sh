set -euo pipefail

MODEL=Llama-3.1-8B-Instruct
CONTEXT_LEN=${CONTEXT_LEN:-102400}
TOKEN_BUDGET=${TOKEN_BUDGET:-256}
DECODE_LEN=${DECODE_LEN:-256}
ITERATION=${ITERATION:-3}
MINFERENCE_PATH=${MINFERENCE_PATH:-/data0/ysy/MInference}
MINFERENCE_CONFIG=${MINFERENCE_CONFIG:-${MINFERENCE_PATH}/minference/configs/Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern_v2.json}
OUTPUT_PATH="test_results/e2e-minference/${MODEL}"
mkdir -p "$OUTPUT_PATH"

for variant in no_cleanup cleanup
do
    cleanup_arg=()
    if [ "$variant" = cleanup ]; then
        cleanup_arg=(--empty_cache_after_prefill)
    fi
    python3 scripts/bench_textgen_minference.py \
        --model "$MODEL" \
        --context_len "$CONTEXT_LEN" \
        --decode_len "$DECODE_LEN" \
        --token_budget "$TOKEN_BUDGET" \
        --page_size 1 \
        --iteration "$ITERATION" \
        --minference_path "$MINFERENCE_PATH" \
        --minference_config "$MINFERENCE_CONFIG" \
        "${cleanup_arg[@]}" \
        > "${OUTPUT_PATH}/log_cleanup_ab_${variant}_${TOKEN_BUDGET}_${CONTEXT_LEN}.log" 2>&1
done
