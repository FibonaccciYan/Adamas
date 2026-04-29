cd evaluation/pg19

# MODELPATH=/data1/model/llama3/meta-llama/Llama-3.1-8B-Instruct
# OUTPUT_DIR=results/Llama-3.1-8B-Instruct
MODELPATH=/data1/model/qwen/Qwen/Qwen3-8B
OUTPUT_DIR=results/Qwen3-8B
mkdir -p $OUTPUT_DIR

for budget in 256 512 1024 2048
do 
    python -u ppl_eval.py \
        --model_name_or_path $MODELPATH \
        --output_dir $OUTPUT_DIR \
        --num_eval_tokens 32000 \
        --Adamas --token_budget $budget --chunk_size 1 
done

python -u ppl_eval.py \
    --model_name_or_path $MODELPATH \
    --output_dir $OUTPUT_DIR \
    --num_eval_tokens 32000 \
    --chunk_size 1 
