cd evaluation/pg19

MODELPATH=/path/to/lmsys/longchat-7b-v1.5-32k
OUTPUT_DIR=results/longchat-7b-v1.5-32k
mkdir -p $OUTPUT_DIR

for budget in 256 512 1024 
do 
    CUDA_VISIBLE_DEVICES=1 python -u ppl_eval.py \
        --model_name_or_path $MODELPATH \
        --output_dir $OUTPUT_DIR \
        --num_eval_tokens 32000 \
        --Adamas --token_budget $budget --chunk_size 1 
done

CUDA_VISIBLE_DEVICES=1 python -u ppl_eval.py \
    --model_name_or_path $MODELPATH \
    --output_dir $OUTPUT_DIR \
    --num_eval_tokens 32000 \
    --chunk_size 1 
