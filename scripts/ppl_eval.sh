cd evaluation/pg19

MODELPATH=/data0/ysy/models/lmsys/longchat-7b-v1.5-32k
OUTPUT_DIR=results/longchat-7b-v1.5-32k
mkdir -p $OUTPUT_DIR

device=0
# budget=256

# for budget in 256 512 1024 
# do 
#     CUDA_VISIBLE_DEVICES=1 python -u ppl_eval.py \
#         --model_name_or_path $MODELPATH \
#         --output_dir $OUTPUT_DIR \
#         --num_eval_tokens 32000 \
#         --HSA --token_budget $budget --chunk_size 1 
# done

CUDA_VISIBLE_DEVICES=1 python -u ppl_eval.py \
    --model_name_or_path $MODELPATH \
    --output_dir $OUTPUT_DIR \
    --num_eval_tokens 32000 \
    --chunk_size 1 