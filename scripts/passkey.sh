cd evaluation/passkey

# MODEL=Llama-3.1-8B-Instruct
# MODELPATH=meta-llama/Llama-3.1-8B-Instruct
MODEL=longchat-7b-v1.5-32k
MODELPATH=/data0/ysy/models/lmsys/longchat-7b-v1.5-32k
OUTPUT_DIR=results/$MODEL

mkdir -p $OUTPUT_DIR

length=100000

for token_budget in 256
do
    python passkey.py -m $MODELPATH \
        --iterations 100 --fixed-length $length \
        --HSA --token_budget $token_budget --chunk_size 1 \
        --output-file $OUTPUT_DIR/$MODEL-HSA-$token_budget.jsonl
done
