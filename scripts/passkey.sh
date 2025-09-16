cd evaluation/passkey

# MODEL=longchat-7b-v1.5-32k
# MODELPATH=/data0/ysy/models/lmsys/longchat-7b-v1.5-32k
MODEL=Yarn-Llama-2-7b-128k
MODELPATH=/data0/ysy/models/NousResearch/Yarn-Llama-2-7b-128k
OUTPUT_DIR=results/$MODEL

mkdir -p $OUTPUT_DIR

length=100000

for token_budget in 16 32 64 128 256 512 1024 2048 4096
do
    python passkey.py -m $MODELPATH \
        --iterations 100 --fixed-length $length \
        --HSA --token_budget $token_budget --chunk_size 1 \
        --output-file $OUTPUT_DIR/$MODEL-HSA-$token_budget.jsonl
done
