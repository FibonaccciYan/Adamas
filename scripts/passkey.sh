cd evaluation/passkey

MODEL=Qwen3-8b
MODELPATH=/data1/model/qwen/Qwen/Qwen3-8B
OUTPUT_DIR=results/$MODEL

mkdir -p $OUTPUT_DIR

length=100000

for token_budget in 16 32 64 128 256 512 1024 2048 4096
do
    python passkey.py -m $MODELPATH \
        --iterations 100 --fixed-length $length \
        --Adamas --token_budget $token_budget --chunk_size 1 \
        --output-file $OUTPUT_DIR/$MODEL-Adamas-$token_budget.jsonl \
        # --thinking
done
