# This file used to profile the efficiency breakdown of kernels.

cd ../kernels/build
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0

avg_length=(8192 16384 32768)
token_budget=(256 512 1024 2048 4096)
page_size=1

# Profile approx_attn kernel
echo "|Profile approx_attn kernel|"
./bench_batch_decode -a seqlen=[8192,16384,32768] -a page_budget=[256,512,1024,2048,4096] -a page_size=$page_size

# Profile topk kernel
echo "|Profile topk kernel|"
./bench_decode_select_k -a seq_len=[8192,16384,32768] -a k=[256,512,1024,2048,4096] 

# Profile estimate kernel
echo "|Profile estimate kernel|"
./bench_max_possible -a seqlen=[8192,16384,32768] -a page_size=$page_size

# Profile full_attn kernel
echo "|Profile full_attn kernel|"
./bench_batch_decode -a seqlen=[8192,16384,32768] -a page_budget=102400 -a page_size=$page_size