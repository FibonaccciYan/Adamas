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
for len in "${avg_length[@]}"; do
  for budget in "${token_budget[@]}"; do
    len_divided=$((len / $page_size))
    budget_divided=$((budget / $page_size))
    ./bench_decode_select_k -a seq_len=$len_divided -a k=$budget_divided
  done
done

Profile estimate kernel
echo "|Profile estimate kernel|"
./bench_max_possible -a seqlen=[8192,16384,32768] -a page_size=$page_size


echo "|Profile full_attn kernel|"
page_sizes=(1 8 16 32)
for page_size in ${page_sizes[@]}; do
  ./bench_batch_decode -a seqlen=[8192,16384,32768] -a page_size=$page_size
done