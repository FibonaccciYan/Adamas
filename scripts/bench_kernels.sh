# This file used to profile the efficiency breakdown of kernels.

cd kernels/build

echo "|Profile approx_attn kernel|"
./bench_batch_decode -a seqlen=[24576,32768,40960,49152,57344,65536] -a page_budget=[256,512,1024,2048,4096] -a page_size=1

echo "|Profile fused_estimate_topk kernel|"
./bench_fused_estimate_topk -a seqlen=[24576,32768,40960,49152,57344,65536]

echo "|Profile full_attn kernel|"
./bench_batch_decode -a seqlen=[24576,32768,40960,49152,57344,65536] -a page_budget=102400 -a page_size=1