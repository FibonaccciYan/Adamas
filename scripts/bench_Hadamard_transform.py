import torch

from flash_attn.utils.benchmark import benchmark_forward, pytorch_profiler

import faster_hadamard_transform 

batch_size = 1
nh = 32
seqlens = [2, 8192, 16384, 32768]
dim = 128
dtype = torch.float16
device = "cuda"

torch.random.manual_seed(0)

for seqlen in seqlens:
    print(f"seqlen = {seqlen}")
    x = torch.randn(batch_size, nh, seqlen, dim, dtype=dtype, device=device)
    benchmark_forward(faster_hadamard_transform.hadamard_transform, x, inplace=True, desc="Hadamard transform")
    print("**************************************")
    pytorch_profiler(faster_hadamard_transform.hadamard_transform, x, inplace=True)