import pytest
from itertools import product

import torch
import torch.nn as nn
import math

import adamas.utils

import faster_hadamard_transform

def pack_2bit(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    *dims, hd = x.shape
    assert hd % 8 == 0, "The last dimension(hd) must be divisible by 8 for 2-bit packing."
    x = x.view(*dims, -1, 8)

    x = x.to(torch.int16)
    shifts = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.int16, device=x.device)
    packed_int16 = torch.zeros((*dims, x.shape[-2]), dtype=torch.int16, device=x.device)
    for i in range(8):
        shifted = torch.bitwise_left_shift(x[..., i], shifts[i])
        packed_int16 = torch.bitwise_or(packed_int16, shifted)

    return packed_int16.view(dtype)

def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, equal_nan=True)

def _ref_cpu_estimate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L413
    # Assume all input layout: NHD 
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    assert k.size(0) == v.size(0)

    head_dim = q.size(2)
    qo_len = q.size(0)
    kv_len = k.size(0)
    
    assert kv_len >= qo_len

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    thresholds = torch.tensor([-10, 0, 10]).to(q.device)
    q_code = torch.bucketize(faster_hadamard_transform.hadamard_transform(q, inplace=False), thresholds, out_int32=True)
    k_code = torch.bucketize(faster_hadamard_transform.hadamard_transform(k, inplace=False), thresholds, out_int32=True)

    approx_attn = nn.functional.pairwise_distance(
        q_code,
        k_code, 
        p=1,
    )[:, :kv_len - 1] # [num_heads, kv_len-1]
    approx_attn = approx_attn.to(q.dtype)
    
    return approx_attn.contiguous()

parameters = list(product(["float16"], [27, 61, 113, 482, 577, 1110, 1541, 2047, 3330]))
@pytest.mark.parametrize("dtype_str, kv_len", parameters)
@torch.inference_mode()
def test_estimate_correctness(dtype_str, kv_len):
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    qo_len = 1

    num_heads = 32
    head_dim = 128
    num_layers = 32
    page_size = 1
    page_budget = 1024 # Not used here. Random initialize
    max_seq_len = 8192

    if kv_len <= page_size:
        pytest.skip("At least one page")
    if qo_len > kv_len:
        pytest.skip("qo_len should be less than kv_len")
        
    # layout: NHD
    q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)

    # Simulate Prefill
    # Doing this since we need begin_forward to prepare metadata
    k_prefill = torch.randn(kv_len-1, num_heads, head_dim, dtype=dtype, device=device)
    v_prefill = torch.randn(kv_len-1, num_heads, head_dim, dtype=dtype, device=device)

    testController = adamas.utils.InferenceController(
        num_layers,
        num_heads,
        head_dim,
        page_size,
        page_budget,
        max_seq_len,
        dtype,
        device,
    )

    h_prefill = faster_hadamard_transform.hadamard_transform(k_prefill, inplace=False)

    # Begin: Fill in prefill kv-data
    testController.prepare_hadamard(kv_len-1)
    testController.begin_forward(kv_len-1)
    # Construct KV
    adamas.utils.append_kvh(k_prefill, v_prefill, h_prefill, testController, 0)
    testController.end_forward()

    k_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    v_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)

    h_decode = torch.cat((q, k_decode), dim=0)
    h_decode = faster_hadamard_transform.hadamard_transform(h_decode, inplace=False)
    # CUDA Evaluation
    testController.prepare_hadamard(qo_len)
    testController.begin_forward(qo_len)
    q_code_2bit = adamas.utils.append_kvh(k_decode, v_decode, h_decode, testController, 0)

    thresholds = torch.tensor([-10, 0, 10], device=device)
    q_code_2bit_ref = pack_2bit(
        torch.bucketize(
            faster_hadamard_transform.hadamard_transform(q, inplace=False), 
            thresholds, 
            out_int32=True
        ), 
        dtype
    )

    assert_close(q_code_2bit, q_code_2bit_ref)

    cuda_estimated_value = adamas.utils.decode_estimate(
        q_code_2bit,
        testController,
        0,
    )
    testController.end_forward()

    # CPU Evaluation
    k = torch.cat([k_prefill, k_decode], dim=0)
    v = torch.cat([v_prefill, v_decode], dim=0)
    host_estimated_value = _ref_cpu_estimate(q, k, v)

    assert_close(cuda_estimated_value, host_estimated_value)