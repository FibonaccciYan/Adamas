import pytest
from itertools import product

import torch
import torch.nn as nn
import math

import Adamas.utils

import faster_hadamard_transform

def pack_2bit(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Packs a tensor of shape `[num_heads, seq_len, head_dim]` into a 2-bit packed tensor.
    The output shape is `[num_heads, seq_len, head_dim // 8]`.
    """
    *dims, seq_len, hd = x.shape
    assert hd % 8 == 0, "The last dimension(hd) must be divisible by 8 for 2-bit packing."
    x = x.view(*dims, seq_len, -1, 8)

    x = x.to(torch.int16)
    shifts = torch.tensor([14, 12, 10, 8, 6, 4, 2, 0], dtype=torch.int16, device=x.device)
    packed_int16 = torch.zeros((*dims, seq_len, x.shape[-2]), dtype=torch.int16, device=x.device)
    for i in range(8):
        shifted = torch.bitwise_left_shift(x[..., i], shifts[i])
        packed_int16 = torch.bitwise_or(packed_int16, shifted)

    return packed_int16.view(dtype)

def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

def _ref_self_attention(
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

    attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)

    attn_mask = torch.ones_like(attn_weights, dtype=torch.bool)
    attn_mask = attn_mask.tril(diagonal=kv_len-qo_len)

    attn_weights[~attn_mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    return torch.matmul(attn_weights, v).transpose(0, 1)

parameters = list(product(["float16"], [1], [27, 61, 113, 482, 577, 1011]))
@pytest.mark.parametrize("dtype_str, qo_len, kv_len", parameters)
@torch.inference_mode()
def test_decode_attention_correctness(dtype_str, qo_len, kv_len):
    if qo_len != 1:
        pytest.skip("qo_len should be 1 for decode.")

    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    num_heads = 32
    head_dim = 128
    num_layers = 32
    page_size = 1
    page_budget = 1024 # Here we do not test approx attention, which is tested in test_approx_attention.py.
    max_seq_len = 2048

    if kv_len <= page_size:
        pytest.skip("At least one page")
        
    # layout: NHD
    q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
    # Simulate Prefill
    k_prefill = torch.randn(kv_len-1, num_heads, head_dim, dtype=dtype, device=device)
    v_prefill = torch.randn(kv_len-1, num_heads, head_dim, dtype=dtype, device=device)

    testController = Adamas.utils.InferenceController(
        num_layers,
        num_heads,
        head_dim,
        page_size,
        page_budget,
        max_seq_len,
        dtype,
        device,
    )

    k_prefill_code = torch.bucketize(faster_hadamard_transform.hadamard_transform(k_prefill, inplace=False), testController.thresholds, out_int32=True).to(device)
    k_prefill_code_2bit = pack_2bit(k_prefill_code, k_prefill_code.dtype).to(device)

    # Begin: Fill in prefill kv-data
    testController.prepare_hadamard(kv_len-1)
    testController.begin_forward(kv_len-1)
    # Construct KV
    Adamas.utils.append_kvh(k_prefill, v_prefill, k_prefill_code_2bit, testController, 0)
    testController.end_forward()

    k_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    v_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)

    k_decode_code = torch.bucketize(faster_hadamard_transform.hadamard_transform(k_decode, inplace=False),   testController.thresholds, out_int32=True).to(device)
    k_decode_code_2bit = pack_2bit(k_decode_code, k_decode_code.dtype).to(device)
    # Real decoding starts
    testController.prepare_hadamard(1)
    testController.begin_forward(1)
    Adamas.utils.append_kvh(k_decode, v_decode, k_decode_code_2bit, testController, 0)
    # No CPU test cases
    assert testController.need_estimate() == False
    o_device = Adamas.utils.decode_sparse_attn(
        q,
        testController,
        0,
        testController.kv_indices_without_last,
    )
    testController.end_forward()

    # stack k,v and get o
    k = torch.cat([k_prefill, k_decode], dim=0)
    v = torch.cat([v_prefill, v_decode], dim=0)
    o_host = _ref_self_attention(q, k, v)

    assert_close(o_device, o_host)