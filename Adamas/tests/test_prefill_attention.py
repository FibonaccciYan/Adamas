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

parameters = list(product(["float16"], [13, 24, 51, 77, 244, 311, 502], [33, 66, 129, 400, 700, 1110]))
@pytest.mark.parametrize("dtype_str, qo_len, kv_len", parameters)
@torch.inference_mode()
def test_prefill_attention_correctness(dtype_str, qo_len, kv_len):
    if qo_len > kv_len:
        pytest.skip("qo_len > kv_len is not supported")

    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    num_heads = 32
    head_dim = 128
    num_layers = 32
    page_size = 1
    page_budget = 1024
    max_seq_len = 2048

    # layout: NHD
    q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device=device)

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

    k_code = torch.bucketize(faster_hadamard_transform.hadamard_transform(k),   testController.thresholds, out_int32=True)
    k_code_2bit = pack_2bit(k_code, k_code.dtype)

    # Begin (prepare kv-cache hadamard data)
    testController.prepare_hadamard(kv_len)
    testController.begin_forward(kv_len)
    # Construct KV with maintained hadamard data
    Adamas.utils.append_kvh(k, v, k_code_2bit, testController, 0)
    o_device = Adamas.utils.prefill_forward(q, testController, 0)
    o_host = _ref_self_attention(q, k, v)

    assert_close(o_device, o_host)