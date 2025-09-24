import pytest
from itertools import product
import torch
import torch.nn as nn

import Adamas.utils

def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, equal_nan=True)

def _ref_pack_2bit(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
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

parameters = list(product(["float16"], [13, 24, 51, 77, 244, 311, 502, 1110], [2, 5, 7, 19, 31, 69, 111, 251]))
@pytest.mark.parametrize("dtype_str, past_kv_len, seq_len", parameters)
@torch.inference_mode()
def test_pack_2bit(dtype_str, past_kv_len, seq_len):
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    num_heads = 32
    head_dim = 128
    
    threshlods = torch.tensor([-10, 0, 10], device=device)
    # NHD: [seq_len, num_heads, head_dim]
    h = torch.bucketize(torch.randn(seq_len, 2 * num_heads, head_dim, dtype=dtype, device=device), threshlods).to(dtype)

    h_ref = _ref_pack_2bit(h, dtype)
    h_gpu = Adamas.utils.pack_2bit(h, past_kv_len, dtype)

    assert_close(h_gpu, h_ref)