import torch

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


def unpack_2bit(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpacks a 2-bit packed tensor to its original shape.
    Input shape: `[num_heads, seq_len, head_dim // 8]`
    Output shape: `[num_heads, seq_len, head_dim]`
    """
    # 将 float16 重新解释为 uint16 并转换为 int16
    packed_int16 = packed.view(torch.uint16).to(torch.int16)
    
    # 创建掩码
    mask = 0x03  # 二进制 00000011
    
    # 计算解包后的维度
    *dims, seq_len, packed_dim = packed_int16.shape
    unpacked = torch.zeros((*dims, seq_len, packed_dim * 8), dtype=torch.uint8, device=packed.device)
    
    # 定义移位值
    shifts = torch.tensor([14, 12, 10, 8, 6, 4, 2, 0], dtype=torch.int16, device=packed.device)
    
    # 解包每个位
    for i in range(8):
        # 右移并应用掩码
        unpacked[..., i::8] = torch.bitwise_and(
            torch.bitwise_right_shift(packed_int16, shifts[i]), 
            mask
        ).to(torch.uint8)
    
    return unpacked


a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                  [9, 10, 11, 12, 13, 14, 15, 16]], dtype=torch.float16)

a = a % 4

packed = pack_2bit(a, torch.float16)
unpacked = unpack_2bit(packed)

print("Original tensor:")
print(a)
print("Packed tensor:")
print(packed)
print("Unpacked tensor:")
print(unpacked)

import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, ProfilerActivity

x = torch.randn(2048, 2048, device="cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    use_cuda=True,
) as prof:
    nvtx.range_push("test_nvtx")
    y = torch.matmul(x, x)
    nvtx.range_pop()
    prof.step()

prof.export_chrome_trace("nvtx_test.json")
