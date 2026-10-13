# This file is modified from Punica Project
# Check ref: https://github.com/punica-ai/punica

from adamas.utils.utils import TensorLayout
import torch

class KvPool:

  def __init__(
      self,
      num_layers: int,
      num_kv_heads: int,
      head_dim: int,
      capacity: int,
      items: int,
      block_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ):
    self._layout = TensorLayout.NHD
    self._buf = torch.empty(
        (num_layers, capacity, items, block_len, num_kv_heads, head_dim),
        dtype=dtype,
        device=device)
    
    # This cache is used by a single active sequence. Allocate page ids
    # monotonically so Python does not need to manage a large free set during
    # prefill.
    self._next_block = 0

  @property
  def layout(self):
    return self._layout

  @property
  def buf(self):
    return self._buf

  @property
  def num_layers(self):
    l, c, _, p, n, d = self._buf.shape
    return l

  @property
  def block_len(self):
    l, c, _, p, n, d = self._buf.shape
    return p

  @property
  def num_free_blocks(self):
    return self.capacity - self._next_block

  @property
  def capacity(self):
    l, c, _, p, n, d = self._buf.shape
    return c

  def alloc_block(self) -> int:
    if self._next_block >= self.capacity:
      raise RuntimeError("KV cache capacity exceeded")
    idx = self._next_block
    self._next_block += 1
    return idx

  def alloc_blocks(self, num_blocks: int) -> list[int]:
    if num_blocks <= 0:
      return []
    if self._next_block + num_blocks > self.capacity:
      raise RuntimeError("KV cache capacity exceeded")
    start = self._next_block
    self._next_block += num_blocks
    return list(range(start, start + num_blocks))

  def free_block(self, idx: int):
    assert 0 <= idx < self.capacity

  def reset(self):
    self._next_block = 0


class KvCache:
  """Key-value cache for one sequence."""

  def __init__(
      self,
      num_layers,
      num_kv_heads,
      head_dim,
      max_seq_len: int,
      page_size,
      items,
      dtype: torch.dtype,
      device: torch.device
    ):
    
    if max_seq_len <= 0:
      raise ValueError("init_len must be non-negative")

    self._pool = KvPool(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        capacity=(max_seq_len + page_size - 1) // page_size,
        items=items,
        block_len=page_size,
        dtype=dtype,
        device=device
    )
  
    self._indicies = []
    self._seqlen = 0

  @property
  def pool(self) -> KvPool:
    return self._pool

  @property
  def seqlen(self) -> int:
    return self._seqlen

  @property
  def last_page_len(self) -> int:
    return (self.seqlen - 1) % self._pool.block_len + 1

  @property
  def indicies(self) -> list[int]:
    return self._indicies
  
  def buf_layer(self, layer_idx: int):
    assert layer_idx < self.pool.num_layers
    return self._pool.buf[layer_idx]

  def append_seq(self, seq_len: int) -> int:
    """Reserve space for tokens and return number of new pages"""
    if seq_len <= 0:
        return 0
    old_num_pages = (self._seqlen + self._pool.block_len - 1) // self._pool.block_len
    new_seqlen = self._seqlen + seq_len
    new_num_pages = (new_seqlen + self._pool.block_len - 1) // self._pool.block_len
    appended_page_count = new_num_pages - old_num_pages
    self._indicies.extend(self._pool.alloc_blocks(appended_page_count))
    self._seqlen = new_seqlen
    return appended_page_count

  def release(self):
    """Release all blocks"""
    self._seqlen = 0
    self._indicies.clear()
    self._pool.reset()
