import math
from typing import Optional, Tuple

import torch
from torch import nn

import types

from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)

import faster_hadamard_transform


def _sigma_thresholds(states: torch.Tensor, sigma_times: float) -> torch.Tensor:
    sigma = states.float().std()
    scale = torch.as_tensor(sigma_times, device=states.device, dtype=sigma.dtype)
    zero = torch.zeros((), device=states.device, dtype=sigma.dtype)
    return torch.stack((-scale * sigma, zero, scale * sigma)).to(states.dtype)


def adamas_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if isinstance(past_key_value, DynamicCache):
        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)
    else:
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

    query_hadamard = faster_hadamard_transform.hadamard_transform(query_states, inplace=False)
    key_hadamard = faster_hadamard_transform.hadamard_transform(key_states, inplace=False)
    thresholds_q = _sigma_thresholds(query_hadamard, self.sigma_times)
    thresholds_k = _sigma_thresholds(key_hadamard, self.sigma_times)
    query_code = torch.bucketize(query_hadamard, thresholds_q, out_int32=True)
    key_code = torch.bucketize(key_hadamard, thresholds_k, out_int32=True)

    token_budget = min(self.token_budget, key_code.shape[-2])
    if self.num_key_value_groups > 1:
        query_code_grouped = query_code.view(
            bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, self.head_dim
        )
        distances = (query_code_grouped[:, :, :, :, None, :] - key_code[:, :, None, None, :, :]).abs().sum(dim=-1)
        group_distances = distances.min(dim=2).values
        _, group_topk_indices = group_distances.topk(k=token_budget, dim=-1, largest=False)
        topk_indices = group_topk_indices.repeat_interleave(self.num_key_value_groups, dim=1)
    else:
        distances = (query_code[:, :, :, None, :] - key_code[:, :, None, :, :]).abs().sum(dim=-1)
        _, topk_indices = distances.topk(k=token_budget, dim=-1, largest=False)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk_indices, True)
    attn_weights = attn_weights.masked_fill(~mask_bottom, torch.tensor(torch.finfo(attn_weights.dtype).min))

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_code.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


layer_id = 32


def enable_adamas_attention_eval(model, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_adamas_attention_eval(module, args)

        global layer_id
        if name == "self_attn":
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(adamas_forward, model._modules[name])
            model._modules[name].token_budget = args.token_budget
            model._modules[name].sigma_times = args.sigma_times
