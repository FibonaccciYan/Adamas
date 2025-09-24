import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

import types

import transformers.models
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    logger,
)

from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward

import faster_hadamard_transform

@torch.jit.script
def repeat_kv_yarn(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, _, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, :, None, :].expand(batch, slen, 2, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, 2, num_key_value_heads * n_rep, head_dim)

def flash_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

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

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def adamas_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
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

    # New cache format
    if isinstance(past_key_value, DynamicCache):
        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)
    # Legacy cache format
    else:
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

    # if past_key_value is not None:
    #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    # key_code = repeat_kv(key_code, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    thresholds = torch.tensor([-10, 0, 10], device=query_states.device)
    key_code   = torch.bucketize(faster_hadamard_transform.hadamard_transform(key_states, inplace=False), thresholds, out_int32=True)
    query_code = torch.bucketize(faster_hadamard_transform.hadamard_transform(query_states, inplace=False), thresholds, out_int32=True)
    
    distances = nn.functional.pairwise_distance(query_code, key_code, p=1).unsqueeze(2)  # [bsz, nh, q_len, kv_seq_len]

    token_budget = min(self.token_budget, key_code.shape[-2])
    _, topk_indices = distances.topk(k=token_budget, dim=-1, largest=False)

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk_indices, True)

    attn_weights = attn_weights.masked_fill(~mask_bottom, torch.tensor(torch.finfo(attn_weights.dtype).min))
    

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_code.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        value_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def yarn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # print("Yarn forward called")
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
    
    has_layer_past = past_key_value is not None

    if has_layer_past:
        past_kv = past_key_value[0]
        past_len = past_key_value[1]
    else:
        past_len = 0

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        # .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        # .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        # .transpose(1, 2)
    )
    
    # [bsz, nh, t, hd]
    query_states, key_states = self.rotary_emb(query_states, key_states, past_len)

    kv = torch.stack([key_states, value_states], 2)
    kv = repeat_kv_yarn(kv, self.num_key_value_groups)

    # Cache QKV values
    if has_layer_past:
        new_len = past_len+q_len
        if new_len > past_kv.size(1):
            past_kv = torch.cat([past_kv, torch.empty(bsz, 256, 2, kv.size(3), kv.size(4), dtype=kv.dtype, device=kv.device)], 1)
        past_kv[:, past_len:new_len] = kv
        kv = past_kv[:, :new_len]
    else:
        past_kv = kv

    past_key_value = (past_kv, past_len+q_len) if use_cache else None

    key_states = kv[:, :, 0].squeeze(2).contiguous()
    value_states = kv[:, :, 1].squeeze(2).contiguous()

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    thresholds = torch.tensor([-10, 0, 10], device=query_states.device)
    key_code   = torch.bucketize(faster_hadamard_transform.hadamard_transform(key_states, inplace=False), thresholds, out_int32=True)
    query_code = torch.bucketize(faster_hadamard_transform.hadamard_transform(query_states, inplace=False), thresholds, out_int32=True)
    
    distances = nn.functional.pairwise_distance(query_code, key_code, p=1).unsqueeze(2)  # [bsz, nh, q_len, kv_seq_len]

    token_budget = min(self.token_budget, key_code.shape[-2])
    _, topk_indices = distances.topk(k=token_budget, dim=-1, largest=False)

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk_indices, True)

    attn_weights = attn_weights.masked_fill(~mask_bottom, torch.tensor(torch.finfo(attn_weights.dtype).min))
    

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_code.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        value_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


global layer_id
layer_id = 32


def enable_adamas_dynamic_cache_for_llama():
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = flash_attention_forward


def enable_adamas_attention_eval(model, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_adamas_attention_eval(
                module,
                args,
            )

        global layer_id
        if name == "self_attn":
        # if isinstance(module, (LlamaAttention, MistralAttention)):
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                yarn_forward, model._modules[name]
            )

            model._modules[name].token_budget = args.token_budget