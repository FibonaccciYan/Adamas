import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv, logger
from transformers.cache_utils import Cache

import Adamas.utils

import faster_hadamard_transform

import pdb


class Adamas(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()


    def _init_rope(self):
        # rope_theta is default to 1e4, as set in RoPE kernel API.
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
            self.rope_scale = 1.0
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "linear":
                # support for Longchat-v1.5.
                self.rope_scale = self.config.rope_scaling["factor"]
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        iController: Optional[Adamas.utils.InferenceController] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        assert bsz == 1, "Adamas only supports batch size 1."
        assert hasattr(self, 'layer_idx'), "Adamas requires layer_idx to inference."

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)

        torch.cuda.nvtx.range_push("RoPE")
        Adamas.utils.apply_rope_in_place(query_states, key_states, iController.kv_cache.seqlen - q_len, rope_scale=self.rope_scale)
        torch.cuda.nvtx.range_pop()
    

        # Prefill/Decode kernels is different
        if q_len > 1:
            torch.cuda.nvtx.range_push("hadamard_transform")
            hadamard_states = faster_hadamard_transform.hadamard_transform(key_states, inplace=False)
            torch.cuda.nvtx.range_pop()

            # Adamas manages KV-Cache internal (with PageAttention)
            # Here we do not concat / stack
            # We concat after RoPE
            torch.cuda.nvtx.range_push("append_kvh")
            Adamas.utils.append_kvh(
                key_states,
                value_states,
                hadamard_states,
                iController,
                self.layer_idx,
            )
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("prefill_attn")
            attn_output = Adamas.utils.prefill_forward(
                query_states,
                iController,
                self.layer_idx,
            )
            torch.cuda.nvtx.range_pop()
        else:
            torch.cuda.nvtx.range_push("hadamard_transform")
            hadamard_states = torch.cat(
                (query_states, key_states), 
                dim=0,
            )

            faster_hadamard_transform.hadamard_transform(hadamard_states, inplace=True)
            torch.cuda.nvtx.range_pop()

            # We concat after RoPE
            torch.cuda.nvtx.range_push("append_kvh")
            query_code_2bit = Adamas.utils.append_kvh(
                key_states,
                value_states,
                hadamard_states,
                iController,
                self.layer_idx,
            )
            torch.cuda.nvtx.range_pop()

            # Skipping layers is controled by PAGE_BUDGET, which is set in LlamaModel.
            if iController.need_estimate() == False:
                torch.cuda.nvtx.range_push("full_attn")
                attn_output = Adamas.utils.decode_sparse_attn(
                    query_states,
                    iController,
                    self.layer_idx,
                    iController.kv_indices_without_last, 
                )
                torch.cuda.nvtx.range_pop()
            else:
                torch.cuda.nvtx.range_push("estimate")
                estimated_attn_score = Adamas.utils.decode_estimate(
                    query_code_2bit,
                    iController,
                    self.layer_idx,
                )
                torch.cuda.nvtx.range_pop()

                # select top-k smallest indices
                torch.cuda.nvtx.range_push("topk")
                Adamas.utils.decode_topk(
                    estimated_attn_score,
                    iController,
                )
                torch.cuda.nvtx.range_pop()

                # if self.layer_idx == 2:
                #     print(f"topk: {iController.topk_dindices_buffer[0]}")

                torch.cuda.nvtx.range_push("approx_attn")
                attn_output = Adamas.utils.decode_sparse_attn(
                    query_states,
                    iController,
                    self.layer_idx,
                    iController.topk_dindices_buffer,
                )
                torch.cuda.nvtx.range_pop()

        attn_output = attn_output.unsqueeze(0) # unsqueeze the batch dimension
        # FlashInfer output is naturally NHD
        # Note that we manully control NHD. Should be more general
        if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        torch.cuda.nvtx.range_push("o_proj")
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop()

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value