"""Adamas attention dispatch for Transformers 4.51.0."""

import types

from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention


def enable_adamas_attention_eval(model, args):
    from evaluation.adamas_attention_qwen3 import adamas_forward_4_51_0

    if args.token_budget is None or args.token_budget <= 0:
        raise ValueError("Adamas requires a positive token budget")
    attention_layers = [
        module for module in model.modules()
        if isinstance(module, (LlamaAttention, Qwen3Attention))
    ]
    if not attention_layers:
        raise ValueError("Adamas supports Hugging Face Llama and Qwen3 attention")
    for module in attention_layers:
        module.token_budget = args.token_budget
        module.forward = types.MethodType(adamas_forward_4_51_0, module)
