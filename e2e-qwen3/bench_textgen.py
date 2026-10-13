import argparse
import time
import types
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import Qwen3ForCausalLM

import adamas.utils
import faster_hadamard_transform
from adamas.utils.controller import InferenceController
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


def adamas_rms_norm(self, hidden_states):
    shape = hidden_states.shape
    hidden_states = hidden_states.reshape(1, -1, shape[-1])
    return adamas.utils.rms_norm_forward(
        hidden_states, self.weight, self.variance_epsilon
    ).reshape(shape)


def adamas_attention(self, hidden_states, **kwargs):
    batch_size, query_len, _ = hidden_states.shape
    if batch_size != 1:
        raise ValueError("Adamas only supports batch size 1")

    shape = (batch_size, query_len, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(shape))[0]
    key_states = self.k_norm(self.k_proj(hidden_states).view(shape))[0]
    value_states = self.v_proj(hidden_states).view(shape)[0]

    controller = self.adamas_controller
    if self.layer_idx == 2:
        controller.end_forward()
        controller.set_page_budget(self.adamas_page_budget)
        controller.begin_forward(query_len, updateTensor=False)

    adamas.utils.apply_rope_in_place(
        query_states,
        key_states,
        controller.kv_cache.seqlen - query_len,
        rope_theta=self.config.rope_theta,
    )

    if query_len > 1:
        hadamard_key_states = faster_hadamard_transform.hadamard_transform(
            key_states, inplace=False
        )
        adamas.utils.append_kvh(
            key_states,
            value_states,
            hadamard_key_states,
            controller,
            self.layer_idx,
        )
        attention_output = adamas.utils.prefill_forward(
            query_states, controller, self.layer_idx
        )
    else:
        query_code = adamas.utils.append_kvh_decode_fused(
            query_states,
            key_states,
            value_states,
            controller,
            self.layer_idx,
        )
        if controller.need_estimate():
            adamas.utils.decode_estimate_topk(
                query_code, controller, self.layer_idx
            )
            indices = controller.topk_dindices_buffer
        else:
            indices = controller.kv_indices_without_last
        attention_output = adamas.utils.decode_sparse_attn(
            query_states, controller, self.layer_idx, indices
        )

    attention_output = attention_output.unsqueeze(0).reshape(
        batch_size, query_len, -1
    )
    return self.o_proj(attention_output), None


def enable_adamas(model, page_size, max_seq_len, token_budget, dtype, device):
    base_model = model.model
    config = model.config
    page_budget = token_budget // page_size
    controller = InferenceController(
        num_layers=config.num_hidden_layers,
        num_qo_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        page_size=page_size,
        page_budget=page_budget,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )

    for module in model.modules():
        if isinstance(module, Qwen3RMSNorm):
            module.forward = types.MethodType(adamas_rms_norm, module)
    for layer in base_model.layers:
        layer.self_attn.adamas_controller = controller
        layer.self_attn.adamas_page_budget = page_budget
        layer.self_attn.forward = types.MethodType(
            adamas_attention, layer.self_attn
        )

    base_model.rotary_emb.forward = types.MethodType(
        lambda self, hidden_states, position_ids: (None, None),
        base_model.rotary_emb,
    )
    base_model._update_causal_mask = types.MethodType(
        lambda self, *args, **kwargs: None, base_model
    )

    original_forward = base_model.forward

    def forward_with_controller(self, *args, **kwargs):
        input_tensor = kwargs.get("inputs_embeds")
        if input_tensor is None:
            input_tensor = kwargs.get("input_ids")
        if input_tensor is None:
            input_tensor = args[0]
        query_len = input_tensor.shape[1]
        controller.prepare_hadamard(query_len)
        controller.set_page_budget(1024 * 1024)
        controller.begin_forward(query_len)
        try:
            kwargs["use_cache"] = False
            return original_forward(*args, **kwargs)
        finally:
            controller.end_forward()

    base_model.forward = types.MethodType(forward_with_controller, base_model)
    model.adamas_clear = controller.clean_states
    print(f"Adamas allocates KV-Cache of {max_seq_len} tokens")
    print(f"Token budget is set to {token_budget}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default="/data1/model/qwen/Qwen/Qwen3-8B"
    )
    parser.add_argument("--context_len", type=int, default=2 * 1024)
    parser.add_argument("--decode_len", type=int, default=256)
    parser.add_argument("--page_size", type=int, default=1)
    parser.add_argument("--token_budget", type=int, default=256)
    parser.add_argument("--iteration", type=int, default=10)
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("test_results/e2e-qwen3/Qwen3-8B/e2e_Adamas.txt"),
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.float16
    torch.set_default_dtype(dtype)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        torch_dtype=dtype,
    ).eval()
    enable_adamas(
        model,
        page_size=args.page_size,
        max_seq_len=args.context_len + args.decode_len + 512,
        token_budget=args.token_budget,
        dtype=dtype,
        device=device,
    )

    hidden_size = model.config.hidden_size

    hidden_states = torch.randn(
        1, args.context_len, hidden_size, dtype=dtype, device=device
    )
    model.model(inputs_embeds=hidden_states)
    model.adamas_clear()

    prefill_latency = []
    decode_latency = []
    for _ in tqdm(range(args.iteration)):
        torch.cuda.empty_cache()

        torch.cuda.synchronize()
        start = time.perf_counter()
        hidden_states = torch.randn(
            1, args.context_len, hidden_size, dtype=dtype, device=device
        )
        model.model(inputs_embeds=hidden_states)
        torch.cuda.synchronize()
        prefill_latency.append(time.perf_counter() - start)

        for _ in range(args.decode_len):
            torch.cuda.synchronize()
            start = time.perf_counter()
            hidden_states = torch.randn(
                1, 1, hidden_size, dtype=dtype, device=device
            )
            model.model(inputs_embeds=hidden_states)
            torch.cuda.synchronize()
            decode_latency.append(time.perf_counter() - start)

        model.adamas_clear()
    peak_memory_gib = torch.cuda.max_memory_allocated(device) / 1024**3

    result = (
        f"{args.page_size},{args.token_budget},{args.context_len},"
        f"{args.decode_len},{np.mean(prefill_latency)},{np.mean(decode_latency)},{peak_memory_gib}"
    )
    print("page_size,token_budget,context_len,decode_len,avg_prefill_latency,avg_decode_latency,peak_memory_gib")
    print(result)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("a") as output:
        output.write(result + "\n")


if __name__ == "__main__":
    main()
