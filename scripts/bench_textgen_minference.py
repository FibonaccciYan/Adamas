# MInference prefill + Adamas decode. Kept separate from bench_textgen.py.

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
from tqdm.auto import tqdm

from adamas import LlamaForCausalLM


@dataclasses.dataclass
class ModelConfig:
    model_path: str
    dtype: str = "float16"
    device: str = "cuda:0"


MODEL_CFGS = {
    "Llama-3.1-8B-Instruct": ModelConfig(
        model_path="/data1/model/llama3/meta-llama/Llama-3.1-8B-Instruct"
    ),
}


def load_model(model_cfg: ModelConfig):
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    torch.set_default_dtype(dtype)
    with device:
        return LlamaForCausalLM.from_pretrained(
            model_cfg.model_path,
            device_map=device,
            torch_dtype=dtype,
        )


def load_minference_kernel(minference_path: Path):
    package_root = minference_path / "minference"
    module_name = "_adamas_minference_reference.ops.pit_sparse_flash_attention_v3"
    module_path = package_root / "ops" / "pit_sparse_flash_attention_v3.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"MInference kernel not found: {module_path}")

    if module_name not in sys.modules:
        for package_name, package_path in (
            ("_adamas_minference_reference", package_root),
            ("_adamas_minference_reference.ops", package_root / "ops"),
        ):
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            sys.modules[package_name] = package

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return sys.modules[module_name].minference_flash_attn_triton_func


def enable_minference_prefill(model, minference_path: Path, config_path: Path):
    pattern_config = json.loads(config_path.read_text())
    num_layers = model._config.num_hidden_layers
    num_heads = model._config.num_attention_heads
    if len(pattern_config) != num_layers:
        raise ValueError(f"MInference config has {len(pattern_config)} layers, expected {num_layers}")

    layer_sizes = []
    for layer_idx, layer_pattern in enumerate(pattern_config):
        try:
            patterns = [layer_pattern[str(head_idx)] for head_idx in range(num_heads)]
        except KeyError as error:
            raise ValueError(f"MInference config layer {layer_idx} is missing head {error.args[0]}") from error
        unsupported = {pattern[0] for pattern in patterns} - {"vertical_and_slash"}
        if unsupported:
            raise ValueError(f"Unsupported MInference patterns in layer {layer_idx}: {sorted(unsupported)}")
        layer_sizes.append(
            ([int(pattern[1]) for pattern in patterns], [int(pattern[2]) for pattern in patterns])
        )

    minference_kernel = load_minference_kernel(minference_path)

    def prefill_attention(query_states, key_states, value_states, layer_idx):
        vertical_sizes, slash_sizes = layer_sizes[layer_idx]
        return minference_kernel(
            query_states.unsqueeze(0),
            key_states.unsqueeze(0),
            value_states.unsqueeze(0),
            vertical_sizes,
            slash_sizes,
            dropout_p=0.0,
            granularity=128,
            causal=True,
        ).squeeze(0)

    for layer in model.model.layers:
        layer.self_attn.prefill_attention = prefill_attention


@torch.inference_mode()
def benchmark_minference_adamas():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CFGS.keys(), default="Llama-3.1-8B-Instruct")
    parser.add_argument("--context_len", type=int, default=2 * 1024)
    parser.add_argument("--decode_len", type=int, default=256)
    parser.add_argument("--page_size", type=int, default=1)
    parser.add_argument("--token_budget", type=int, default=256)
    parser.add_argument("--iteration", type=int, default=10)
    parser.add_argument("--empty_cache_after_prefill", action="store_true")
    parser.add_argument("--minference_path", type=Path, default=Path("/data0/ysy/MInference"))
    parser.add_argument("--minference_config", type=Path)
    args = parser.parse_args()

    model_cfg = MODEL_CFGS[args.model]
    model = load_model(model_cfg)
    dtype = getattr(torch, model_cfg.dtype)
    device = torch.device(model_cfg.device)
    model.Adamas_init(
        page_size=args.page_size,
        max_seq_len=args.context_len + args.decode_len + 512,
        token_budget=args.token_budget,
        dtype=dtype,
        device=device,
    )
    config_path = args.minference_config or (
        args.minference_path
        / "minference/configs/Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern_v2.json"
    )
    enable_minference_prefill(model, args.minference_path, config_path)
    hidden_size = model._config.hidden_size

    hidden_states = torch.randn(1, args.context_len, hidden_size, dtype=dtype, device=device)
    model.model(inputs_embeds=hidden_states)
    hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
    model.model(inputs_embeds=hidden_states)
    torch.cuda.synchronize()
    model.Adamas_clear()

    prefill_latency = []
    cleanup_latency = []
    decode_latency = []
    first_decode_latency = []
    steady_decode_latency = []
    for _ in tqdm(range(args.iteration)):
        torch.cuda.empty_cache()

        torch.cuda.synchronize()
        ts = time.perf_counter()
        hidden_states = torch.randn(1, args.context_len, hidden_size, dtype=dtype, device=device)
        model.model(inputs_embeds=hidden_states)
        torch.cuda.synchronize()
        prefill_latency.append(time.perf_counter() - ts)

        if args.empty_cache_after_prefill:
            ts = time.perf_counter()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            cleanup_latency.append(time.perf_counter() - ts)

        for decode_idx in range(args.decode_len):
            ts = time.perf_counter()
            hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
            model.model(inputs_embeds=hidden_states)
            torch.cuda.synchronize()
            latency = time.perf_counter() - ts
            decode_latency.append(latency)
            (first_decode_latency if decode_idx == 0 else steady_decode_latency).append(latency)

        model.Adamas_clear()

    avg_prefill_latency = np.mean(prefill_latency)
    avg_cleanup_latency = np.mean(cleanup_latency) if cleanup_latency else 0.0
    avg_first_decode_latency = np.mean(first_decode_latency)
    avg_steady_decode_latency = np.mean(steady_decode_latency) if steady_decode_latency else float("nan")
    avg_decode_latency = np.mean(decode_latency)
    avg_e2e_latency = avg_prefill_latency + avg_cleanup_latency + args.decode_len * avg_decode_latency
    print("prefill_backend=MInference,decode_backend=Adamas")
    print(
        "page_size,token_budget,context_len,decode_len,empty_cache_after_prefill,"
        "avg_prefill_latency,avg_cleanup_latency,avg_first_decode_latency,"
        "avg_steady_decode_latency,avg_decode_latency,avg_e2e_latency"
    )
    print(
        f"{args.page_size},{args.token_budget},{args.context_len},{args.decode_len},"
        f"{int(args.empty_cache_after_prefill)},{avg_prefill_latency},{avg_cleanup_latency},"
        f"{avg_first_decode_latency},{avg_steady_decode_latency},{avg_decode_latency},{avg_e2e_latency}"
    )

    output_file = (
        Path(__file__).resolve().parents[1]
        / "test_results"
        / "e2e-minference"
        / args.model
        / "e2e_MInference_Adamas.txt"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_file.exists() or output_file.stat().st_size == 0
    with output_file.open("a") as file:
        if write_header:
            file.write(
                "page_size,token_budget,context_len,decode_len,empty_cache_after_prefill,"
                "avg_prefill_latency,avg_cleanup_latency,avg_first_decode_latency,"
                "avg_steady_decode_latency,avg_decode_latency,avg_e2e_latency\n"
            )
        file.write(
            f"{args.page_size},{args.token_budget},{args.context_len},{args.decode_len},"
            f"{int(args.empty_cache_after_prefill)},{avg_prefill_latency},{avg_cleanup_latency},"
            f"{avg_first_decode_latency},{avg_steady_decode_latency},{avg_decode_latency},"
            f"{avg_e2e_latency}\n"
        )


if __name__ == "__main__":
    benchmark_minference_adamas()
