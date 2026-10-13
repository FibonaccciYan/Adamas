"""Check prefill, cached decoding and generation on local Llama/Qwen3 checkpoints."""

import argparse
import gc
from types import SimpleNamespace

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from evaluation.attention import enable_adamas_attention_eval
from evaluation.adamas_cache import AdamasDynamicCache
from evaluation.llama import enable_tuple_kv_cache_for_llama


@torch.inference_mode()
def decode(model, inputs, steps, custom_cache=False):
    cache = AdamasDynamicCache() if custom_cache else None
    mask = inputs.attention_mask
    output = model(**inputs, past_key_values=cache, use_cache=True)
    logits = [output.logits[:, -1].float()]
    tokens = []
    for _ in range(steps):
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(token)
        mask = torch.cat([mask, torch.ones_like(token)], dim=-1)
        output = model(input_ids=token, attention_mask=mask,
                       past_key_values=output.past_key_values, use_cache=True)
        logits.append(output.logits[:, -1].float())
        assert torch.isfinite(output.logits).all()
    assert output.past_key_values.get_seq_length() == inputs.input_ids.shape[1] + steps
    return torch.cat(tokens, dim=-1), logits


@torch.inference_mode()
def check(path, args):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(["The capital of France is", "Answer briefly: What is 2 + 2?"],
                       padding=True, return_tensors="pt").to(args.device)
    config = AutoConfig.from_pretrained(path)
    if args.tiny:
        config.hidden_size = 256
        config.intermediate_size = 512
        config.num_hidden_layers = 3
        config.num_attention_heads = 4
        config.num_key_value_heads = 4 if "longchat" in path.lower() else 2
        config.head_dim = 64
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.bfloat16, attn_implementation=args.backend
        ).to(args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation=args.backend,
            device_map={"": args.device}, local_files_only=True
        )
    model.eval()
    full_tokens, full_logits = decode(model, inputs, args.steps)
    generation_args = dict(max_new_tokens=args.steps, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
    full_generated = model.generate(**inputs, **generation_args)

    enable_adamas_attention_eval(model, SimpleNamespace(
        token_budget=inputs.input_ids.shape[1] + args.steps
    ))
    for custom_cache in [False, True]:
        tokens, logits = decode(model, inputs, args.steps, custom_cache)
        assert torch.equal(tokens, full_tokens), "Full-budget greedy tokens differ"
        for actual, expected in zip(logits, full_logits):
            torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
        cache_args = {"past_key_values": AdamasDynamicCache()} if custom_cache else {}
        generated = model.generate(**inputs, **generation_args, **cache_args)
        assert torch.equal(generated, full_generated), "Full-budget generation differs"

    enable_adamas_attention_eval(model, SimpleNamespace(token_budget=args.budget))
    sparse_tokens, _ = decode(model, inputs, args.steps, custom_cache=True)
    sparse_generated = model.generate(**inputs, **generation_args,
                                      past_key_values=AdamasDynamicCache())
    assert sparse_generated.shape[1] > inputs.input_ids.shape[1]
    print(f"PASS {path}: Full/Adamas, native/custom cache, padding, prefill, decode, generate",
          flush=True)
    if not args.tiny:
        print("Full:", tokenizer.batch_decode(full_tokens, skip_special_tokens=True), flush=True)
        print("Adamas:", tokenizer.batch_decode(sparse_tokens, skip_special_tokens=True), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=["eager", "sdpa", "flash_attention_2"],
                        default="flash_attention_2")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.budget <= 0:
        parser.error("steps and budget must be positive")
    assert transformers.__version__ == "4.51.0", transformers.__version__
    from transformers.models.llama.modeling_llama import LlamaModel
    original = LlamaModel.forward
    enable_tuple_kv_cache_for_llama()
    assert LlamaModel.forward is original
    cache = AdamasDynamicCache()
    tensor = torch.arange(8).reshape(2, 1, 1, 4)
    cache.update(tensor, tensor + 1, tensor + 2, 0)
    indices = torch.tensor([1, 0, 1])
    cache.reorder_cache(indices)
    assert torch.equal(cache.hadamard_cache[0], (tensor + 2)[indices])
    torch.manual_seed(42)
    for path in args.models:
        check(path, args)


if __name__ == "__main__":
    main()
