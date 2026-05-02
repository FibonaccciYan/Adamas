import argparse
import json
import os
import random
import re

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.adamas_attention import enable_adamas_attention_eval as enable_adamas_attention_eval_llama3
from evaluation.adamas_attention_qwen3 import enable_adamas_attention_eval as enable_adamas_attention_eval_qwen3
from evaluation.adamas_cache import AdamasDynamicCache


SYSTEM_PROMPT = (
    # "Solve the following math problem step by step. Put your answer inside \boxed{{}}."
    "Please reason step by step, and put your final answer within \boxed{}."
    "{question}"
    "Remember to put your answer inside \boxed{}."
)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["Meta-Llama-3.1-8B-Instruct", "Qwen3-8b"],
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="pred")
    parser.add_argument("--Adamas", action="store_true", help="Enable Adamas Attention")
    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 thinking mode (only valid when --model is set to Qwen3)")
    return parser.parse_args(args)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def extract_answer(text: str):
    boxed = re.findall(r"\\boxed\{(-?\d+)\}", text)
    if boxed:
        return int(boxed[-1])

    tagged = re.findall(r"Answer:\s*(-?\d+)", text)
    if tagged:
        return int(tagged[-1])

    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def resolve_model_path(model_name: str) -> str:
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "LongBench", "config", "model2path.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        model2path = json.load(f)
    return model2path[model_name]


def load_model_and_tokenizer(model_name, args):
    if "llama" in model_name.lower():
        from evaluation.llama import enable_tuple_kv_cache_for_llama
        enable_tuple_kv_cache_for_llama()

    model_path = resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model = model.eval()

    if args.Adamas:
        if "llama" in model_name.lower():
            enable_adamas_attention_eval_llama3(model, args)
        elif "qwen3" in model_name.lower():
            enable_adamas_attention_eval_qwen3(model, args)

    return model, tokenizer


def build_prompt(model_name, tokenizer, problem, enable_thinking=False):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{problem}"},
    ]
    if "llama3" in model_name.lower():
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif "qwen3" in model_name.lower():
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
    return prompt


def generate_one(model, tokenizer, problem, max_new_tokens, model_name, enable_thinking=False):
    prompt = build_prompt(model_name, tokenizer, problem, enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to("cuda")

    if "qwen3" in model_name.lower():
        past_key_values = AdamasDynamicCache()

        generated_ids = model.generate(
            **inputs,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
        )
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content
    else:
        past_key_values = None

        with torch.no_grad():
            output = model(
                input_ids=inputs.input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]

            for _ in range(max_new_tokens - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content.append(pred_token_idx.item())
                if pred_token_idx.item() == tokenizer.eos_token_id:
                    break

        generated_ids = torch.tensor(generated_content, device=inputs.input_ids.device)
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, dataset, args):
    correct = 0
    results = []

    for ex in tqdm(dataset):
        problem = ex["Problem"]
        gold = int(ex["Answer"])
        generated = generate_one(model, tokenizer, problem, args.max_new_tokens, args.model, args.thinking)
        pred = extract_answer(generated)
        is_correct = pred == gold
        correct += int(is_correct)

        results.append(
            {
                "id": ex["ID"],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw_output": generated,
            }
        )

    accuracy = correct / len(results) if results else 0.0
    summary = {
        "model": args.model,
        "split": args.split,
        "adamas": args.Adamas,
        "token_budget": args.token_budget if args.Adamas else None,
        "chunk_size": args.chunk_size if args.Adamas else None,
        "max_new_tokens": args.max_new_tokens,
        "num_samples": len(results),
        "correct": correct,
        "accuracy": accuracy,
    }
    return results, summary


def save_outputs(results, summary, args):
    model_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)

    suffix = f"{args.split}-{args.token_budget}" if args.Adamas else f"{args.split}-full"
    pred_path = os.path.join(model_dir, f"aime-{suffix}.jsonl")
    summary_path = os.path.join(model_dir, f"aime-{suffix}-summary.json")

    with open(pred_path, "w", encoding="utf-8") as f:
        for row in results:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return pred_path, summary_path


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split=args.split)
    model, tokenizer = load_model_and_tokenizer(args.model, args)
    results, summary = evaluate(model, tokenizer, dataset, args)
    pred_path, summary_path = save_outputs(results, summary, args)

    print(
        f"Accuracy: {summary['accuracy']:.4f} "
        f"({summary['correct']}/{summary['num_samples']})"
    )
    print(f"Predictions saved to: {pred_path}")
    print(f"Summary saved to: {summary_path}")
