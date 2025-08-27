import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import numpy as np
import random
import argparse

from hadamard_attention import enable_hadamard_attention_eval, enable_hadamard_dynamic_cache_for_llama
from hadamard_cache import HadamardDynamicCache


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama/Llama-2-7b-chat-hf",
            "llama/Meta-Llama-3-8B-Instruct",
            "lmsys/longchat-7b-v1.5-32k",
        ],
    )
    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--hadamard", action="store_true", help="Enable Hadamard Attention")

    return parser.parse_args(args)


def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
    )
    model = model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=10, past_key_values=None):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids, 
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            # output_attentions=True,
        )
        past_key_values = outputs.past_key_values

        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_content = [pred_token_idx.item()]
        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                # output_attentions=True,
            )

            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content += [pred_token_idx.item()]
            if pred_token_idx.item() == tokenizer.eos_token_id:
                break
    
    response = tokenizer.decode(generated_content, skip_special_tokens=False)
    return response, generated_content


def get_attn_weights(model, tokenizer, prompt, max_new_tokens=10, model_type="Vanilla"):
    # attns_weights: [output_len * [layers * Tensor[batch_size, num_heads, q_len, kv_len]]]
    # attns_weights_v: [output_len * [layers * Tensor[batch_size, num_heads, q_len, kv_len]]]
    past_key_values = HadamardDynamicCache() if model_type == "Hadamard" else None
    response, output_ids = generate_response(model, tokenizer, prompt, max_new_tokens, past_key_values)

    prefill_len = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.shape[1]
    seperated_response = []
    for tok in output_ids:
        tok = tokenizer.decode(tok) 
        tok = tok.replace(" ", "<space>")
        tok = tok.replace("\n", "<newline>")
        seperated_response.append(tok)

    print("")
    print(f"{model_type}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"response after budget: {seperated_response[64-prefill_len:]}")
    print("")

    return output_ids


if __name__ == "__main__":
    seed_everything(42)

    args = parse_args()
    layers = 32
    nh = 32

    # MODEL_DIR = "/cpfs/2926428ee2463e44/shared/public/wendao/models/"
    # model_path = os.path.join(MODEL_DIR, args.model)
    model_path = "/cpfs/user/yansiyuan/models/longchat-7b-v1.5-32k"
    prompt = "Where did yellowstone national park get its name?"
    # prompt = "What can i do for lunch with tomato, egg, and bread?"

    if "longchat" in model_path or "vicuna" in model_path:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    max_new_tokens = 128

    # Vanilla
    model, tokenizer = load_model_and_tokenizer(model_path)
    vanilla_output_ids = get_attn_weights(model, tokenizer, prompt, max_new_tokens, "Vanilla")

    # Hadamard
    enable_hadamard_dynamic_cache_for_llama()
    model, tokenizer = load_model_and_tokenizer(model_path)
    enable_hadamard_attention_eval(model, args)
    hadamard_output_ids = get_attn_weights(model, tokenizer, prompt, max_new_tokens, "Hadamard")
    