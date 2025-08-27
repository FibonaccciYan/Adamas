from transformers import AutoTokenizer
import torch
import argparse

# MODEL_PATH = "/cpfs/2926428ee2463e44/shared/public/wendao/models/llama/Llama-2-7b-chat-hf"
MODEL_PATH = "/cpfs/user/yansiyuan/models/longchat-7b-v1.5-32k"
DEVICE = torch.device("cuda:0")
DTYPE = torch.float16
torch.set_default_dtype(DTYPE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

RUNTIME_CFGS = [
    "HSA",
    "hg",
    ]

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=RUNTIME_CFGS, default="HSA")
parser.add_argument("--token_budget", type=int, default=1024)
args = parser.parse_args()

if args.method == "HSA":
    from HSA import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE)

    # Init HSA Controller
    model.hsa_init(page_size=1, max_seq_len=8192, token_budget=args.token_budget)
else:
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE)
    
# First Round
# prompt = "In an animal kingdom, the lion is the king. One day, the lion announces a competition to choose the most hardworking animal. The turtle, rabbit, monkey, zebra, and giraffe all decide to participate. After a day of observation, the lion notices that all the animals are working hard, except for the rabbit, who is sleeping. So why does the lion choose the rabbit as the most hardworking animal?"
prompt = "Where did yellowstone national park get its name?"

if "longchat" in MODEL_PATH or "vicuna" in MODEL_PATH:
    from fastchat.model import get_conversation_template

    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
print(f"Input Sequence Length: {inputs.input_ids.shape[1]}")

generate_ids = model.generate(
                            inputs.input_ids,
                            max_new_tokens=128,
                            use_cache=True # Managed by our InferenceController
                            )
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])