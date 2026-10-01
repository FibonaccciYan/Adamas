# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch
from types import SimpleNamespace
from typing import List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

QWEN3_ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768


def _normalize_method() -> str:
    method = os.getenv('METHOD', 'full').strip().lower()
    aliases = {
        'adamas': 'adamas',
        'hf': 'full',
        'full': 'full',
        'none': 'full',
    }
    if method not in aliases:
        raise ValueError(f'Unsupported METHOD: {method}; use adamas or full.')
    return aliases[method]


def _current_ruler_seq_length() -> int:
    try:
        return int(os.getenv('RULER_CURRENT_SEQ_LENGTH', '0'))
    except ValueError:
        return 0


def _qwen3_yarn_factor(seq_length: int) -> float:
    if seq_length <= QWEN3_ORIGINAL_MAX_POSITION_EMBEDDINGS:
        return 1.0
    return seq_length / QWEN3_ORIGINAL_MAX_POSITION_EMBEDDINGS


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.method = _normalize_method()
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        lower_name_or_path = name_or_path.lower()
        self.is_llama = 'llama' in lower_name_or_path
        self.is_qwen3 = 'qwen3' in lower_name_or_path
        if self.is_llama and self.method == 'adamas':
            from evaluation.llama import enable_tuple_kv_cache_for_llama
            enable_tuple_kv_cache_for_llama()

        model_kwargs = {}
        if 'Yarn-Llama' not in name_or_path:
            model_kwargs['attn_implementation'] = 'flash_attention_2'

        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        if self.is_qwen3:
            seq_length = _current_ruler_seq_length()
            yarn_factor = _qwen3_yarn_factor(seq_length)
            if yarn_factor > 1.0:
                config.rope_scaling = {
                    'rope_type': 'yarn',
                    'factor': yarn_factor,
                    'original_max_position_embeddings': QWEN3_ORIGINAL_MAX_POSITION_EMBEDDINGS,
                }
                config.max_position_embeddings = int(QWEN3_ORIGINAL_MAX_POSITION_EMBEDDINGS * yarn_factor)
                print(
                    f'Qwen3 YaRN enabled: seq_length={seq_length}, '
                    f'factor={yarn_factor:g}, max_position_embeddings={config.max_position_embeddings}'
                )

        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            config=config,
            **model_kwargs,
        )
        self.model.eval()

        if self.method == 'adamas':
            adamas_args = SimpleNamespace(
                token_budget=int(os.getenv('ADAMAS_TOKEN_BUDGET', '1024')),
            )
            from evaluation.attention import enable_adamas_attention_eval
            enable_adamas_attention_eval(self.model, adamas_args)

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def _generate_one_sparse(self, prompt: str) -> str:
        if self.method == 'adamas' and self.is_qwen3:
            from evaluation.adamas_cache import AdamasDynamicCache
            past_key_values = AdamasDynamicCache()
        else:
            past_key_values = None

        max_new_tokens = int(self.generation_kwargs.get('max_new_tokens', 0))
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        generated_tokens = []

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.get('attention_mask'),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())

            for _ in range(max_new_tokens - 1):
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                token_id = next_token.item()
                generated_tokens.append(token_id)
                if token_id == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.method == 'adamas':
            generated_texts = [self._generate_one_sparse(prompt) for prompt in prompts]
        else:
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **self.generation_kwargs,
                )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        results = []
        for text, prompt in zip(generated_texts, prompts):
            tokenized_prompt = self.tokenizer(prompt, return_tensors='pt', padding=True)
            prompt_decoded = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(prompt_decoded):
                text = text[len(prompt_decoded):]
            elif text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results
