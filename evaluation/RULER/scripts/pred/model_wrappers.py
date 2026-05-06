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
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def _normalize_method() -> str:
    method = os.getenv('METHOD', 'full').strip().lower()
    aliases = {
        'adamas': 'adamas',
        'quest': 'quest',
        'streamingllm': 'streamingllm',
        'streaming_llm': 'streamingllm',
        'streaming': 'streamingllm',
        'hf': 'full',
        'full': 'full',
        'none': 'full',
    }
    return aliases.get(method, method)


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.method = _normalize_method()
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        is_llama = 'llama' in name_or_path.lower()

        if is_llama and self.method in {'adamas', 'streamingllm'}:
            from evaluation.llama import enable_tuple_kv_cache_for_llama
            enable_tuple_kv_cache_for_llama()

            # if self.method == 'adamas':
            #     from evaluation.adamas_attention import (
            #         enable_adamas_attention_eval,
            #         enable_adamas_dynamic_cache_for_llama,
            #     )
            #     enable_adamas_dynamic_cache_for_llama()
            # elif self.method == 'streamingllm':
            #     from evaluation.streamingllm_attention import (
            #         enable_streamingllm_attention_eval,
            #         enable_streamingllm_dynamic_cache_for_llama,
            #     )
            #     enable_streamingllm_dynamic_cache_for_llama()

        model_kwargs = {}
        if 'Yarn-Llama' not in name_or_path:
            model_kwargs['attn_implementation'] = 'flash_attention_2'

        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()

        if is_llama and self.method == 'adamas':
            from evaluation.adamas_attention import enable_adamas_attention_eval
            adamas_args = SimpleNamespace(
                token_budget=int(os.getenv('ADAMAS_TOKEN_BUDGET', '1024')),
                bucket_threshold=int(os.getenv('ADAMAS_BUCKET_THRESHOLD', '10')),
            )
            enable_adamas_attention_eval(self.model, adamas_args)
        elif is_llama and self.method == 'streamingllm':
            from evaluation.streamingllm_attention import enable_streamingllm_attention_eval
            sink_size = int(os.getenv('STREAMINGLLM_SINK_SIZE', '4'))
            window_size = int(os.getenv('STREAMINGLLM_WINDOW_SIZE', '1024'))
            enable_streamingllm_attention_eval(self.model, sink_size=sink_size, window_size=window_size)
        elif self.method == 'quest':
            print('METHOD=quest currently falls back to the plain Hugging Face baseline in RULER; no dedicated Quest patch is present in this repo path.')

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def _generate_one_sparse(self, prompt: str) -> str:
        max_new_tokens = int(self.generation_kwargs.get('max_new_tokens', 0))
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        generated_tokens = []

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.get('attention_mask'),
                past_key_values=None,
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
        if self.method in {'adamas', 'streamingllm'}:
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


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        self.device = 'cuda'
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        tokens = self.tokenizer(prompt, return_tensors='pt')
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
