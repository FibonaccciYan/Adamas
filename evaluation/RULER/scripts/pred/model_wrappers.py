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
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

QWEN3_ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768


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


def _current_ruler_seq_length() -> int:
    try:
        return int(os.getenv('RULER_CURRENT_SEQ_LENGTH', '0'))
    except ValueError:
        return 0


def _qwen3_yarn_factor(seq_length: int) -> float:
    if seq_length <= QWEN3_ORIGINAL_MAX_POSITION_EMBEDDINGS:
        return 1.0
    return seq_length / QWEN3_ORIGINAL_MAX_POSITION_EMBEDDINGS


def _streamingllm_config() -> Optional[Tuple[int, int]]:
    window_size = os.getenv('STREAMINGLLM_WINDOW_SIZE', os.getenv('TOKEN_BUDGET', ''))
    if not window_size:
        return None

    window_size = int(window_size)
    if window_size <= 0:
        return None

    sink_size = int(os.getenv('STREAMINGLLM_SINK_SIZE', '4'))
    sink_size = min(max(sink_size, 0), window_size)
    return window_size, sink_size


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.method = _normalize_method()
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        lower_name_or_path = name_or_path.lower()
        self.is_llama = 'llama' in lower_name_or_path
        self.is_qwen3 = 'qwen3' in lower_name_or_path
        self.streamingllm_config = _streamingllm_config() if self.method == 'streamingllm' else None

        if self.is_llama and self.method in {'adamas', 'streamingllm'}:
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
            if self.is_llama:
                from evaluation.adamas_attention import enable_adamas_attention_eval
                enable_adamas_attention_eval(self.model, adamas_args)
            elif self.is_qwen3:
                from evaluation.adamas_attention_qwen3 import enable_adamas_attention_eval
                enable_adamas_attention_eval(self.model, adamas_args)
        elif self.method == 'streamingllm':
            if self.streamingllm_config is None:
                raise ValueError('METHOD=streamingllm requires STREAMINGLLM_WINDOW_SIZE or TOKEN_BUDGET.')
            window_size, sink_size = self.streamingllm_config
            self.streamingllm_recent_size = window_size - sink_size
            from evaluation.streamingllm_attention import (
                build_streamingllm_cache,
                enable_llama_pos_shift_attention,
                enable_qwen3_pos_shift_attention,
            )
            self.streamingllm_kv_cache = build_streamingllm_cache(
                name_or_path,
                start_size=sink_size,
                recent_size=self.streamingllm_recent_size,
            )
            if self.is_llama:
                enable_llama_pos_shift_attention(self.model)
            elif self.is_qwen3:
                enable_qwen3_pos_shift_attention(self.model)
            else:
                raise ValueError(f'METHOD=streamingllm is only wired for Llama/LongChat/Qwen3, got {name_or_path}.')
            print(
                f'StreamingLLM enabled: window_size={window_size}, '
                f'sink_size={sink_size}, recent_size={self.streamingllm_recent_size}'
            )
        else:
            self.streamingllm_kv_cache = None
            self.streamingllm_recent_size = None

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
            past_key_values = self._maybe_prune_streamingllm_cache(outputs.past_key_values)
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

    def _maybe_prune_streamingllm_cache(self, past_key_values):
        if self.method != 'streamingllm':
            return past_key_values

        if self.is_qwen3:
            from evaluation.streamingllm_attention import prune_dynamic_cache_start_recent
            _, sink_size = self.streamingllm_config
            return prune_dynamic_cache_start_recent(
                past_key_values,
                start_size=sink_size,
                recent_size=self.streamingllm_recent_size,
                device=self.model.device,
            )

        return self.streamingllm_kv_cache(past_key_values)

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
