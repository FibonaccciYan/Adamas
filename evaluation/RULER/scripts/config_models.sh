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

TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"
SEQ_LENGTHS=(131072 65536 32768 16384 8192 4096)

MODEL_SELECT() {
    local model_name="$1" model_path template
    case "$model_name" in
        llama3.1-8b-instruct)
            model_path="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
            template="meta-llama3"
            ;;
        qwen3-8b)
            model_path="${MODEL_PATH:-Qwen/Qwen3-8B}"
            template="qwen3"
            ;;
        *)
            echo "Unsupported model: $model_name" >&2
            return 1
            ;;
    esac
    echo "$model_path:$template:hf:$model_path:hf"
}
