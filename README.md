# Adamas: Paper Reproduction

This branch contains Adamas and full-attention reproduction code for LongBench, RULER,
AIME 2024/2025, Passkey, PG19, and end-to-end decoding benchmarks.
Datasets, predictions, logs, historical results, and figures are not distributed.
Standalone NIAH, other sparse baselines, and ablation experiments are not included.

## Installation

Run the installation commands from the repository root. For a Git checkout,
initialize the pinned third-party dependencies:

```bash
git submodule update --init --recursive
```

Anonymous ZIP downloads do not include submodule contents or the root `.git`
directory. Instead, fetch each third-party repository at the exact commit below
into its expected location. Use fresh or empty dependency directories:

```bash
fetch_dependency() (
    set -e
    local directory="$1" repository="$2" commit="$3"
    git init "$directory"
    git -C "$directory" fetch --depth 1 "$repository" "$commit"
    git -C "$directory" checkout --detach FETCH_HEAD
    git -C "$directory" submodule update --init --recursive
)

mkdir -p kernels/3rdparty
fetch_dependency kernels/3rdparty/flashinfer https://github.com/flashinfer-ai/flashinfer 9f49803b1db0a40ea0019ad98b8bb5d4f1593c77
fetch_dependency kernels/3rdparty/pybind https://github.com/pybind/pybind11 768cebe17e65c2a0a64ed067510729efc3c7ff6c
fetch_dependency kernels/3rdparty/applied-ai https://github.com/meta-pytorch/applied-ai e51188201328043a3a9c6413af9f740449ecff70
```

These commands create independent Git checkouts inside the extracted ZIP;
they do not require the Adamas source repository or its Git history.
Both installation methods preserve third-party licenses and attribution.
After either method, install the Python dependencies:

```bash
pip install -e .
pip install ninja packaging
pip install flash-attn==2.5.8 --no-build-isolation
pip install -e kernels/3rdparty/applied-ai/kernels/cuda/inference/hadamard_transform
```

Use separate environments: Llama-3.1 and LongChat use Transformers 4.45.2;
Qwen3 uses Transformers 4.51.0. In the Qwen3 environment, run
`pip install transformers==4.51.0` after installing this package.
Use PyTorch 2.5.0 and FlashAttention 2.5.8 in both environments.
The editable installation pins the Llama environment; reinstalling it resets
Transformers. GPU evaluations require CUDA and access to the model weights.
LongChat uses FastChat's `fastchat.model` module, provided by the `fschat` dependency.

E2E additionally requires CMake >= 3.26.4, a CUDA compiler, and compiled operators.
The build downloads RAPIDS/RAFT dependencies automatically and requires network access:

```bash
(cd adamas/ops && bash setup.sh)
```

RULER's Hugging Face data-generation and evaluation dependencies are included
in this package. Use the model-specific Transformers versions above.
LongBench, AIME, and PG19 download their datasets through Hugging Face.
RULER also needs SQuAD/HotpotQA and Paul Graham source data; follow its README.

## Run

All entries work from any current directory. Select the environment with
`PYTHON_BIN`, and optionally provide a local checkpoint using `MODEL_PATH`.
For example, use `MODEL_PATH=/path/to/Llama-3.1-8B-Instruct` for local weights;
otherwise the scripts use public Hugging Face model IDs.
Adamas and full attention run by default where the paper reports both.
Passkey runs Adamas only, matching the absence of a Full row in Table 5.

```bash
bash scripts/longbench.sh
MODEL=longchat-v1.5-7b-32k bash scripts/longbench.sh
MODEL=Qwen3-8b PYTHON_BIN=/path/to/qwen3/python bash scripts/longbench.sh
bash scripts/ruler.sh
PYTHON_BIN=/path/to/qwen3/python bash scripts/aime.sh
bash scripts/passkey.sh
MODEL_PATH=lmsys/longchat-7b-v1.5-32k CONTEXT_LENGTH=32000 bash scripts/passkey.sh
bash scripts/ppl_eval.sh
MODEL=longchat-7b-v1.5-32k CONTEXT_LENGTHS="49152 57344" bash scripts/bench_efficiency_e2e.sh
```

| Entry | Defaults / controls |
| --- | --- |
| LongBench | Llama-3.1; six paper tasks; `MODEL`, `TASKS`, `BUDGETS` |
| RULER | Llama-3.1; all synthetic tasks, 4K-128K; `MODEL=llama3.1-8b-instruct` or `qwen3-8b`, `GPU`, `BUDGETS`, `RULER_SEQ_LENGTHS` (comma-separated), `RULER_TASKS`, `RULER_NUM_SAMPLES` |
| AIME | Both 2024 and 2025, Qwen3 thinking mode, greedy decoding; budgets 1024/2048/4096 and Full; `BENCHMARKS`, `BUDGETS_AIME`, `REPEATS=3`, `MAX_NEW_TOKENS=38912` |
| Passkey | Yarn-Llama-2-7B-128K, length argument 100000; `MODEL_PATH`, `CONTEXT_LENGTH`, `ITERATIONS=100`, `BUDGETS_PASSKEY` |
| PG19 | Llama-3.1, first PG19 test document; `MODEL_PATH`, `NUM_EVAL_TOKENS=32000`, `BUDGETS` |
| E2E | Llama-3.1, 32K; `MODEL` (Llama-3.1 or LongChat), `MODEL_PATH`, `CONTEXT_LENGTHS`, `DECODE_LENGTH=256`, `ITERATIONS=10`, `BUDGETS` |

The shared accuracy interface in `evaluation/attention.py` dispatches by
`model.config.model_type`: Llama-3.1/LongChat use the Llama implementation;
Qwen3 uses its own implementation. E2E uses the existing CUDA Llama pipeline
and does not support Qwen3. Yarn's remote implementation requires separate
runtime verification.

The Passkey evaluator's historical `--fixed-length` argument scales the filler
character count and reports the actual token count; it is not an exact token
length guarantee. AIME repeats save into separate directories. LongBench has
an evaluation step, RULER generates task summaries, AIME saves answer scores,
Passkey saves retrieval results, PG19 saves NLL/PPL, and E2E saves latency logs.
Outputs are ignored by Git. Each experiment script directly invokes its evaluator.
E2E uses a token budget covering the allocated cache for its existing Full path.
