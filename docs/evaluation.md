# Inference and evaluation

FocusVTC uses an OpenAI-compatible vLLM backend and a local `zoom_region`
gateway. The gateway shares the tool parser, box validation, and crop geometry
with the GRPO environment. Models are loaded from a merged Hugging Face
checkpoint; model files and benchmark data are not included.

Export a local GRPO FSDP checkpoint in the training environment with:

```bash
python eval/tool_agent/merge_fsdp_checkpoint.py \
  --actor outputs/grpo/global_step_100/actor \
  --base-model models/focusvtc-sft --target models/focusvtc-grpo
```

The exporter supports one-dimensional, dim-0 DTensor shards and requires an
empty target directory. It needs `torch` and `safetensors`; other distributed
checkpoint layouts require their corresponding exporter.

## Installation

Use Python 3.10 or newer. Install the lightweight client and gateway dependencies:

```bash
pip install -r eval/requirements.txt
```

To render benchmark inputs, also install `eval/rendering/requirements.txt`
and the system Poppler tools. The [rendering guide](../eval/rendering/README.md)
covers dataset sources, fonts, and commands; the generic document renderer
uses PDFium and does not need Poppler.

Install the backend in a separate GPU environment when its PyTorch/CUDA
requirements differ from training:

```bash
pip install -r eval/requirements-serving.txt
```

The serving reference version is vLLM 0.19.1, taken from the original environment.
The launcher requires Qwen3.5 support, the multimodal `/tokenize` endpoint,
`--reasoning-parser qwen3`, and hybrid Mamba prefix caching. This release has not
been used to run a new GPU evaluation. Choose a compatible CUDA build and size
`tensor_parallel_size`, `max_model_len`, `max_images`, and `max_num_seqs` for your
hardware. `BACKEND_PYTHON` can select the backend interpreter; `CONFIG_PYTHON`
selects the interpreter for the gateway.

## Configure paths

Run commands from the repository root. Configuration files support recursive
`extends`, `${ENV_VAR}` interpolation, and paths relative to the repository root.
The defaults are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FOCUSVTC_MODEL` | `models/FocusVTC` | Merged local Hugging Face checkpoint |
| `FOCUSVTC_DATA_ROOT` | `datasets` | Rendered pages and benchmark sources |
| `FOCUSVTC_OUTPUT_ROOT` | `outputs` | Predictions, trajectories, scores, and logs |
| `VTCBENCH_ROOT` | `external/VTCBench` | Optional external VTCBench checkout |

Set variables to absolute paths when keeping these files outside the checkout.
For example:

```bash
export FOCUSVTC_MODEL=/path/to/merged/checkpoint
export FOCUSVTC_DATA_ROOT="$(pwd)/datasets"
export FOCUSVTC_OUTPUT_ROOT="$(pwd)/outputs"
```

Customize `eval/tool_agent/configs/_base.json`, or create a local JSON file that
extends a benchmark configuration. Pass the same `--config` to the benchmark
runner that you passed to the server. The backend and gateway bind to loopback
by default; this is a local evaluation service without authentication.

## Ask a question over a document

Render your own text into paired low-resolution and high-resolution pages
using DejaVu Sans:

```bash
python eval/rendering/render_document.py \
  --text-file /path/to/context.txt \
  --font-path /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
  --out-root datasets/demo --dpis 72,144
```

Start the backend and zoom gateway:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval/tool_agent/serve_stack.sh \
  eval/tool_agent/configs/inference.json
```

In another shell with the same environment variables:

```bash
python eval/infer.py \
  --images datasets/demo/dpi_72/page_*.png \
  --question "What evidence in the document answers your question?" \
  --output outputs/demo.json
```

Page order follows the supplied `--images` arguments. The CLI sends local file
URLs, so the client and server must share these files. The gateway maps
`dpi_72/<page>` to `dpi_144/<page>` for a requested crop. Preserve matching
filenames and document layout between DPI trees. A missing paired page becomes
an explicit failed tool observation. For ordinary photographs or unpaired
source images, create a config with `require_high_res: false` and
`high_res_policy: "source_image"`.

For a checkpoint used without the zoom policy, start the vLLM backend directly
and point the same client at it:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "$FOCUSVTC_MODEL" --served-model-name FocusVTC \
  --host 127.0.0.1 --port 18450 \
  --allowed-local-media-path "$FOCUSVTC_DATA_ROOT" \
  --reasoning-parser qwen3

# In another shell:
python eval/infer.py --base-url http://127.0.0.1:18450/v1 \
  --images datasets/demo/dpi_72/page_*.png --question "Summarize the document."
```

## Benchmarks

| Name | Included implementation | Data location under `datasets/` |
| --- | --- | --- |
| `ruler_v1` | Inference from images/text and RULER v1 metrics | `RULER_v1_VTC/dpi_72/index.jsonl` |
| `ruler_v2` | Inference from images/text and RULER v2 metrics | `RULER_v2_VTC/dpi_72/<task>/test.jsonl` |
| `longbench` | Inference from images/text and LongBench metrics | `LongBench_VTC/dpi_72/<dataset>/test.jsonl` |
| `mrcr` | Inference from images, raw-text baseline, and scoring | `MRCR_VTC/dpi_72/<subset>/test.jsonl` |
| `vtcbench` | Launcher for an external VTCBench installation | `VTCBench-Wild/<tier>/` |
| `mmmu` | Launcher using the upstream lmms-eval `mmmu_val` task | Downloaded by lmms-eval |
| `ocrbench` | lmms-eval task adapter and launcher | Downloaded by lmms-eval |

Rendered RULER, LongBench, and MRCR tool evaluations require a corresponding
`dpi_144` tree. Their JSONL `images` entries (`image` in RULER v1) may be absolute paths or paths
relative to the manifest's directory. No benchmark manifests or rendered data
are shipped. The benchmark-specific scripts under
[`eval/rendering/`](../eval/rendering/README.md) create these manifests from
locally supplied source datasets, preserving evaluation questions and answers.
Use those scripts for benchmark inputs; `render_document.py` produces pages
and a manifest for one standalone text document.

The main manifest contracts are:

- RULER v1: `id`, `task`, `question`, `image`, `answer`; optional
  `answer_prefix`, `max_new_tokens` and token metadata.
- RULER v2: `index`, `task`, `question`, `images`, `expected_answer`.
- LongBench: `index`, `dataset`, `question`, `images`, `answers`, `all_classes`,
  `length`.
- MRCR: `index`, `subset`, `question`, `images`, `answer`,
  `random_string_to_prepend`, `length_bin`, `num_pages`, `image_sizes`, and
  associated source metadata. See `MRCR/run_eval.py` for the loader and token
  budget estimates.

For LongBench, RULER v2, and MRCR, put one `<image>` placeholder per page in
`question`, at the position where the page appears in the prompt. RULER v1
appends its question to the page block instead. Scoring and prompt settings
are retained in each generated run configuration.

Example LongBench tool evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash eval/tool_agent/serve_stack.sh \
  eval/tool_agent/configs/longbench.json

# In another shell:
python eval/tool_agent/run_benchmark.py longbench --concurrency 4
```

The same entrypoint accepts `ruler_v1`, `ruler_v2`, `mrcr`, `vtcbench`, `mmmu`,
and `ocrbench`. `--limit N` restricts the samples per task/subset. To use a
custom configuration:

```bash
python eval/tool_agent/run_benchmark.py longbench \
  --config /path/to/longbench.local.json --concurrency 4
```

The standalone `run_eval.py` files also accept a direct backend `--base-url`,
`--results-dir`, and prompt/decoding options for SFT and base-model comparisons.
Run the matching `score.py --results-dir <directory>` to rescore saved predictions.
Raw-text input locations can be overridden with `--text-source` (RULER v1),
`--text-root` (RULER v2 and LongBench), or `--source-root` (MRCR). The MRCR raw-text
baseline additionally needs `pyarrow` and `transformers`, and its tokenizer
must match the chosen checkpoint.

### Optional external evaluators

Install [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) using its
installation instructions for MMMU/OCRBench. Select its interpreter with
`--lmms-python`. OCRBench reuses upstream prompt/scoring functions; no benchmark
implementation or data is vendored. The MMMU validation task is used without
local dataset-path overrides.

For VTCBench, install an external checkout of
[Moenupa/VTCBench](https://github.com/Moenupa/VTCBench) and set `VTCBENCH_ROOT`.
The adapter targets its `examples/run_wild.py` / `examples/collect.py` CLI and
writes a temporary model configuration under your output directory. It runs
Retrieval, Reasoning, and Memory across the four configured length tiers.
The VTCBench evaluator code, benchmark data, caches, and its local experimental
variants are not included. `eval/rendering/render_vtcbench.py` can render
locally supplied `_context` text while preserving `_gt` and other evaluation
fields; see its [instructions](../eval/rendering/README.md#vtcbench-wild).
The launcher defaults to the original `VTCBench-Wild` images. To use rendered
pages, set `data_root` in a custom configuration to
`VTCBench-Wild_VTC/dpi_72` under your data root. The default adapter crops the
source images; it does not reconstruct paired high-resolution pages from
inline image data, even when a rendered `dpi_144` tree exists.
Upstream evaluator changes may require adapting these external launchers.

## Tool policy and output fields

The default policy uses at most 8 zoom calls, 9 turns, 2,048 generated tokens per
turn, and a 10,240-token trajectory budget. MRCR uses a longer trajectory budget.
These are configurable policy limits, not benchmark results. The trajectory
budget counts generated text and returned observations, including crop-image
tokens, through vLLM's multimodal `/tokenize` endpoint. Training uses an 8,192-token initial prompt limit; this evaluation gateway
allows longer initial inputs up to the configured backend context capacity. A backend length stop ends the trajectory;
a model that reaches the tool-call limit is prompted to give a final answer.

The custom Qwen3.5 chat template retains the reasoning from historical tool
turns, matching GRPO's appended-token trajectory. The gateway returns a normal
`choices[0].message.content` answer plus `vtc_*` fields for the per-turn model
outputs, crop trace, stop reason, initial prompt length, and cumulative usage.
`eval/infer.py --output` saves the full response. Benchmark client coverage of
these diagnostic fields differs; LongBench and MRCR preserve detailed traces.

RULER v2 retains the original repository's scoring option that also credits
exact references found in reasoning for selected retrieval tasks. This differs
from final-answer-only scoring; inspect `eval/RULER/ruler_v2_dpi/score.py` when
comparing against an upstream leaderboard. Always keep the prompt, sampling,
DPI, tool budget, and scorer policy together with a reported score.

## Attribution

LongBench configuration and metrics derive from
[THUDM/LongBench](https://github.com/THUDM/LongBench), with its MIT license in
`eval/LongBench/LICENSE`. RULER metrics derive from
[NVIDIA/RULER](https://github.com/NVIDIA/RULER) and
[NVIDIA/NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills); see
`eval/RULER/LICENSE`. The adapted Qwen3.5 template retains its upstream
attribution in `eval/tool_agent/NOTICE`. External benchmark datasets and models
retain their respective licenses and access terms.
