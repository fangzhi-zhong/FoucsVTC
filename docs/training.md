# Training FocusVTC

The release provides a Qwen3.5 supervised fine-tuning stage and an agent GRPO
stage. The configurations are portable starting points from the local codebase;
training data, checkpoints, and experimental outputs are not included.
All commands below run from the repository root.

## Environments

Use separate Python 3.12 environments for SFT and GRPO. NVIDIA GPUs and a
CUDA-compatible PyTorch build are required. The original GRPO environment
records PyTorch 2.10.0, Transformers 5.16.1, vLLM 0.19.1, and FlashAttention 2.8.3.
Compiled kernels must match the installed PyTorch and CUDA toolchain.

```bash
python3.12 -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install 'torch==2.10.0' 'torchvision==0.25.0'
python -m pip install --no-build-isolation -r train/SFT/requirements.txt
```

Use the appropriate official CUDA wheel index when the default PyTorch wheel
is unsuitable for your machine. Install GRPO into another environment:

```bash
python3.12 -m venv .venv-grpo
source .venv-grpo/bin/activate
python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install 'torch==2.10.0' 'torchvision==0.25.0'
python -m pip install --no-build-isolation -r train/GRPO/requirements.txt
```

These requirements describe the supplied code; a fresh GPU environment and
full training run have not been validated as part of packaging this release.

## Supervised fine-tuning

Prepare the public [REL-CoT dataset](https://www.modelscope.cn/datasets/zhongfangzhi/REL-CoT)
with the [`data/` tools](../data/README.md). The release supplies original text,
72 DPI images, and existing supervision. The renderer creates seven DPI views
(48, 60, 72, 84, 96, 120, and 144), all using DejaVu Sans, and writes SFT
manifests under `datasets/REL-CoT_SFT/`. It remaps evidence and page references
to the new layout while retaining the existing reasoning and answers; it does
not call a model to create new supervision. The data guide covers download,
unpacking, font selection for reading older source pages, and resuming renders.

Place the base Hugging Face model, tokenizer, and processor assets under
`models/Qwen3.5-9B`, or set `FOCUSVTC_MODEL`. Prepare a JSONL manifest containing
`image` (one path or a list of paths) and `conversations` (an inline list or a
path to a JSON conversation). Conversation records use `from`/`value` with
`human` and `gpt` roles. Page placeholders must match the supplied images.
The thinking recipe expects supervised assistant targets containing the
appropriate `<think>...</think>` and answer text.

```bash
source .venv-sft/bin/activate
FOCUSVTC_MODEL="$PWD/models/Qwen3.5-9B" \
FOCUSVTC_SFT_DATASET="$PWD/datasets/REL-CoT_SFT/train.jsonl" \
FOCUSVTC_SFT_ASSETS="$PWD/datasets/REL-CoT_SFT" \
NGPUS=8 bash train/SFT/run.sh
```

Point `FOCUSVTC_SFT_DATASET` and `FOCUSVTC_SFT_ASSETS` at your existing
training manifest and its asset root. Absolute asset paths are also accepted.
The combined REL-CoT manifest contains all selected DPI variants; use a
`train_<dpi>dpi.jsonl` manifest to train on one DPI. Keep the downloaded REL-CoT
directory available because prepared records retain absolute source-text paths.
When creating training and validation sets, split by `metadata.base_id` so all
renderings of one source document stay in the same split.

Edit [`qwen3_5.yaml`](../train/SFT/configs/qwen3_5.yaml), or pass
`CONFIG=/absolute/path/to/config.yaml`, to change hyperparameters. The recipe
freezes `model.visual`, enables random grounding prompts, packs to 32K tokens,
and uses FSDP2. Keep `use_rmpad: false` and `sp_ulysses_degree: 1` for this
Qwen3.5 hybrid-attention implementation. Packing provides both mRoPE positions
and attention/DeltaNet/convolution sequence boundaries.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FOCUSVTC_MODEL` | `models/Qwen3.5-9B` | Base model directory |
| `FOCUSVTC_SFT_DATASET` | `datasets/sft/train.jsonl` | Manifest |
| `FOCUSVTC_SFT_ASSETS` | `datasets/sft` | Root for relative assets |
| `FOCUSVTC_SFT_OUTPUT` | `outputs/sft` | Checkpoints and training metadata |
| `NGPUS` | `8` | GPUs per node |
| `NNODES`, `NODE_RANK` | `1`, `0` | Distributed launch topology |
| `MASTER_ADDR`, `MASTER_PORT` | `127.0.0.1`, `29500` | Rendezvous |

The trainer resumes from existing checkpoint directories in the output path.
Use a new output directory for an independent run. SFT checkpoints are sharded;
export the chosen checkpoint before using it for GRPO or serving:

```bash
python train/SFT/merge_fsdp.py \
  --input_dir outputs/sft --type fsdp2 \
  --output_dir models/focusvtc-sft
```

`--step` selects a saved step; omission selects the latest checkpoint. The
export utility reads the local FSDP2 rank shards and requires sufficient host
memory for the reconstructed model.

## GRPO data

REL-CoT outputs include `high_res_images` when the rendering selected 144 DPI,
providing page pairs for later GRPO preparation. The published REL-CoT manifest
does not supply the `gold` reference-answer field required by the converter.
Provide independent reference answers before GRPO conversion; the renderer
does not treat the supervised assistant response as a reference answer.

The converter accepts VTC/RULER JSONL manifests with explicit low/high-resolution
page pairs, conversations, and reference answers in `metadata.gold` (or `gold`).
Supply source reference answers and explicit high-resolution paths in the
manifest. A manifest containing only `image` is sufficient for SFT, but must
be supplemented with high-resolution pages before GRPO conversion.
It removes the final supervised answer from the prompt and stores it only in
reward metadata. Evidence annotations use one-based pages and boxes normalized
to `[0, 1000]`.

```bash
source .venv-grpo/bin/activate
python train/GRPO/examples/data_preprocess/prepare_qwen35_vtc_grpo.py \
  --input datasets/sft/train.jsonl \
  --output datasets/grpo/train.parquet --limit 0
python train/GRPO/examples/data_preprocess/prepare_qwen35_vtc_grpo.py \
  --input datasets/sft/val.jsonl \
  --output datasets/grpo/val.parquet --limit 0
```

Use disjoint training and validation source manifests. The converter resolves
relative image and conversation paths from the source manifest directory.
The converter accepts `high_res_images` (also `images_high_res`,
`images_144dpi`, `image_dpi144`, `images_96dpi`, or `image_dpi96`) and never
substitutes the low-resolution `image` field for missing high-resolution pages.
Provide both lists in the same page order. It reads image headers to check equal
page counts, distinct paths, higher resolution, matching page geometry, and DPI
ratios when explicitly supplied. Missing or mismatched pairs fail conversion.
The highest-resolution input is an explicit exception: with both
`low_res_dpi: 144` and `high_res_dpi: 144`, matching pages may use the same image
path. This preserves the native 144-DPI training case, whose DPI tool bonus is
zero. The exception does not apply to lower-DPI inputs or omitted DPI metadata.
The Parquet files reference local images and do not embed their pixels; all
worker processes must be able to read those paths. Both `from`/`value` and
`role`/`content` conversations are supported, and the last assistant target is
removed from either schema before constructing the rollout prompt.

The main columns are `prompt`, `images`, `high_res_images`, `env_name`,
`enable_tools`, `reward_model`, and `extra_info`. Only low-resolution images
enter the initial prompt. `zoom_region(page, bbox_2d)` lazily loads the matching
high-resolution page and returns a crop as the next visual observation.

## Agent GRPO

```bash
source .venv-grpo/bin/activate
VTC_GRPO_MODEL="$PWD/models/focusvtc-sft" \
VTC_GRPO_DATA_ROOT="$PWD/datasets/grpo" \
NGPUS=8 bash train/GRPO/run.sh
```

The launcher uses local Ray workers, FSDP, and the modified vLLM SPMD backend.
It sets `VLLM_ENABLE_V1_MULTIPROCESSING=0`, keeps
`use_remove_padding=false`, and uses fixed-size microbatches. Batch and sequence
budgets must fit the target hardware; the reference recipe targets a node with
eight large-memory GPUs and is not a memory-fit guarantee for other hardware.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VTC_GRPO_MODEL` | `models/focusvtc-sft` | Merged SFT model |
| `VTC_GRPO_TRAIN`, `VTC_GRPO_VAL` | `datasets/grpo/{train,val}.parquet` | Input shards |
| `VTC_GRPO_OUTPUT` | `outputs/grpo` | Training output |
| `VTC_TRAIN_BATCH`, `VTC_PPO_MINI` | `8`, `8` | Prompt batch and PPO minibatch |
| `VTC_ROLLOUT_N` | `4` | Sampled trajectories per prompt |
| `VTC_PPO_MICRO` | `1` | PPO microbatch per GPU |
| `VTC_ROLLOUT_LOGPROB_MICRO` | `1` | Log-probability microbatch per GPU |
| `VTC_MAX_PROMPT_LENGTH` | `8192` | Initial prompt budget |
| `VTC_MAX_RESPONSE_LENGTH` | `10240` | Continuation and tool-observation budget |
| `VTC_TOTAL_EPOCHS`, `VTC_TOTAL_STEPS` | `1`, `0` | Epochs; nonzero steps impose a cap |
| `VTC_SAVE_FREQ`, `VTC_TEST_FREQ` | `50`, `50` | Save and validation intervals |

Additional Hydra overrides may be passed after `run.sh`. For example:

```bash
bash train/GRPO/run.sh actor_rollout_ref.actor.optim.lr=5e-7
```

The default interaction allows eight tool calls and one final answer turn.
The custom reward combines answer accuracy, structural format, and tool quality.
The tool bonus is gated on a fully correct answer and depends on evidence IoU,
crop area, DPI, and redundant calls. See
[`qwen35_vtc_reward.py`](../train/GRPO/examples/reward_function/qwen35_vtc_reward.py)
for the exact implementation. Empty/missing evidence annotations cannot produce
an evidence-overlap bonus.

## Logging and exports

Both launchers disable external tracking by default. For GRPO, opt into W&B
with `WANDB_MODE=online VTC_LOGGERS="['console','wandb']"`; authenticate in your
own environment. For SFT, also set `trainer_args.report_to: [wandb]` in your
chosen YAML. No API keys or account names are bundled.

GRPO saves rank-local actor checkpoints. Export one with the evaluation helper:

```bash
python eval/tool_agent/merge_fsdp_checkpoint.py \
  --actor outputs/grpo/global_step_50/actor \
  --base-model models/focusvtc-sft \
  --target models/focusvtc-grpo
```

This helper supports the one-dimensional, dimension-zero DTensor layout used
by the included recipe. The target must be empty. All weights and outputs stay
under the ignored `models/` and `outputs/` directories.
