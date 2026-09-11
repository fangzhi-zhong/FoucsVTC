# Qwen3.5-VL VTC GRPO 复现指南

本目录包含 Qwen3.5-VL-9B 的视觉工具调用（VTC）GRPO recipe。训练轨迹可以多次
调用 `zoom_region(page, bbox_2d)`，工具从高 DPI 页面裁剪证据图，再把裁剪结果作为
新的视觉 observation。最终奖励由答案正确性、输出格式、工具调用质量和调用惩罚组成。

文档中的命令都从 `train/GRPO` 执行。请先固定本次代码版本，之后不要在训练过程中
修改 checkout：

```bash
cd /path/to/VTC/train/GRPO
git rev-parse HEAD                 # 记录到实验卡片
git status --short                 # 应为空，或只包含你明确的本地改动
```

## 机器和软件要求

已验证的组合如下；不同 CUDA/驱动可以使用对应的官方 wheel，但应保持这些核心版本
一致，否则 vLLM、FlashAttention 和 Qwen3.5 的多模态实现可能不兼容。

| 项目 | 已验证值 |
| --- | --- |
| Python | 3.12 |
| GPU | NVIDIA，正式配置为单机 8×80 GB；2×80 GB 可跑 smoke |
| PyTorch | 2.10.0（CUDA 12.8 wheel） |
| Transformers | 5.16.1 |
| vLLM | 0.19.1 |
| flash-attn | 2.8.3 |
| verl | 本目录代码，版本文件为 `verl/version/version` |

建议在目标机器建立独立环境，并用与驱动匹配的 CUDA wheel：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# 下面是已验证的版本；CUDA wheel 的 index-url 按机器改为官方或内部镜像。
python -m pip install \
  'torch==2.10.0' \
  'transformers==5.16.1' \
  'vllm==0.19.1' \
  'flash-attn==2.8.3' \
  'ray[default]>=2.10' 'hydra-core' 'datasets' 'pyarrow>=15' \
  'accelerate' 'numpy' 'pandas' 'peft' 'scipy' 'torchdata' \
  'pybind11' 'pylatexenc' 'Pillow' 'wandb' 'dill' 'codetiming' \
  'tensordict<=0.6.2' 'liger-kernel'

# 只安装本 checkout，避免 setup.py 中面向旧 vLLM 的可选依赖覆盖上面的版本。
python -m pip install -e . --no-deps
```

验证环境和当前 checkout：

```bash
export VTC_GRPO_ROOT="$PWD"
export VTC_GRPO_PY="$PWD/.venv/bin/python"
source examples/agent/qwen3_5_vtc/env.sh
```

该命令会打印 torch、Transformers 和 vLLM 版本，并设置
`VLLM_ENABLE_V1_MULTIPROCESSING=0`。正式脚本使用 `flash_attention_2` 和
`use_remove_padding=false`；不要在 Qwen3.5 配置中直接打开 Qwen2/2.5-VL 的 Ulysses
或 remove-padding 路径。

## 模型、图片和工具 schema

训练脚本需要一个已经合并的 HuggingFace 格式 Qwen3.5-VL checkpoint。目录至少应有
`config.json`、tokenizer 文件和模型权重；SFT checkpoint 不在本仓库中。设置路径：

```bash
export VTC_GRPO_MODEL=/path/to/qwen3_5_9b_sft_merged
```

数据 Parquet 只保存图片路径，不包含图片像素。因此所有训练节点必须能访问相同的低
清页面和高 DPI 页面；如果路径不同，请在目标机器重新生成 Parquet，而不要直接复制
本机的 Parquet。工具定义在：

```bash
export VTC_GRPO_TOOLS="$VTC_GRPO_ROOT/examples/agent/qwen3_vl_vtc_tool/zoom_region_tools.json"
```

仓库内的五个小 Parquet fixture 也保留了生成机器的绝对图片路径，主要用于检查 schema。
只有在这些路径已挂载时它们才能直接用于 smoke；通常应按下节命令在目标机器重新生成
8 条数据。50K 构建器同样不会改写 JSONL 中的图片路径，需保持源数据的绝对路径可见，
或先把 JSONL 中的路径批量改成目标机器的路径。

## 准备数据

### 先跑 8 条连通性数据

`prepare_qwen35_vtc_grpo.py` 接受 JSONL（文件扩展名可以是 `.json`），把答案放进
`reward_model.ground_truth`，把低清图放入 `images`，高 DPI 图放入 `high_res_images`：

```bash
python examples/data_preprocess/prepare_qwen35_vtc_grpo.py \
  --input /path/to/RULER_v2_SFT/train.json \
  --output examples/data/qwen35_vtc/train_8_deepeyes_prompt.parquet \
  --limit 8 --ruler-length 8192
python examples/data_preprocess/prepare_qwen35_vtc_grpo.py \
  --input /path/to/RULER_v2_SFT/train.json \
  --output examples/data/qwen35_vtc/val_4_deepeyes_prompt.parquet \
  --skip 8 --limit 4 --ruler-length 8192
```

### 50K 正式数据

构建器按来源、DPI 和子任务分层抽样，固定默认 seed `20260904`，输出 50,000 条训练
样本、5,000 条验证样本和 `manifest.json`。源目录需要包含
`gemini-3.5-flash-30k`、`LongBench_SFT`、`MRCR_SFT`、`RULER_v1_SFT`、
`RULER_v2_SFT` 五个子目录及其 `train*.json` 文件；具体配额见脚本顶部的
`source_specs()`。

```bash
python examples/data_preprocess/build_qwen35_vtc_grpo_50k.py \
  --source-root /path/to/data/VTC/SFT \
  --output-root /path/to/data/VTC/GRPO \
  --seed 20260904
```

若输出已存在，需显式加 `--force`。把生成目录设置给启动脚本：

```bash
export VTC_GRPO_DATA_ROOT=/path/to/data/VTC/GRPO
export VTC_GRPO_TRAIN="$VTC_GRPO_DATA_ROOT/train.parquet"
export VTC_GRPO_VAL="$VTC_GRPO_DATA_ROOT/val.parquet"
```

## 运行顺序

先做不占 GPU 的 schema、工具和图片路径检查：

```bash
python examples/agent/qwen3_5_vtc/check_pipeline.py \
  --model "$VTC_GRPO_MODEL" \
  --parquet examples/data/qwen35_vtc/train_8_deepeyes_prompt.parquet
```

然后用 2 卡跑 1 step，确认数据加载、vLLM、工具调用、奖励和 FSDP 更新全部连通：

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline \
  bash examples/agent/qwen3_5_vtc/run_grpo_2gpu_smoke.sh \
  2>&1 | tee /tmp/qwen35_vtc_smoke.log
```

正式单机 8 卡基线：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=offline \
  VTC_GRPO_VAL="$VTC_GRPO_DATA_ROOT/val.parquet" \
  bash examples/agent/qwen3_5_vtc/run_grpo_8gpu_baseline.sh \
  2>&1 | tee /tmp/qwen35_vtc_8gpu.log
```

上面的命令显式使用构建器生成的 `val.parquet`。如果另外生成了分层的
`val_500_uniform.parquet`，可以把 `VTC_GRPO_VAL` 改成该文件。基线使用 8 次 rollout、
训练 batch 32、PPO micro batch 2、最大 prompt 8192、最大 response 10240。没有 W&B
凭证时必须设置 `WANDB_MODE=offline`；在线记录则先在目标环境执行 `wandb login`。

如果目标机器已有脚本约定的 `train_10k_uniform.parquet`，1000-step 包装脚本可直接使用；
对刚生成的 50K 数据，应显式覆盖输入文件：

```bash
VTC_GRPO_TRAIN="$VTC_GRPO_DATA_ROOT/train.parquet" \
VTC_GRPO_VAL="$VTC_GRPO_DATA_ROOT/val.parquet" \
VTC_TOTAL_STEPS=1000 WANDB_MODE=offline \
  bash examples/agent/qwen3_5_vtc/run_grpo_8gpu_native_1000steps.sh
```

脚本中的所有本机路径都可由环境变量覆盖，最常用的是
`VTC_GRPO_ROOT`、`VTC_GRPO_PY`、`VTC_GRPO_MODEL`、`VTC_GRPO_DATA_ROOT`、
`VTC_GRPO_TRAIN`、`VTC_GRPO_VAL`、`VTC_GRPO_OUTPUT` 和 `VTC_GRPO_TOOLS`。

## 显存和多机调整

显存不足时，按这个顺序降低 rollout 压力：

```bash
VTC_PPO_MICRO=1 VTC_ROLLOUT_LOGPROB_MICRO=1 \
VTC_GPU_MEMORY_UTILIZATION=0.65 VTC_MAX_NUM_BATCHED_TOKENS=16384 \
  bash examples/agent/qwen3_5_vtc/run_grpo_8gpu_baseline.sh
```

`VTC_PPO_MICRO` 和 `VTC_ROLLOUT_LOGPROB_MICRO` 独立生效；降低
`VTC_MAX_RESPONSE_LENGTH` 会改变长工具轨迹的截断行为。多机脚本
`run_grpo_4nodes_8gpu_baseline.sh` 已将 `trainer.nnodes=4`，但仍需要集群管理员提供
Ray 集群启动、节点间网络和 `CUDA_VISIBLE_DEVICES` 配置；先在单机 8 卡验证后再接入
多机调度器。

## 复现实验记录

每次运行至少保存以下信息：`git rev-parse HEAD`、上述核心包版本、完整的模型
路径或模型 revision、数据 `manifest.json`、启动脚本及其环境变量、GPU 型号、日志和
W&B run id。数据抽样和默认配置是确定的，但不同 GPU、驱动、FlashAttention 或分布式
通信顺序仍可能造成非 bitwise 的差异；以验证集 `answer_exact`、`best_iou`、
`tool_calls` 和 `invalid_tool_calls` 的趋势比较结果。

更多奖励公式、5K 修正数据和长度过滤说明见 `docs/qwen35_vtc_reward_5k.md` 以及
`examples/agent/qwen3_5_vtc/README.md`。
