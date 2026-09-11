# Qwen3.5-VL-9B VTC GRPO（A 路线）

这套 recipe 把一条样本处理成：低清页面 → 模型决定是否调用
`zoom_region(page, bbox_2d)` → 工具从对应 144-DPI 页面裁剪 → 把裁剪图作为
新的视觉 observation → 模型继续回答或再次 zoom。GRPO 的终局 reward 是答案
是否正确；工具 IoU、调用次数与框面积共同决定小幅附加奖惩，只有最终完全答对
才给工具正奖励。2026-09-11 的新公式与 5K 修正训练集见
[证据约束奖励与 5K 数据说明](../../../docs/qwen35_vtc_reward_5k.md)。

## 0. 环境

本机已经建立独立 Python 环境：

```bash
source /vepfs-mlp2/c20250405/400042/VTC/train/GRPO/examples/agent/qwen3_5_vtc/env.sh
```

它是从 `qwen35-vllm` 复制出的 Python 3.12 venv overlay（不是 conda 环境），
所以用上面的 `source` 或绝对路径 `VTC_GRPO_PY`，不要执行
`conda activate vtc-grpo`。当前关键版本是 torch 2.10.0+cu128、Transformers
5.16.1、vLLM 0.19.1、flash-attn 2.8.3。正式 8 卡脚本使用
`attn_implementation=flash_attention_2`；继承环境里的 xformers 是 torch 2.9 ABI，
不能启用，也不要为它降级 torch。

## 1. 准备小数据 shard

下面的命令会生成带 DeepEyes 风格交互 prompt 的新 shard；已有的
`train_8.parquet`/`val_4.parquet` 不会被覆盖，debug 脚本默认使用下面两个新文件。

源文件扩展名虽然是 `.json`，实际是 JSONL。先做 8 条训练、4 条验证：

```bash
cd /vepfs-mlp2/c20250405/400042/VTC/train/GRPO
source examples/agent/qwen3_5_vtc/env.sh
${VTC_GRPO_PY} examples/data_preprocess/prepare_qwen35_vtc_grpo.py \
  --input /vepfs-mlp2/c20250405/400042/data/VTC/SFT/RULER_v2_SFT/train.json \
  --output examples/data/qwen35_vtc/train_8_deepeyes_prompt.parquet \
  --limit 8 --ruler-length 8192
${VTC_GRPO_PY} examples/data_preprocess/prepare_qwen35_vtc_grpo.py \
  --input /vepfs-mlp2/c20250405/400042/data/VTC/SFT/RULER_v2_SFT/train.json \
  --output examples/data/qwen35_vtc/val_4_deepeyes_prompt.parquet \
  --skip 8 --limit 4 --ruler-length 8192
```

转换器只把低清路径放入 `images`，把 144-DPI 路径放入
`high_res_images`；`RLHFDataset` 会把后者放在
`origin_multi_modal_data["high_res_image"]`，工具按选中的页懒加载。最终答案
不进入 prompt，只放在 `reward_model.ground_truth`。

## 2. 不占 GPU 的检查

```bash
${VTC_GRPO_PY} examples/agent/qwen3_5_vtc/check_pipeline.py \
  --model /vepfs-mlp2/c20250405/400042/VTC/train/SFT/output/qwen3_5_9b_vtc_250k_freeze_visual_lr_1e-6_1000steps_merged \
  --parquet examples/data/qwen35_vtc/train_8_deepeyes_prompt.parquet
```

看到 `offline pipeline check: OK` 后再申请空闲 GPU。这个检查会验证工具 schema、
Qwen3.5 的 `mm_token_type_ids`/mRoPE、XML tool call 和高清裁剪，但不会加载 9B
权重。

## 3. 首次 1-step GRPO

```bash
source examples/agent/qwen3_5_vtc/env.sh
CUDA_VISIBLE_DEVICES=0,1 examples/agent/qwen3_5_vtc/run_grpo_debug.sh \
  2>&1 | tee /tmp/qwen35_vtc_grpo_debug.log
```

脚本固定：FSDP actor、vLLM SPMD、TP=2、`rollout.n=4`、batch=4、最多 3 次
zoom tool call，并额外保留 1 轮最终回答（`max_turns=4`）。
`max_response_length=8192`（这是整条 agent 轨迹的预算；每轮由
`single_response_max_tokens=512` 限制，给重试和高清 observation 留出上下文）、
`adv_estimator=grpo`（不创建 critic）、1 个 training step。actor 参数和 Adam
状态在 rollout 阶段 offload 到 CPU，以降低 9B + 24K 序列的显存峰值。Qwen3.5 actor
的非 rmpad forward 只计算最后 `response_length+1` 个 logits，避免
为 16K prompt 额外物化整段 24K×词表 logits；因此首轮保持
`use_remove_padding=false`。启动前必须确认这两张卡没有被评测服务占用；脚本不会
自动杀进程。

## 4. 观察与放大

日志中的 `answer_exact`、`best_iou`、`tool_calls`、`invalid_tool_calls` 是首要
指标。若工具调用格式正确但答案仍错，先检查 crop/observation；若完全没有
`<tool_call>`，提高 SFT 起点的 tool-following 数据或降低采样温度，而不是先
增大 batch。首轮稳定后按顺序放大：`train_8 → 全量 Parquet`、steps、再增加
`max_response_length`/`max_turns`。生产规模时保持
chunked prefill，以便按较小 token 预算处理长输入；关闭 chunked prefill 时才要求
`max_num_batched_tokens >= max_model_len`。保留 `VLLM_ENABLE_V1_MULTIPROCESSING=0`。

## 5. 50K 数据集与 8 卡基线

正式基线的数据构建器和输出位置如下：

```bash
python /vepfs-mlp2/c20250405/400042/VTC/train/GRPO/examples/data_preprocess/build_qwen35_vtc_grpo_50k.py \
  --output-root /vepfs-mlp2/c20250405/400042/data/VTC/GRPO \
  --force
```

它生成 `data/VTC/GRPO/train.parquet`（50,000 条）、`val.parquet`（5,000 条）和
`manifest.json`。样本按来源、DPI 和子任务分层，Parquet 只保存共享盘上的图片路径，
不会复制图片。构建器用 Qwen3.5 的视觉 patch 预算做 16K 上限的保守估计，并把
`prompt_length_estimate`/`length_bucket` 写入 `extra_info`；当前构建结果的估计最大值为
16,204。受指定 DPI 配额和各来源自然长度分布限制，8K–16K 桶实际约占 30%，其余约
70% 不超过 8K；manifest 记录了每层计数。

在一台 8 卡机器上启动基线：

```bash
source /vepfs-mlp2/c20250405/400042/VTC/train/GRPO/examples/agent/qwen3_5_vtc/env.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /vepfs-mlp2/c20250405/400042/VTC/train/GRPO/examples/agent/qwen3_5_vtc/run_grpo_8gpu_baseline.sh
```

首轮显存不足时可以只调小这些环境变量，不需要改脚本：
`VTC_PPO_MICRO=1`（默认）、`VTC_GPU_MEMORY_UTILIZATION=0.60`（默认）或
`VTC_TOTAL_STEPS=1`（仅做一轮连通性调试）。基线使用
`attn_implementation=flash_attention_2`、`use_remove_padding=false`；后者是当前
Qwen3.5 actor 的稳妥设置，避免 rmpad 分支物化完整词表 logits。

## 80GB GPU 上的 micro=2

当前 native 脚本使用 `train_batch=8`、`rollout.n=4`、`ppo_mini=8`，
`VTC_PPO_MICRO=2` 与 `VTC_ROLLOUT_LOGPROB_MICRO=2` 分别控制 PPO 更新和
old-logprob 重算。两者可以独立调整。日志
`qwen3_5_9b_10k_steps100_bs8_rn4_mini8_micro2.log` 的 OOM 出现在 old-logprob
的熵计算阶段，尚未执行 PPO 更新。

本地 DeepEyes recipe 使用 Qwen2.5-VL-7B、`use_remove_padding=true`、PPO micro=4、
log-prob micro=8。Qwen3.5 当前使用 padded 路径，词表有 248320 项，即使实际响应很短，
也会按 `response_length=10240` 计算响应区 logits。micro=2 的一个完整 BF16 张量约
9.47 GiB，FP32 临时张量约 18.95 GiB，后者正是此次 OOM 的分配量。

当前优化保留现有 log-prob 算子和 PPO 梯度路径：Qwen3.5 直接投影需要的响应预测位置，
避免 causal slice 后 reshape 复制整块 logits；old-logprob 的无梯度熵计算按 1024 token
分块，保留熵指标。一个 FP32 分块张量约 0.947 GiB，这是单个临时张量大小，并非总显存。
如果启用非零 `entropy_coeff`，有梯度熵仍走原计算路径，不能据此认为正则项反向也已分块。
这些优化降低计算中的临时开销，micro=2 的整步峰值仍需实际训练确认。

| 环境变量 | 默认值 | 调整用途 |
| --- | --- | --- |
| `VTC_PPO_MICRO` | native / baseline 均为 `2` | PPO 更新的每卡 micro batch |
| `VTC_ROLLOUT_LOGPROB_MICRO` | native `2`，baseline `1` | 独立控制 old-logprob；设为 1 不会降低 PPO micro |
| `VTC_ENTROPY_CHUNK_SIZE` | `1024` | 无梯度熵计算的 token 块大小；0 恢复整块计算 |
| `VTC_MAX_PROMPT_LENGTH` | `8192` | prompt 长度预算 |
| `VTC_MAX_RESPONSE_LENGTH` | `10240` | agent 响应及工具 observation 区域预算；降低会更早截断轨迹 |
| `VTC_MAX_NUM_SEQS` | `32` | vLLM 并发序列上限，现在实际传给引擎 |
| `VTC_MAX_NUM_BATCHED_TOKENS` | `32768` | rollout token 调度上限；当前开启 chunked prefill，可降为 8192/16384，可能降低吞吐 |
| `VTC_ENFORCE_EAGER` | `false` | 设为 true 关闭 CUDA Graph，减少图相关显存，可能降低生成速度 |
| `VTC_GPU_MEMORY_UTILIZATION` | `0.80` | vLLM 的显存预算；可降到 0.65，主要影响 rollout 的 KV cache |

例如保留 PPO micro=2，仅将 old-logprob 的 micro 调为 1：

```bash
VTC_PPO_MICRO=2 VTC_ROLLOUT_LOGPROB_MICRO=1 \
bash /vepfs-mlp2/c20250405/400042/VTC/train/GRPO/examples/agent/qwen3_5_vtc/run_grpo_8gpu_native_1000steps.sh
```

模型总长度由 prompt + response 自动计算，默认是 **18432**，修正了原脚本的 18422。
同时修正了 chunked prefill 的反向检查：开启时可以用小于模型总长度的调度预算，
关闭时才要求 `max_num_batched_tokens >= max_model_len`。调小此预算主要减少 rollout
的单轮 prefill 开销，不会直接改变 PPO actor 的 batch 或熵计算。
降低响应预算可以节省 padded logits 和激活显存，但会改变长轨迹的截断行为；先使用
保留 10240 预算的分块优化，再根据实际样本的长度决定是否缩短。

以下开关当前不能当作 Qwen3.5 的直接解决办法：

- `use_remove_padding=true` / Ulysses SP：现有多模态补丁主要适配 Qwen2/2.5-VL；
  Qwen3.5 还需要混合注意力的样本边界和 `mm_token_type_ids` 同步处理。
- `use_dynamic_bsz=true`：当前多模态 actor 优先走固定分组，log-prob 的动态还原分支
  还会引用未生成的索引。固定分组下，降低 `ppo_max_token_len_per_gpu` 也不会自动拆批。
- `free_cache_engine=true`：当前 vLLM 已在 rollout 退出时执行 `sleep(level=1)` 释放
  GPU 上的权重和 KV cache；该 flag 的旧版路径不能额外解决此次熵计算 OOM。
- `mm_processor_cache_gb` 控制 CPU 预处理缓存；Liger 虽然已开启，当前 GRPO 不传
  labels，仍会生成响应 logits。它们不能消除本次整块熵临时张量。
- SFT checkpoint 名称中的 `freeze_visual` 不会自动冻结 GRPO 视觉塔。进一步冻结可省
  视觉反向开销，但需同时处理 FSDP 包装和训练目标，当前未改变视觉塔的训练策略。

## 8K 多模态 prompt 过滤

当前保留 `VTC_MAX_PROMPT_LENGTH=8192` 和 `data.filter_overlong_prompts=true`。
两个 8 卡脚本默认使用 `data.filter_overlong_prompts_method=image_size`：

- 只读取初始图片的文件头尺寸，不解码、缩放或构造像素张量。
- 按当前 image processor 的最小/最大像素数和 `patch_size × merge_size` 计算缩放、
  对齐后的尺寸。当前 Qwen3.5 为 `16 × 2 = 32`，每张图片视觉 token 数为
  `H' × W' / 1024`；这里的 `H'、W'` 是对齐后的尺寸，不能直接用任意原始尺寸相乘。
- 使用相同聊天模板与工具 schema 渲染文本，再由 tokenizer 计数。总长度为文本
  token 数加上各图片的 `视觉 token 数 - 1`，减掉已计入文本的单个图片占位 token。
  部分 RULER 问题文本较长，因此不再用固定 1024 token 预留替代文本计数。

过滤不调用 processor 的像素预处理，也不使用旧的 `extra_info.prompt_length_estimate`。
仅供后续工具使用的高清图不计入初始 prompt。等于 8192 保留，超过则剔除整条样本；
训练集和验证集都会过滤，原 Parquet 不变。日志显示
`Filtering prompts by image dimensions + text tokens ...`。

两个 8 卡脚本同时启用 `data.skip_overlong_prompts=true`。取样进行实际预处理后，
若展开后的 `input_ids` 或 `raw_prompt_ids` 仍超过 8192，则返回空样本并记录跳过日志，
在 padding/mRoPE 之前丢弃，不再触发原长度报错；文件损坏等其他异常仍正常上报。
训练按 sampler 顺序继续积累有效样本，凑满配置的 batch 后才更新，epoch 末不足一批
的有效样本丢弃。验证直接使用剩余样本，不重复取其他行补数。补批缓存随 DataLoader
状态一起保存，供 checkpoint 恢复。实际训练步数可能少于启动时的预估上限。

图片文件头读取与文本计数默认使用 8 个过滤进程，可通过 `VTC_PROMPT_FILTER_WORKERS`
调整；不产生每个 worker 的图片像素张量。`VTC_PROMPT_FILTER_METHOD=exact` 可切换回
逐条 processor 测长，此时建议同时设 `VTC_PROMPT_FILTER_WORKERS=1`。通用 YAML 默认
`filter_overlong_prompts_method=exact`、`skip_overlong_prompts=false`，其他数据集可按需
开启上述功能；`image_size` 当前用于 Qwen 图片数据，视频数据使用 `exact`。
当前正在运行的旧进程不会自动切换，重新启动后生效。

## 训练中验证

两个 8 卡脚本默认使用 `data/VTC/GRPO/val_500_uniform.parquet` 的固定 500 条子集，
每批 16 条，验证 DataLoader 使用 0 个子进程。完整的 `val.parquet` 仍有 5000 条；
小验证集用于训练中观察趋势，完整验证集用于更全面的比较，尤其是样本较少的子任务。
500 条子集按来源、子任务、DPI 和长度分层抽样，抽样命令和实际分布见
[数据说明](../../../../../../data/VTC/GRPO/README.md)。

验证实际条数由输入文件经过长度过滤后决定；每批大小控制解码图像、预处理张量和
Ray 传输的峰值内存。`data.val_batch_size` 现在实际生效，不再一次加载完整验证集。
若 500 条全部通过过滤，则分成 32 批，最后 4 条按 GPU 数补齐；实际批数以启动日志
为准。生成后去掉补齐项，再统一汇总全部真实样本的指标。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `VTC_GRPO_VAL` | `.../val_500_uniform.parquet` | 验证候选文件，实际条数取决于长度过滤 |
| `VTC_VAL_BATCH` | `16` | 全部 GPU 合计每批的原始样本数，8 卡时每卡 2 条 |
| `VTC_VAL_NUM_WORKERS` | `0` | 验证加载子进程数；先保持 0，避免图像预取占内存 |
| `VTC_TEST_FREQ` | native 脚本 `50`，baseline `100` | 每多少训练步验证；`0` 跳过验证 |

保持每批 16 条，改用完整 5000 条验证集：

```bash
VTC_GRPO_VAL=/vepfs-mlp2/c20250405/400042/data/VTC/GRPO/val.parquet \
VTC_VAL_BATCH=16 \
bash /vepfs-mlp2/c20250405/400042/VTC/train/GRPO/examples/agent/qwen3_5_vtc/run_grpo_8gpu_native_1000steps.sh
```

直接启动 Python 时，对应配置为 `data.val_files`、`data.val_batch_size`、
`data.val_num_workers` 和 `trainer.test_freq`；`val_batch_size=null` 回退到训练 batch 大小。
验证默认 `actor_rollout_ref.rollout.val_kwargs.n=1`、`do_sample=false`，不会按训练的
`rollout.n=4` 生成四份轨迹。若手动增加验证 `n`，每批实际轨迹数也会增加。

多模态验证指标按每条原始样本分组，避免把“文字相同、图片不同”的问题误算成多次采样。
同一步既保存又验证时，先保存 checkpoint，再开始验证，验证异常不会跳过该步保存。

## 格式奖励与生成前缀

总奖励为 `0.8 × acc_reward + 0.2 × format_reward + 1.0 × tool_reward − penalties`。
格式合格得 `1`，违规得 `0`；72 DPI 下答案全部正确、调用工具且格式合格时最高为 `2.0`。

Qwen3.5 的初始 prompt 已预填 `<think>`。奖励管理器向显式声明 `prompt_str` 的自定义
奖励函数传递真实 prompt，格式检查据此计入这一个开标签；不会按 response 的标签差
自动补齐。检查只忽略末尾的 `<|im_end|>` / `<|endoftext|>` 控制标记，缺失或空的
`<answer>`、未配对的思考标签、答案后的额外文字仍会扣分。

直接调用 `compute_score` 检查生成的 continuation 时，应传入对应的 `prompt_str`；
完整自包含答案可以省略。修复不调整答案/工具奖励、DPI 权重或总奖励权重，已运行的
训练进程需要重新加载代码后才会生效；完整公式与调用惩罚见
`train/GRPO/docs/qwen35_vtc_reward_5k.md`。

## 代码入口

- 训练入口：`verl/trainer/main_ppo.py`
- 数据：`verl/utils/dataset/rl_dataset.py`
- agent 循环：`verl/workers/agent/parallel_env.py`
- Qwen3.5 actor loader：`verl/workers/fsdp_workers.py`
- vLLM SPMD：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- 工具：`verl/workers/agent/envs/mm_process_engine/visual_toolbox_qwen3_vtc.py`
- reward：`examples/reward_function/qwen35_vtc_reward.py`
