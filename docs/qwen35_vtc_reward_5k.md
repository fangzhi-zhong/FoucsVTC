# Qwen3.5 VTC：证据约束奖励与 5K 修正训练集

2026-09-11。修改的是下一次 GRPO 运行读取的规则奖励，并从已有 50K
`train.parquet` 选择 5K 原始行；本次没有启动训练或评测。

## 奖励定义

实现：[qwen35_vtc_reward.py](../examples/reward_function/qwen35_vtc_reward.py)。

```text
R = 0.8 A + 0.2 F + 1.0 C D Q η − 0.2 P − 0.1 U

η = min(1, E / N)               # N > 0；N = 0 时工具加分为 0
P = max(0, N − E) / max(1, N)   # 超出证据额度的调用比例
U = 无效调用数 / max(1, N)
```

- `A`：最终答案中匹配到的标准答案数 / 标准答案总数。保留原有一对一答案
  匹配规则，补上词边界，防止 `4` 命中 `42`、`no` 命中 `unknown`。
  多答案只命中一部分时仍保留答案部分分，但不给工具加分。
- `C`：存在完整、终止位置正确的 `<answer>...</answer>`，且所有标准答案
  按上述规则匹配成功时为 1，否则为 0。这是规则判定，不是额外模型语义裁判。
- `F`：格式正确为 1，否则为 0。识别 Qwen 的 thinking prefill 和 EOS；
  只检查 assistant 输出，排除 observation 和 user 中重复附加的格式示例。
- `D`：保留原 DPI 系数 `clip((144 − dpi) / 72, 0, 1)²`。
  72/96/144 DPI 对应 1、4/9、0，因此 144 DPI 不额外奖励 zoom。
- `E`：有效且去重的 `(page, bbox)` 标注数量。只取一个 DPI 版本，优先较高 DPI；
  不跨页合并，不将不同位置的证据合并，也不把没有标注的样本伪造为一条证据。
  **现有 50K 没有语义证据 ID；这里计数的是标注区域，跨页片段仍分别计数。**
- `N`：assistant 轨迹中的全部调用尝试，包括无效 JSON/XML、非法参数和截断后
  未闭合的调用。无效调用不能获得 IoU，但仍计入超用分母。

IoU 先限定同一页，再计算每个调用与每个证据框之间的匹配质量：

```text
a_i = crop 面积 / 整页面积
g_i = min(1, 2 × (1 − a_i))
evidence'_j = expand(evidence_j, 20%, min_width=80, min_height=80)
q_ij = IoU(crop_i, evidence'_j) × g_i  # 页码不同则为 0
S = 一对一最大权重匹配的 q_ij 总和
Q = S / min(N, E)                    # N 或 E 为 0 时 Q = 0
```

每个证据框先以中心为基准扩大 20%；宽或高小于 80（归一化坐标）时，扩大到至少
80。扩框后平移到 `[0,1000]` 页面边界内。每个证据框至多匹配一个调用，每个调用至多匹配一个证据框。半页以内不增加面积
折扣，超过半页线性衰减，整页框的工具加分为零。重复 zoom 同一个证据不会累加
该证据的分数。答对时用少于证据数量的调用也可以拿到完整工具加分，不要求把所有
标注逐个查看；额外调用必须提供新的、有质量的证据，才可能维持相同的平均质量。

工具正奖励权重为 **1.0**，答案分为 0.8、格式分为 0.2，因此无扣分时的最高
奖励为 **2.0**。超用与非法调用的扣分不受答对开关控制：答错或轨迹截断也不能
逃避扣分。`E=0` 时无正工具奖励，发生调用扣 0.2 的超用项；这是对缺少定位监督
样本的保守策略，并不说明它们一定不需要工具。

以下例子均为 72 DPI、格式正确、最终完全答对，且准确的小框 IoU=1：

| 证据数 E | 调用情况 | 工具加分 | 工具扣分 | 总奖励 |
| --- | --- | --- | --- | --- |
| 2 | 0 次 | 0 | 0 | 1.00 |
| 2 | 1 次准确定位 | 1.00 | 0 | 2.00 |
| 2 | 2 次，各定位不同证据 | 1.00 | 0 | 2.00 |
| 2 | 2 次，重复同一处 | 0.50 | 0 | 1.50 |
| 2 | 4 次，前两次已覆盖两处证据 | 0.50 | 0.10 | 1.40 |
| 2 | 8 次，只有两处独立证据 | 0.25 | 0.15 | 1.10 |
| 2 | 1 次整页框 | 0 | 0 | 1.00 |
| 0 | 1 次合法调用 | 0 | 0.20 | 0.80 |

缺少 final answer 时答案分与正工具奖励均为 0；不会用 reasoning、工具参数或
observation 中出现的正确答案兜底。新增日志指标包括 `answer_correct`、
`final_answer_present`、`evidence_count`、`matched_evidence`、`iou_reward`、
`call_efficiency`、`excess_call_ratio`、`tool_bonus`、`tool_penalty`。
保留 `best_iou`、`acc_reward` 等既有诊断指标。

## 5K 来源与任务配额

抽样脚本：[select_qwen35_vtc_grpo_5k.py](../examples/data_preprocess/select_qwen35_vtc_grpo_5k.py)。
输出：`/vepfs-mlp2/c20250405/400042/data/VTC/GRPO/train_5k_tool_rebalanced.parquet`。
同名 `.manifest.json` 保存实际子任务配额、原始行索引、长度与证据分布。

| 来源 | 条数 | 新占比 | 原占比 |
| --- | --- | --- | --- |
| Gemini QA | 2,000 | 40% | 60% |
| LongBench SFT | 1,250 | 25% | 10% |
| MRCR | 500 | 10% | 10% |
| RULER v1 | 750 | 15% | 10% |
| RULER v2 | 500 | 10% | 10% |
| 合计 | 5,000 | 100% | 100% |

LongBench 增加 passage counting 和代码任务；MRCR 保留目标定位训练；RULER v1
增加全局计数与频次任务。Gemini 内继续保留短/长多跳、TriviaQA、NarrativeQA
与文档 QA。RULER v2 降低要求复制多篇长文的任务，保留短答案定位、多目标检索
及 QA。这是针对工具策略和终止行为的修正数据，不能补齐缺失的摘要与视觉 few-shot。

抽样约束：

- 只从现有训练集抽取，不读取评测答案或抽取验证集样本；Parquet 中原行字段完整保留。
- 使用当前 checkpoint-7000 的 processor、tokenizer、工具 schema 和训练 prompt
  渲染逻辑，按图片文件头计算视觉 token；初始 prompt ≤8192，标准答案 ≤1024 tokens。
  高清工具图不算进初始 prompt。这个口径与训练的 `image_size` 过滤一致，
  不是旧 `prompt_length_estimate` 字段，也不是实际生成轨迹长度的保证。
- 固定来源与子任务配额，在内部按 DPI、实际输入长度和证据数量分层，无放回抽取。
  每个精确 `source/id` 至多保留一个 DPI 视图；ID 本身含 DPI 的行不做模糊合并。
- 随机种子 `20260911`。不复制图片，不改提示词、答案、工具开关或硬性调用上限。
- 8K 筛选无法解决几十页长上下文的分布外问题。LongBench 的 `long` 只有 23 页
  样本，本修正集不为它扩大 prompt 预算。

### 实际抽样结果

| 指标 | 数量 / 比例 |
| --- | --- |
| 72 / 96 / 144 DPI | 3,532 / 1,060 / 408（70.64% / 21.20% / 8.16%） |
| 实际输入 ≤4K / (4K, 8K] | 3,157 / 1,843（63.14% / 36.86%） |
| 无有效证据框 / 有有效证据框 | 1,590 / 3,410（31.80% / 68.20%） |
| 1 个 / 2 个 / ≥3 个证据框 | 1,192 / 1,331 / 887（23.84% / 26.62% / 17.74%） |
| 计数与频次任务 | 950（19%）：LongBench 600 + RULER v1 350 |
| 代码任务 | 650（13%）：补全 400 + 整页代码 250 |
| 最长实际输入 / 最长标准答案 | 8,191 / 1,000 tokens |
| 页数范围 | 1–22 页；实际视觉 token 数随页面尺寸变化 |
| 覆盖子任务 | 45 个 source × subset 组合 |

全量长度统计中，6,816 条超过 8K 初始输入上限，随后另有 618 条超过 1,024-token
标准答案上限。筛选后在候选池内按固定配额抽取。DPI 比例来自候选容量与分层分配，
没有强行维持原 64% / 24% / 12%。输出已与原始索引对应行逐字段核对一致，
含 Parquet schema metadata；5,000 个行索引和精确 source/id 均不重复。

### 子任务明细

| 来源 | 子任务 | 条数 | 占 5K |
| --- | --- | --- | --- |
| gemini-3.5-flash-30k | `ChatQA-Training-Data/drop` | 150 | 3.00% |
| gemini-3.5-flash-30k | `ChatQA-Training-Data/quoref` | 125 | 2.50% |
| gemini-3.5-flash-30k | `ChatQA-Training-Data/ropes` | 125 | 2.50% |
| gemini-3.5-flash-30k | `ChatQA-Training-Data/tatqa` | 150 | 3.00% |
| gemini-3.5-flash-30k | `ChatQA2-Long-SFT-data/NarrativeQA_131072` | 50 | 1.00% |
| gemini-3.5-flash-30k | `ChatQA2-Long-SFT-data/long_sft` | 350 | 7.00% |
| gemini-3.5-flash-30k | `multihop/2wiki` | 75 | 1.50% |
| gemini-3.5-flash-30k | `multihop/2wiki_long` | 150 | 3.00% |
| gemini-3.5-flash-30k | `multihop/finqa` | 100 | 2.00% |
| gemini-3.5-flash-30k | `multihop/hotpotqa` | 75 | 1.50% |
| gemini-3.5-flash-30k | `multihop/hotpotqa_long` | 200 | 4.00% |
| gemini-3.5-flash-30k | `multihop/musique` | 75 | 1.50% |
| gemini-3.5-flash-30k | `multihop/musique_long` | 150 | 3.00% |
| gemini-3.5-flash-30k | `trivia_qa/rc_json` | 225 | 4.50% |
| LongBench_SFT | `code_complete` | 400 | 8.00% |
| LongBench_SFT | `code_page` | 250 | 5.00% |
| LongBench_SFT | `count_pcnt` | 600 | 12.00% |
| MRCR_SFT | `mrcr_4needle` | 250 | 5.00% |
| MRCR_SFT | `mrcr_8needle` | 250 | 5.00% |
| RULER_v1_SFT | `count_cwe` | 125 | 2.50% |
| RULER_v1_SFT | `count_fwe` | 125 | 2.50% |
| RULER_v1_SFT | `cwe` | 50 | 1.00% |
| RULER_v1_SFT | `fwe` | 50 | 1.00% |
| RULER_v1_SFT | `needle_multikey` | 25 | 0.50% |
| RULER_v1_SFT | `needle_single` | 25 | 0.50% |
| RULER_v1_SFT | `niah_multikey_1` | 25 | 0.50% |
| RULER_v1_SFT | `niah_multikey_2` | 25 | 0.50% |
| RULER_v1_SFT | `niah_multikey_3` | 25 | 0.50% |
| RULER_v1_SFT | `niah_multiquery` | 50 | 1.00% |
| RULER_v1_SFT | `niah_multivalue` | 50 | 1.00% |
| RULER_v1_SFT | `niah_single_1` | 25 | 0.50% |
| RULER_v1_SFT | `niah_single_2` | 25 | 0.50% |
| RULER_v1_SFT | `niah_single_3` | 25 | 0.50% |
| RULER_v1_SFT | `qa_1` | 25 | 0.50% |
| RULER_v1_SFT | `qa_2` | 25 | 0.50% |
| RULER_v1_SFT | `vt` | 50 | 1.00% |
| RULER_v2_SFT | `mk_niah_basic` | 50 | 1.00% |
| RULER_v2_SFT | `mk_niah_easy` | 50 | 1.00% |
| RULER_v2_SFT | `mk_niah_medium` | 50 | 1.00% |
| RULER_v2_SFT | `mk_niah_hard` | 50 | 1.00% |
| RULER_v2_SFT | `mv_niah_basic` | 100 | 2.00% |
| RULER_v2_SFT | `qa_basic` | 50 | 1.00% |
| RULER_v2_SFT | `qa_easy` | 50 | 1.00% |
| RULER_v2_SFT | `qa_medium` | 50 | 1.00% |
| RULER_v2_SFT | `qa_hard` | 50 | 1.00% |

### 复现与使用

复现抽样（输出路径须不存在）：

```bash
/vepfs-mlp2/c20250405/400042/miniconda3/envs/vtc-grpo/bin/python \
  /vepfs-mlp2/c20250405/400042/VTC/train/GRPO/examples/data_preprocess/select_qwen35_vtc_grpo_5k.py
```

下次用 5K 启动现有训练 recipe 时，显式指定数据和新的输出目录：

```bash
VTC_GRPO_TRAIN=/vepfs-mlp2/c20250405/400042/data/VTC/GRPO/train_5k_tool_rebalanced.parquet \
VTC_GRPO_OUTPUT=/vepfs-mlp2/c20250405/400042/data/VTC/GRPO/checkpoints/qwen35_vtc_tool_rebalanced_5k \
WANDB_NAME=qwen35_vtc_tool_rebalanced_5k \
bash /vepfs-mlp2/c20250405/400042/VTC/train/GRPO/examples/agent/qwen3_5_vtc/run_grpo_8gpu_native_1000steps.sh
```

原 recipe 的单轮生成、轨迹预算与工具硬上限仍独立生效；奖励中的 `E` 是软额度，
不会要求环境为每个证据框预留一次调用。奖励变更不会让已经启动的训练进程热更新。

## 验证

运行仅覆盖此次奖励变更的 CPU 回归：最终答案判定、部分答对、重复证据、IoU、
超用比例、整页框、非法/截断调用、DPI、多轮 observation，以及 reward manager
的 prompt 传递。24 项回归全部通过。无需 GPU、模型 rollout 或全量评测。

```bash
cd /vepfs-mlp2/c20250405/400042/VTC/train/GRPO
/vepfs-mlp2/c20250405/400042/miniconda3/envs/vtc-grpo/bin/python \
  -m unittest discover -s tests/agent -p 'test_qwen35_vtc_reward*.py' -v
```
