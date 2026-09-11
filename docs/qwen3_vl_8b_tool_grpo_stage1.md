# Qwen3-VL-8B VTC Tool-GRPO：阶段 1 实施记录

日期：2026-08-05
更新：2026-08-06（tools schema 改为数据加载时统一注入）

状态：**无训练两轮 pipeline 已跑通；尚未进入 GRPO optimizer step。**

## 1. 本阶段完成内容

### 1.1 Qwen3-VL mRoPE

新增：

- `verl/models/transformers/qwen3_vl.py`
- `verl/models/transformers/qwen_vl.py`

实现 Qwen3-VL 单样本 3D mRoPE position ids，并通过 processor family 分发器
区分 Qwen2/2.5-VL 与 Qwen3-VL。修改以下两处，防止 Qwen3-VL 因类名判断失败
退化为普通 1D position ids：

- `verl/utils/dataset/rl_dataset.py`
- `verl/workers/agent/parallel_env.py`

### 1.2 Qwen3-VL 原生 tools schema

`RLHFDataset` 支持在数据加载时统一注入全局 schema，VTC 的每条 Parquet 样本
不再重复存整份 tools JSON。

相关配置为：

- `data.tools_schema_path`：全局 JSON schema 列表，dataset 初始化时只读取一次。
- `data.tools_enabled_key`：可选逐样本布尔开关，默认列名 `enable_tools`。
- `data.tools_key`：保留的可选逐样本 schema 覆盖，默认列名 `tools`。

长度过滤和实际 processor/tokenizer 统一调用 `_resolve_tools`，保证两处传给
`apply_chat_template` 的 schema 完全相同。最终生效规则按以下优先级执行：

1. `enable_tools: false`：该样本不注入任何工具。
2. 样本含非空 `tools`：使用逐样本 schema 覆盖全局 schema。
3. 其余样本：使用 `tools_schema_path` 加载的全局 schema。

因此 VTC 的常规 tool-GRPO 样本只需要这些业务字段：

```text
prompt
images
env_name
ground_truth / reward 所需字段
evidence_locations
enable_tools（可选；缺省即启用全局 schema）
```

配置示例见 `examples/agent/qwen3_vl_vtc_tool/data_tools_global.yaml`。

需要模型学习“看得清时不调用”的 clear/no-tool 对照样本仍应保持 schema 启用，
让 GRPO reward 约束是否调用。`enable_tools: false` 只用于完全不向模型暴露工具的
非 agent 样本；该开关只控制 chat template，执行器仍由 `env_name` 控制，所以此类
样本也应使用空的 `env_name`。

这会使用 checkpoint 自带的 Qwen3-VL 格式：

```text
<tools>...</tools>
<tool_call>...</tool_call>
<tool_response>image + text</tool_response>
```

### 1.3 VTC zoom 环境

新增：

- `verl/workers/agent/envs/mm_process_engine/vtc_zoom.py`
- `verl/workers/agent/envs/mm_process_engine/visual_toolbox_qwen3_vtc.py`

`vtc_zoom.py` 是不依赖 verl 的纯协议实现，供正式环境和 standalone smoke
共同使用。`Qwen3VLVTCZoomTool` 注册名为 `qwen3_vl_vtc_zoom`。

实现行为：

- 工具名严格为 `zoom_region`。
- 页码严格 1-based。
- bbox 严格为 `[0, 1000]` 范围内的四个数。
- 从原始页面列表选择页面并裁剪。
- 默认加入 5% context padding。
- 阶段 1 最多成功调用一次。
- 返回 Qwen3-VL 原生 tool role 图片 observation。

### 1.4 Standalone vLLM smoke

新增：

- `examples/agent/qwen3_vl_vtc_tool/smoke_rollout.py`

该脚本使用 vLLM 0.11.2 的公开 `LLM.chat` 接口跑两轮，但复用与 verl 环境相同
的 `vtc_zoom.py`。它用于隔离验证模型、模板、tool parser、crop 和第二张图片
observation，不声称已经验证 FSDP/vLLM 混合训练。

Triton cache 默认写入：

```text
/vepfs-mlp2/c20250405/400042/.tmp/triton/qwen3_vl_tool_smoke
```

原因是主机 `/root/.triton/cache` 所在分区已满。脚本不会删除或覆盖原有 cache。

### 1.5 测试

新增：

- `tests/agent/test_vtc_zoom.py`
- `tests/models/test_qwen3_vl_rope.py`

## 2. 验证结果

### 2.1 静态与单元测试

```text
Python py_compile                         PASS
zoom protocol tests                     4/4 PASS
RLHFDataset global tools injection       5/5 PASS
Qwen3-VL mRoPE vs Transformers reference 1/1 PASS
git diff --check                         PASS
```

mRoPE 测试直接调用 Transformers 4.57.6 的
`Qwen3VLModel.get_rope_index` 作为 reference，与本地 helper 逐值比较。

tools 注入测试同时校验全局默认、显式禁用、逐样本覆盖、开关类型和 JSON schema
与 `vtc_zoom.TOOL_SCHEMA` 的逐值一致性。

### 2.2 原生 chat template

`--template-only` 结果包含：

- tools schema：是
- assistant tool call：是
- 原页面视觉占位符：1 个
- tool response crop 视觉占位符：1 个
- 第二轮 assistant generation prompt：是

### 2.3 verl observation 集成

不加载 8B 权重，直接通过 `ToolBase.create` 和 `execute_tool_call` 执行 GT crop：

```text
registered tool       qwen3_vl_vtc_zoom
observation keys      multi_modal_data, multi_modal_inputs,
                      prompt_token_ids_model, prompt_token_ids_vllm
model observation     105 tokens
vLLM placeholder      32 tokens
image_grid_thw        [[1, 4, 74]]
done                   False
crop pixels            [0, 86, 581, 98]
crop size              581 x 12
```

这证明 crop 已被 processor 转换为新的 `pixel_values + image_grid_thw`，而不只是
在文本中写了一个假的 `<image>`。

### 2.4 实际 Qwen3-VL-8B 两轮 rollout

环境：GPU 0，A100 80GB，vLLM 0.11.2，Transformers 4.57.6。

第一次运行：模型权重成功加载，随后因 `/root/.triton/cache` 无空间失败。将
Triton cache 改到工作区 `.tmp` 后重跑成功。

模型第一轮输出：

```json
{
  "name": "zoom_region",
  "arguments": {
    "page": 1,
    "bbox_2d": [100, 117, 880, 312]
  }
}
```

工具执行结果：

```text
source size          589 x 126
pixel bbox           [35, 13, 542, 41]
crop size            507 x 28
first prompt tokens  420
first output tokens  47
final prompt tokens  565
final output tokens  23
```

第二轮最终答案：

```text
The components making up the total Compensation of Key Management Personnel
are short-term employee benefits and post-employment benefits.
```

## 3. 结果解释

### Pipeline 结论

阶段 1 的机械闭环成立：

```text
Qwen3-VL 原图
  -> 合法 tool call
  -> 归一化 bbox 转像素并 crop
  -> crop 作为新的视觉 observation
  -> Qwen3-VL 第二轮回答
```

第二轮 prompt 从 420 增至 565 tokens，同时 verl processor smoke 得到了新的
`pixel_values` 和 `image_grid_thw`，因此 crop 确实进入了视觉上下文。

### 行为质量结论

本轮定位和答案质量不合格：

- GT bbox：`[20, 690, 940, 770]`
- 模型 bbox：`[100, 117, 880, 312]`
- 页码正确，但模型框没有命中 GT evidence。
- GT 答案还有 `Share-based payments - equity-settled`，模型最终答案漏掉该项。

因此该结果只能证明 pipeline 可运行，不能证明 base model 已具备目标定位能力。
这正是后续数据筛选、reward 和 GRPO 要优化的部分。

## 4. 尚未解决的兼容门槛

当前环境被拆成两套：

```text
qwen3vl-vllm: Qwen3-VL + vLLM 0.11.2，可运行实际 rollout，缺完整 verl 依赖
lmms-eval:    可导入 verl 和执行 processor/tool smoke，没有 vLLM
```

此外，旧 GRPO fork 的 `setup.py` 限制 `vllm<=0.8.3`，内部 SPMD rollout 和
FSDP-vLLM sharding manager 也依赖旧 vLLM API。Qwen3-VL 训练不能简单降级 vLLM。

所以本阶段没有宣称以下内容已通过：

- 同一 Python 环境中的 FSDP actor + vLLM rollout。
- actor 权重热同步到 vLLM 0.11.2。
- GRPO backward/optimizer step。

这些属于下一阶段的首要任务。

## 5. 修改留痕

| 文件 | 变更 |
|---|---|
| `docs/qwen3_vl_8b_tool_grpo_stage0.md` | 冻结模型、tool、坐标和阶段边界 |
| `docs/qwen3_vl_8b_tool_grpo_stage1.md` | 本实施与验证记录 |
| `verl/models/transformers/qwen3_vl.py` | Qwen3-VL 单样本 mRoPE helper |
| `verl/models/transformers/qwen_vl.py` | Qwen-VL processor family 分发 |
| `verl/utils/dataset/rl_dataset.py` | 全局/逐样本 tools 解析和 Qwen3-VL mRoPE |
| `verl/trainer/config/ppo_trainer.yaml` | 增加全局 schema、开关和兼容覆盖配置 |
| `verl/workers/agent/parallel_env.py` | agent trajectory 使用 Qwen3-VL mRoPE |
| `verl/workers/agent/envs/mm_process_engine/vtc_zoom.py` | 共享 zoom 协议和裁剪实现 |
| `verl/workers/agent/envs/mm_process_engine/visual_toolbox_qwen3_vtc.py` | verl tool wrapper |
| `verl/workers/agent/__init__.py` | 注册新工具环境 |
| `examples/agent/qwen3_vl_vtc_tool/smoke_rollout.py` | 实际两轮 vLLM smoke |
| `examples/agent/qwen3_vl_vtc_tool/zoom_region_tools.json` | 全局 zoom schema 配置文件 |
| `examples/agent/qwen3_vl_vtc_tool/data_tools_global.yaml` | data 配置覆盖示例 |
| `tests/agent/test_vtc_zoom.py` | zoom 协议测试 |
| `tests/agent/test_rl_dataset_tools.py` | 全局注入、禁用、覆盖和 schema 一致性测试 |
| `tests/models/test_qwen3_vl_rope.py` | mRoPE reference 对照测试 |

本次修改未触碰 SFT 目录、VTC_REL 原始数据或已有 `visual_toolbox_v2.py`。

补丁工具说明：主机禁用了内置 `apply_patch` 处理已有文件所需的 user namespace；
新增文件仍由 `apply_patch` 创建，已有文件使用统一 diff 的 `patch` 命令完成。
自动产生的 `.orig/.rej` 临时文件已经删除，不作为修改留痕。

## 6. 下一阶段入口

1. 建立同时满足 Transformers 4.57+、vLLM 0.11.2 和 verl 训练依赖的隔离环境。
2. 对旧 `vllm_rollout_spmd.py` 与 `fsdp_vllm.py` 做 0.11.2 API 迁移。
3. 验证一条 batch 的 `action_mask`、log-prob 和 FSDP -> vLLM 权重热同步。
4. 生成最小 Parquet：`prompt + images + env_name`，schema 由 dataset 全局注入。
5. 最后再进入 1～10 个 GRPO optimizer steps。
