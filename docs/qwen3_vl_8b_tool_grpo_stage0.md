# Qwen3-VL-8B VTC Tool-GRPO：阶段 0 设计冻结

状态：冻结（2026-08-05；tools 注入方式补充于 2026-08-06）

## 1. 目标

先用 `Qwen3-VL-8B-Instruct` 跑通以下无训练闭环，再进入 GRPO 更新：

```text
页面图片 + 问题
  -> Qwen3-VL 生成 tool call
  -> agent 环境解析并裁剪原页面
  -> crop 作为新的图片 observation 返回
  -> Qwen3-VL 生成最终答案
```

阶段 1 只验证模型、模板、工具和多模态 observation 的连接，不验证 GRPO
backward、FSDP 到 vLLM 的权重同步或最终策略质量。

## 2. 框架决策

- 训练框架：仓库现有 DeepEyes/verl fork。
- 训练后端：FSDP。
- rollout 后端：vLLM。
- 算法：GRPO，不增加工具轨迹 SFT。
- 基座：`/vepfs-mlp2/c20250405/400042/models/Qwen3-VL-8B-Instruct`。
- 不引入 TRL 或第二套 agent 框架，避免同时维护两条动态图片 rollout 链路。

选择现有 verl 的原因是它已经具备多轮 `agent_rollout_loop`、动态多模态
observation、action mask 和并行环境接口。

## 3. 第一版交互协议

### 3.1 工具 schema

唯一工具名为 `zoom_region`：

```json
{
  "type": "function",
  "function": {
    "name": "zoom_region",
    "description": "Crop a potentially unreadable region from a document page and return it as a new image.",
    "parameters": {
      "type": "object",
      "properties": {
        "page": {
          "type": "integer",
          "description": "1-based page number."
        },
        "bbox_2d": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 4,
          "maxItems": 4,
          "description": "[x1, y1, x2, y2] normalized to [0, 1000]."
        }
      },
      "required": ["page", "bbox_2d"]
    }
  }
}
```

注入方式冻结为“数据加载时统一注入”：

- schema 独立存为一个 JSON 列表，由 `data.tools_schema_path` 指定。
- VTC Parquet 不重复保存 `tools` 列。
- 缺省启用全局 schema；仅对显式 `enable_tools: false` 的样本禁用。
- 保留逐样本 `tools` 作为兼容覆盖，不作为 VTC 主数据格式。
- clear/no-tool 对照仍启用 schema；完全非 agent 样本才关闭并使用空 `env_name`。

页码固定为 **1-based**，与 VTC_REL 的 `metadata.evidence_locations[].page`
保持一致。bbox 固定为 `[0, 1000]` 归一化坐标，与
`metadata.evidence_locations[].bbox` 保持一致。

### 3.2 工具语义

- 每条阶段 1 trajectory 最多成功调用一次工具。
- 每次都从原始页面裁剪，不从上一次 crop 继续裁剪。
- crop 在四周加入目标框宽高各 5% 的上下文 padding，并裁到页面边界。
- 阶段 1 只做原 PNG crop，不进行 PDF 高 DPI 重渲染。
- 非法 JSON、未知工具、非法页码和非法 bbox 都显式失败，不静默纠正。
- crop 作为 Qwen3-VL 原生 `tool` role 中的新图片返回。

### 3.3 模型协议

使用 checkpoint 自带的 Qwen3-VL chat template：

```text
system: tools schema
user: original page image + question
assistant: <tool_call>...</tool_call>
tool: <tool_response> + crop image
assistant: final answer
```

不使用 Glyph box token，也不手写另一套 function-call token。

## 4. 阶段 1 数据范围

- 单页样本。
- 单一主要 evidence bbox。
- 一次工具调用。
- 优先使用规则可判分的短答案。
- prompt 上限暂定 8192 tokens。
- 单次 action 上限暂定 512 tokens。
- 第二轮最终答案上限暂定 256 tokens。

固定 smoke 样本：

- ID：`ChatQA-Training-Data_tatqa_10955`
- 页面：1 页，589 x 126 px。
- GT bbox：`[20, 690, 940, 770]`（0～1000）。
- 问题：表格中 Key Management Personnel 总薪酬由哪些部分组成。

固定样本仅用于连接测试，不作为模型定位质量结论。

## 5. 阶段 1 验收标准

必须满足：

1. Qwen3-VL processor 正确渲染原生 tools、tool call 和带图片的 tool response。
2. Qwen3-VL 使用 3 维 mRoPE position ids，而不是退化为普通 1 维位置。
3. `zoom_region` 正确执行 1-based page 和归一化 bbox 转换。
4. crop 作为第二张视觉 observation 被加入第二轮上下文。
5. 工具 observation 的 token 在 verl trajectory 中 `action_mask=0`。
6. 至少一条 Qwen3-VL-8B 的实际两轮无训练 rollout 成功完成，或明确记录阻塞它的环境兼容问题。

## 6. 明确不在阶段 1 处理的内容

- GRPO reward 和 advantage。
- optimizer step、checkpoint 和恢复。
- FSDP 到 vLLM 的热更新。
- tool-needed 数据筛选。
- clear/no-tool 对照数据。
- 多页、多 evidence、多次调用。
- PDF 区域高 DPI 重渲染。
- 最终答案准确率提升。

## 7. 版本门槛与当前环境

Qwen3-VL 官方仓库要求 Transformers >= 4.57.0。本机
`qwen3vl-vllm` 环境为：

```text
Python       3.12.13
PyTorch      2.9.0
Transformers 4.57.6
vLLM         0.11.2
```

该环境能够运行 Qwen3-VL/vLLM，但缺少完整 verl 训练依赖。现有 GRPO
`setup.py` 又将 vLLM 限制为 `<=0.8.3`，因此完整训练环境需要在后续阶段单独
解决，不能通过降级 vLLM 破坏 Qwen3-VL 支持。

## 8. 参考

- Qwen3-VL 官方仓库：https://github.com/QwenLM/Qwen3-VL
- Transformers Qwen3-VL 文档：https://huggingface.co/docs/transformers/model_doc/qwen3_vl
- vLLM 文档：https://docs.vllm.ai/
- DeepEyes：https://github.com/Visual-Agent/DeepEyes

