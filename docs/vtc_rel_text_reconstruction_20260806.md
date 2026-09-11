# VTC-REL 原始渲染文本恢复记录（2026-08-06）

## 结果

- 主 manifest：`/vepfs-mlp2/c20250405/400042/data/VTC_REL/gemini-3.5-flash-30k/train.jsonl`
- 文本目录：`/vepfs-mlp2/c20250405/400042/data/VTC_REL/gemini-3.5-flash-30k/text`
- 总样本数：77,584
- 非空文本文件：77,584
- 恢复文本总字节数：3,568,247,628 bytes（UTF-8）
- `train.jsonl` 每行已加入绝对路径字段 `text_path`
- 更新前备份：`train.jsonl.bak_pre_text_path`
- 主 manifest 在原子替换后再次逐行检查，77,584 个 `text_path` 均存在且非空。

实际数据除用户指定的三个来源外，还有 3,024 条 `RULER_v1_SFT`。为保证 manifest 全覆盖，本次一并恢复。

## 目录结构

```text
text/
├── TRANSCRIBE_SFT/{page,box,needle}/<id>.txt
├── VTC_SFT/<source>/<subset>/<id>.txt
├── VTC_GAP/<task>/<variant>/<id>.txt
└── RULER_v1_SFT/<task>/<id>.txt
```

每个 `.txt` 保存实际进入原渲染链路的完整文本，而不是只保存 answer 或 evidence。渲染脚本会移除的软连字符 `\u00ad` 和零宽空格 `\u200b` 也按原逻辑清理。

## 覆盖统计

| 来源 | 数量 | 恢复方式 |
|---|---:|---|
| TRANSCRIBE_SFT | 29,959 | 按 seed 重放 ChatQA2 page/box 切片；RULER needle 按上游 ID 回查 |
| VTC_SFT | 29,411 | 按样本 ID 回查对应原始字段 |
| VTC_GAP | 15,190 | 按原 seed/源码/生成器重建并校验 gold、SHA 或 next-line |
| RULER_v1_SFT | 3,024 | 由 ID 中的 task、length、row index 回查 RULER JSONL context |

TRANSCRIBE 细分：

- ChatQA2 page/box：24,959
- RULER v1 needle：4,072
- RULER v2 needle：928

VTC_GAP 细分：

- long：6,190
- count：4,000（cwe 1,500；fwe 1,500；pcnt 1,000）
- needle：2,000（single 1,000；multikey 1,000）
- code：3,000（page 1,500；complete 1,500）

## 原渲染脚本映射

### TRANSCRIBE_SFT

- 渲染/生成脚本：`data/TRANSCRIBE_SFT/build_transcribe_sft.py`
- page/box：按原始 seed `20260804`、原 ChatQA2 pool 顺序、shuffle 和 `chunk_doc` 规则重放。
- 首轮 pool 与原链路一致：long_sft 47,184（扫描前 57,000 个源对象），NarrativeQA 18,000（扫描前 18,000 个源对象）。
- needle：直接回查 `data/huggingface/ruler_jsonl` 或 `data/RULER_v2/.../test.jsonl` 中的 context/haystack。

### VTC_SFT

| 子集 | 原渲染脚本 | 保存字段 |
|---|---|---|
| ChatQA-Training-Data | `word2png/word2png_ChatQA-Training-Data-1.py` | `document` |
| multihop | `word2png/word2png_multihop.py` | `document` |
| ChatQA2 long_sft | `word2png/word2png_ChatQA2-Long-SFT-data-long-sft.py` | `question` |
| ChatQA2 NarrativeQA | `word2png/word2png_ChatQA2-Long-SFT-data-Narrative.py` | `sub-paragraphs` |
| TriviaQA | `word2png/word2png_trivia_qa.py` | `search_results.search_context`；list 用三个换行拼接 |

渲染脚本目录：`data/VTC_SFT/word2png`。

### VTC_GAP

| 子集 | 原脚本 | 恢复与校验 |
|---|---|---|
| long | `build_long.py` | 按 seed、evidence 文档和同 corpus distractor pool 重建；6,190 条 `body_sha256`、长度、evidence start/end 全匹配 |
| count | `build_count.py` + `gen_count.py` | 恢复原 4,011 段 paragraph pool；4,000 条 context/gold/counts 全匹配 |
| needle | `build_needle.py` + `gen_needle.py` | 恢复原 1,500 个 haystack；2,000 条 UUID gold 全匹配 |
| code | `build_code.py` | page 使用原脚本保证与渲染输入相同的 transcript；complete 按 repo/path、seed 重建前缀并核对 next line |

共享 PDF/PNG 排版入口：`data/VTC_GAP/common.py`。

### RULER_v1_SFT

- 渲染/生成脚本：`data/RULER_v1_SFT/build_ruler_sft.py`
- PDF 构建入口：`data/RULER_v1_VTC/render_ruler_v1.py`

## 可复现脚本

脚本：`VTC/train/GRPO/examples/data_preprocess/reconstruct_vtc_rel_text.py`

```bash
cd /vepfs-mlp2/c20250405/400042/VTC
python3 train/GRPO/examples/data_preprocess/reconstruct_vtc_rel_text.py
```

脚本行为：

1. 流式读取大 JSON array，避免加载 5–9GB 文件到内存。
2. 按来源恢复文本，并逐个以临时文件 + `os.replace` 写入。
3. 全部文本通过覆盖、非空、唯一性及来源专属校验后，才创建备份并原子替换 `train.jsonl`。
4. 替换后再次逐行检查 `text_path`。

## 后续阶段

本次不实现高清局部渲染和 cache。下一阶段应直接针对上表中的现有渲染脚本增加“给定 `text_path`、page、bbox 和目标 DPI，只重渲染/裁剪对应区域”的入口，而不是重新写一个统一排版器。
