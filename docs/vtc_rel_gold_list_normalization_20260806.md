# VTC-REL `metadata.gold` 列表化记录（2026-08-06）

## 目的

修复 Hugging Face `Dataset.from_list()` / PyArrow 建表时报出的：

```text
pyarrow.lib.ArrowInvalid: cannot mix list and non-list, non-null values
```

根因是 `data/VTC_REL/gemini-3.5-flash-30k/train.jsonl` 中的
`metadata.gold` 同时存在 list 和 string。

## 修改

- 总行数：77,584
- 原本为 list、保持不变：4,000 条
- 从 string 改为单元素 list：2,000 条
  - `needle_single`：1,000 条
  - `needle_multikey`：1,000 条
- 修改形式：`"gold": "value"` → `"gold": ["value"]`
- 修改后非空 `metadata.gold`：6,000 条，类型全部为 list

逐行语义对比确认：除上述 2,000 个 `metadata.gold` 的包装外，ID、图片、
对话、文本路径、DPI 144 图片字段及所有其他 metadata 均未变化。

## 文件与恢复点

- 修改文件：
  `data/VTC_REL/gemini-3.5-flash-30k/train.jsonl`
- 修改前备份：
  `data/VTC_REL/gemini-3.5-flash-30k/train.jsonl.bak_pre_gold_list_20260806`
- 修改前 SHA-256：
  `f2e55663eec42ed8714fc8bcbc59c9a47e5d528b613bd13b805ec216eb3eaa1b`
- 修改后 SHA-256：
  `8a74ef5ee77157ba9bda08abcf6c62331e5715047f30c95bff06297e10f13917`

文件先写入同目录临时文件，完成逐行语义校验和 PyArrow 建表测试后，再通过
`os.replace()` 原子替换；目录也执行了 `fsync`。

## 验证

使用训练环境：

```text
/vepfs-mlp2/c20250405/400042/miniconda3/envs/vtc/bin/python
datasets==4.8.4
```

验证结果：

```text
semantic_diff_OK rows 77584 changed 2000
gold_types {'list': 6000}
Dataset.from_list_OK rows 77584
atomic_replace_OK
```

因此原来触发 `cannot mix list and non-list` 的 `metadata.gold` schema 冲突已
消除。
