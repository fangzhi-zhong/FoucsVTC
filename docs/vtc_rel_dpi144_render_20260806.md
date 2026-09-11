# VTC-REL 144 DPI 高清重建记录（2026-08-06）

## 结论

在全量渲染前，已选取 15 个样本，覆盖 VTC_SFT、RULER_v1_SFT、
TRANSCRIBE_SFT、VTC_GAP 以及代码/非代码排版。共比较 58 个旧页面：使用
原数据集的排版参数重新渲染 72 DPI 后，58/58 页面尺寸和像素均与旧图完全
一致。随后用同一份矢量 PDF 渲染 144 DPI，裁剪尺寸相对旧图严格二倍缩放的
最大偏差为 1 px，文字 ink bounding box 的最大偏差也为 1 px。

因此全量任务固定排版点数，只把 raster DPI 从 72 提高到 144；不通过放大旧
PNG 生成高清图。

## 字体反查

| 来源/类型 | 字体 | SHA-256 |
| --- | --- | --- |
| VTC_SFT | `data/VTC_SFT/word2png/config/Verdana.ttf` | `96ed14949ca4b7392cff235b9c41d55c125382abbe0c0d3c2b9dd66897cae0cb` |
| RULER_v1_SFT | `data/VTC_SFT/word2png/config/Verdana.ttf` | `96ed14949ca4b7392cff235b9c41d55c125382abbe0c0d3c2b9dd66897cae0cb` |
| TRANSCRIBE_SFT | `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` | `690243adfefe0ce154b547db6205794bd30ac4277275179517a90994f4980648` |
| VTC_GAP 非代码 | `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` | `690243adfefe0ce154b547db6205794bd30ac4277275179517a90994f4980648` |
| VTC_GAP 代码 | `/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf` | `39c29931201f08dd89fdba4129c76288e6baeecf7c94fe4f6a757f2b50718b1b` |

VTC_SFT 中有 3,967 个文本包含 CJK 字符，但代表性 CJK 样本用 Verdana 可以与
旧图逐像素一致；改用 SourceHan 则不能。因此不能根据文本是否包含中文自动
切换字体。

## 代表样本

试验覆盖 ChatQA TatQA、ROPES、ChatQA2 long_sft、NarrativeQA、Trivia、包含
CJK 的 multihop、TRANSCRIBE page/box/needle、RULER，以及 VTC_GAP 的
long/count/needle/code-page/code-complete。详细逐页结果位于：

- `data/VTC_REL/gemini-3.5-flash-30k/dpi144_probe/probe_report.json`
- `data/VTC_REL/gemini-3.5-flash-30k/dpi144_probe/<id>/dpi72/`
- `data/VTC_REL/gemini-3.5-flash-30k/dpi144_probe/<id>/dpi144/`

VTC_SFT 的旧裁剪脚本存在一个需要保留的历史细节：宽度为
`last_col + 1 + margin`，最后一页高度为 `last_row + margin`，高度分支没有
`+1`。144 DPI 时 margin 按 `dpi / 72` 缩放。保留该差异后 72 DPI 才能逐像素
复现旧图。

## 实现与输出

- 试验脚本：`VTC/train/GRPO/examples/data_preprocess/probe_vtc_rel_dpi144.py`
- 全量脚本：`VTC/train/GRPO/examples/data_preprocess/render_vtc_rel_dpi144.py`
- 高清图：`data/VTC_REL/gemini-3.5-flash-30k/images_dpi144/`
- 可续跑索引：`images_dpi144/index.jsonl`
- 失败记录：`images_dpi144/failed.jsonl`
- 后台日志：`images_dpi144/render.log`
- 后台 PID：`images_dpi144/render.pid`

高清图按 `images_dpi144/<来源>/<原相对子目录>/<id>/page_NNN.png` 保存。每个
文档的全部页面完成、尺寸校验通过并原子落盘后，才写一条 index 记录。中断后
重启会跳过 index 中已完成且文件存在的文档。

PNG 使用 `compress_level=1`。该参数只改变 PNG 编码与文件大小，不改变解码后
的任何像素；已用 Pillow `ImageChops.difference` 验证 `pixel_equal=True`。在
174 张试验图的并发负载基准中，默认压缩为 131.8 秒，level 1 为 89.1 秒。

## 全量规模与清单更新条件

- 文档数：77,584
- 页面数：567,385
- VTC_SFT：253,926 页
- RULER_v1_SFT：15,005 页
- TRANSCRIBE_SFT：48,751 页
- VTC_GAP：249,703 页

脚本只有在 77,584 个文档全部成功后才会：

1. 校验 ID 覆盖、页面数量、文件存在性和相对二倍尺寸偏差；
2. 按 `train.jsonl` 顺序规范化 index；
3. 备份为 `train.jsonl.bak_pre_dpi144`；
4. 原子写回 `train.jsonl`，新增 `image_dpi144` 和
   `image_sizes_dpi144`；原 `image` 字段保持不变。

## 启动状态

正式渲染于 2026-08-06 启动。交互试跑与续跑阶段已完成 3,402 个文档且失败为
0，随后转为与终端解耦的后台可续跑任务。最终状态以 `render.log` 中的
`REPORT` 行、全量 index 和更新后的 train 清单为准。

后台启动命令：

```bash
/vepfs-mlp2/c20250405/400042/miniconda3/envs/vtc/bin/python -u \
  VTC/train/GRPO/examples/data_preprocess/render_vtc_rel_dpi144.py \
  --processes 96
```
