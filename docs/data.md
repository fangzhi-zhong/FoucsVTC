# Data preparation

For the public [REL-CoT training dataset](https://www.modelscope.cn/datasets/zhongfangzhi/REL-CoT),
see [`data/README.md`](../data/README.md). It covers downloading the original
text, 72 DPI images, and supervision, then rendering seven DPI variants with
DejaVu Sans and preparing SFT manifests. `data/` holds tools and instructions;
downloaded data and rendered outputs belong under `datasets/` or outside the
checkout. The tools reuse existing reasoning and answers and remap their
evidence references to the new layout without generating new supervision.

Evaluation-data renderers and their instructions are under
[`eval/rendering/`](../eval/rendering/README.md). The guide covers documents,
RULER v1/v2, LongBench, MRCR, and VTCBench-Wild, including source layouts,
fonts, dependencies, and commands for paired DPI pages. MMMU and OCRBench
use dataset-provided images.

See the [training guide](training.md) for SFT and conversion of existing
supervision to GRPO Parquet inputs, the [evaluation guide](evaluation.md)
for benchmark configuration and scoring, and
[font calibration](../eval/font_acuity/README.md) for font and point-size
experiments.
