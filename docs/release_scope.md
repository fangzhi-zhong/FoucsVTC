# Code release scope

This repository packages REL-CoT rendering and SFT format conversion,
test-document rendering, the SFT and GRPO training stages, and tool-assisted
inference and benchmark evaluation. The REL-CoT tools render existing dataset
text and convert supplied supervision; automatic REL annotation and synthetic
training-task generation are not included. Local training-framework
modifications are retained because the model, loss, multimodal rollout, and
crop behavior depend on them. See the module guides for their installation environments.

The release contains source code, prompts, configuration templates, dependency
declarations, documentation, and two selected paper illustrations used in the
README: Figure 1, including its result plots, and the method overview. It
excludes raw experiment outputs and score files, the full paper source,
other paper figures, generated annotations, original training/evaluation
data and dataset images, font binaries, model weights and intermediate checkpoints,
tracking logs, caches, editor settings, and historical Git metadata.

Paths and service credentials are supplied by the user through command-line
arguments, environment variables, or configuration files. Generated data,
checkpoints, and outputs belong under `datasets/`, `models/`, and `outputs/`
(or an external directory); these paths are ignored by Git. `data/` contains
only REL-CoT rendering/conversion tools and instructions. Download the public
[REL-CoT dataset](https://www.modelscope.cn/datasets/zhongfangzhi/REL-CoT)
separately, for example to `datasets/REL-CoT/`, and write prepared SFT data to
`datasets/REL-CoT_SFT/`. The dataset itself is not bundled in `data/`.
`eval/rendering/` contains the document, RULER v1/v2, LongBench, MRCR, and
VTCBench-Wild test-data renderers. Font calibration remains under
`eval/font_acuity/`.
`.env.example` is a template; `.env` is ignored and
must be explicitly sourced if used.

This is a source release, not a bundled reproduction environment. It requires
user-provided model checkpoints, dataset inputs, fonts, and suitable GPU
software for training and model serving. The public examples describe runnable
interfaces and configuration choices without asserting that every recipe
exactly reproduces a paper experiment. The selected illustrations present
paper results; underlying experiment outputs are not bundled.

Third-party copyright headers and license texts are retained where code is
redistributed. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
