# FocusVTC evaluation

See [the evaluation guide](../docs/evaluation.md) for installation, paired-page
inference, configuration, benchmark data formats, and tool-policy details.

- `infer.py`: question answering over ordered local document pages.
- [`rendering/`](rendering/README.md): document, RULER v1/v2, LongBench, MRCR,
  and VTCBench-Wild test-data rendering.
- `tool_agent/`: shared zoom gateway, server launcher, and benchmark configs.
- `RULER/`, `LongBench/`, `MRCR/`: core inference and scoring code.
- `OCRBench/tasks/`: adapter to the external lmms-eval implementation.
- [`font_acuity/`](font_acuity/README.md): document font calibration utilities.

MMMU and OCRBench use dataset-provided images and do not need a text renderer.

No experimental predictions, figures, logs, datasets, or model weights are
included. Runtime outputs belong under `outputs/` at the repository root.
