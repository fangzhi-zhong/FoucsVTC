# FocusVTC GRPO runtime

This directory contains the locally modified
[verl](https://github.com/volcengine/verl) runtime and the FocusVTC Qwen3.5
training recipe. Install this checkout to retain its model and agent changes.

The FocusVTC path includes:

- Qwen3.5 loading, position handling, response-only logits, and chunked entropy;
- vLLM SPMD rollouts with repeated `zoom_region` calls and image observations;
- low-resolution prompts with lazy high-resolution page loading;
- prompt-length filtering and skipped-sample batch refill;
- answer, format, and evidence-aware tool rewards.

The agent registry contains only the local document zoom environment. Private
retrieval/search services, old experiment launchers, logs, datasets, and
checkpoints are excluded. Shared upstream runtime modules are retained to keep
the framework import graph intact; the documented recipe uses FSDP and vLLM.

See [training instructions](../../docs/training.md). Framework source is
Apache-2.0, with original copyright and attribution comments preserved.
Adapted torchtune helpers also retain their
[BSD-3-Clause license](licenses/LICENSE.torchtune). The tool abstraction retains
its [Agent-R1 attribution](https://github.com/0russwest0/Agent-R1) and
[MIT license](licenses/LICENSE.Agent-R1).
