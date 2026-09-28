# Third-party code and attribution

Third-party components retain their own licenses and copyright notices. Local
modifications provide FocusVTC model support, multimodal training, evidence
localization, resolution-enhancement tools, or portable configuration.

| Component | Included code | License / source |
| --- | --- | --- |
| LMMs-Engine | Selected SFT runtime under `train/SFT/lmms-engine/` | [Apache-2.0](train/SFT/lmms-engine/LICENSE); [upstream](https://github.com/EvolvingLMMs-Lab/lmms_engine) |
| verl and its retained components | GRPO runtime under `train/GRPO/verl/` | [Apache-2.0](train/GRPO/LICENSE); [upstream](https://github.com/volcengine/verl); per-file notices also apply |
| LongBench | Dataset prompt/metric configuration and evaluation adaptations under `eval/LongBench/` | [MIT](eval/LongBench/LICENSE); [upstream](https://github.com/THUDM/LongBench) |
| NVIDIA RULER / NeMo-Skills | RULER evaluation metric adaptations under `eval/RULER/` and prompt templates in `eval/rendering/ruler2_split.py` | [Apache-2.0](eval/RULER/LICENSE); [RULER license](https://github.com/NVIDIA/RULER/blob/main/LICENSE), [NeMo-Skills license](https://github.com/NVIDIA/NeMo-Skills/blob/main/LICENSE) |
| Qwen3.5-9B | Adapted inference chat template in `eval/tool_agent/` | [Apache-2.0](eval/tool_agent/LICENSE), [modification notice](eval/tool_agent/NOTICE); [upstream license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE) |
| torchtune | FSDP2 utility adaptations in the training frameworks | BSD-style license; see each framework's `licenses/LICENSE.torchtune` and retained Meta copyright notices |
| Agent-R1 | Agent/tool abstractions retained in the GRPO runtime | MIT; see `train/GRPO/licenses/` and retained source notices |

The framework subdirectories document the scope of their retained code and
local changes. Copyright headers in source files remain authoritative for
their respective components.

Models, datasets, fonts, and external evaluation suites are obtained separately
and remain subject to their own terms. VTCBench and lmms-eval integrations
expect separately installed tools; their full source trees and datasets are
not redistributed here.

No project-wide license for original FocusVTC code has been selected in this
prepared release. The repository owner should add the chosen root `LICENSE`
before publishing it as an open-source project. This does not alter the
licenses of the third-party components listed above.
