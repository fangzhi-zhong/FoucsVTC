# FocusVTC SFT runtime

This directory vendors the FocusVTC-relevant portion of the locally modified
[lmms-engine](https://github.com/EvolvingLMMs-Lab/lmms_engine) by LMMs-Lab.
It is installed as `lmms_engine`; install this directory instead of an unrelated
upstream release when running the supplied recipe.

The retained implementation includes Qwen3.5/Qwen3-VL processors, iterable
multimodal loading, target masking, grounding prompts, sequence packing,
FSDP2 training, and the Qwen3.5 Liger/packing patches. Qwen3.5 packing carries
per-example mRoPE positions and the sequence boundaries required by both
attention and DeltaNet/causal convolution. Unrelated custom model families,
archived experiments, data, and model artifacts are omitted.

Release preparation also restricts package registration to the retained
modules, resolves environment variables in the training YAML, and resolves
relative conversation paths against the configured data folder.

See [training instructions](../../../docs/training.md). The upstream framework
is Apache-2.0; original source notices remain in place. The adapted FSDP helpers
also retain the [torchtune BSD-3-Clause license](licenses/LICENSE.torchtune).
