from functools import partial, wraps
from types import MethodType

from transformers import PreTrainedModel

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


_DECODER_PACKING_KEYS = {
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "max_length_q",
    "max_length_k",
    "seq_idx",
}


def _get_composite_and_text_model(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5TextModel,
    )

    if isinstance(model, Qwen3_5ForConditionalGeneration):
        return model.model, model.model.language_model
    if isinstance(model, Qwen3_5Model):
        return model, model.language_model
    if isinstance(model, Qwen3_5ForCausalLM):
        return None, model.model
    if isinstance(model, Qwen3_5TextModel):
        return None, model
    raise TypeError(
        "Unsupported Qwen3.5 model type. Expected Qwen3_5ForConditionalGeneration, "
        f"Qwen3_5Model, Qwen3_5ForCausalLM, or Qwen3_5TextModel; got {type(model)}."
    )


@MONKEY_PATCHER.register("qwen3_5", "packing")
def apply_qwen3_5_packing_patch(
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
) -> None:
    """Keep decoder-only packing metadata out of Qwen3.5's vision tower."""
    if use_rmpad:
        raise NotImplementedError("Qwen3.5 uses its native packing path, not use_rmpad.")
    if model is None:
        raise ValueError("The Qwen3.5 packing patch requires a loaded model instance.")

    composite_model, _ = _get_composite_and_text_model(model)
    if composite_model is None:
        return
    if getattr(composite_model, "_lmms_packing_patch_applied", False):
        return

    for method_name in ("get_image_features", "get_video_features"):
        original_method = getattr(composite_model, method_name)

        @wraps(original_method)
        def without_decoder_packing(_model, *args, __method=original_method, **kwargs):
            for key in _DECODER_PACKING_KEYS:
                kwargs.pop(key, None)
            return __method(*args, **kwargs)

        setattr(
            composite_model,
            method_name,
            MethodType(without_decoder_packing, composite_model),
        )

    composite_model._lmms_packing_patch_applied = True


@MONKEY_PATCHER.register("qwen3_5", "liger")
def apply_liger_kernel_to_qwen3_5(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
) -> None:
    """Apply compatible Liger kernels to the official Qwen3.5 model."""
    if cross_entropy and fused_linear_cross_entropy:
        raise ValueError(
            "cross_entropy and fused_linear_cross_entropy cannot both be enabled."
        )
    if use_rmpad:
        raise NotImplementedError("Qwen3.5 uses its native packing path, not use_rmpad.")
    if rope:
        raise NotImplementedError(
            "Liger RoPE does not support Qwen3.5's partial interleaved MRoPE."
        )

    from liger_kernel.transformers.functional import liger_cross_entropy
    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    from .qwen3_5_liger import (
        LigerQwen3_5RMSNorm,
        LigerQwen3_5SwiGLUMLP,
        qwen3_5_lce_forward,
    )

    patch_qwen3_5_rms_norm = partial(
        _patch_rms_norm_module,
        offset=1.0,
        casting_mode="gemma",
        in_place=False,
    )

    if rms_norm:
        modeling_qwen3_5.Qwen3_5RMSNorm = LigerQwen3_5RMSNorm
    if swiglu:
        modeling_qwen3_5.Qwen3_5MLP = LigerQwen3_5SwiGLUMLP
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy
    if fused_linear_cross_entropy:
        if model is not None:
            if not isinstance(model, Qwen3_5ForConditionalGeneration):
                raise TypeError(
                    "Fused linear cross entropy requires "
                    f"Qwen3_5ForConditionalGeneration; got {type(model)}."
                )
            model.forward = MethodType(qwen3_5_lce_forward, model)
            model.accepts_loss_kwargs = True
        else:
            modeling_qwen3_5.Qwen3_5ForConditionalGeneration.forward = (
                qwen3_5_lce_forward
            )
            modeling_qwen3_5.Qwen3_5ForConditionalGeneration.accepts_loss_kwargs = True

    if model is None:
        return

    _, text_model = _get_composite_and_text_model(model)
    if rms_norm:
        patch_qwen3_5_rms_norm(text_model.norm)

    for decoder_layer in text_model.layers:
        if rms_norm:
            patch_qwen3_5_rms_norm(decoder_layer.input_layernorm)
            patch_qwen3_5_rms_norm(decoder_layer.post_attention_layernorm)
            if decoder_layer.block_type == "full_attention":
                patch_qwen3_5_rms_norm(decoder_layer.self_attn.q_norm)
                patch_qwen3_5_rms_norm(decoder_layer.self_attn.k_norm)
        if swiglu:
            _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
