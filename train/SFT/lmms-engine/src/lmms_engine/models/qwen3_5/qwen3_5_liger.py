from typing import Optional, Union

import torch
from transformers import Qwen3_5ForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
)

from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP


class LigerQwen3_5RMSNorm(LigerRMSNorm):
    """Liger RMSNorm with Qwen3.5's zero-centered weights."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(
            dim,
            eps=eps,
            offset=1.0,
            casting_mode="gemma",
            init_fn="zeros",
            in_place=False,
        )


class LigerQwen3_5SwiGLUMLP(LigerSwiGLUMLP):
    """Accept the extra intermediate-size argument used by Qwen3.5."""

    def __init__(self, config, intermediate_size=None):
        super().__init__(config)


def qwen3_5_lce_forward(
    self: Qwen3_5ForConditionalGeneration,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Qwen3_5CausalLMOutputWithPast:
    """Qwen3.5 forward using fused linear cross entropy during training."""
    num_items_in_batch = kwargs.pop("num_items_in_batch", None)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        **kwargs,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None

    if labels is not None:
        # Packed sample starts are already -100 in Qwen3_5PackingCollator, so
        # the ordinary causal shift cannot train across packed boundaries.
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_hidden_states = shift_hidden_states.view(
            -1,
            self.config.text_config.hidden_size,
        )
        shift_labels = shift_labels.view(-1)

        reduction = "sum" if num_items_in_batch is not None else "mean"
        lce = LigerFusedLinearCrossEntropyLoss(
            reduction=reduction,
            accum_dtype=torch.float32,
        )
        loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
    else:
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    return Qwen3_5CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )
