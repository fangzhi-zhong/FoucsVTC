"""Small dispatcher for Qwen-VL processor families used by verl datasets and agents."""

from typing import Optional

import torch


def get_processor_family(processor) -> Optional[str]:
    if processor is None:
        return None

    processor_name = processor.__class__.__name__
    if processor_name in {"Qwen3VLProcessor"}:
        return "qwen3_vl"
    if processor_name in {"Qwen2VLProcessor", "Qwen2_5_VLProcessor"}:
        return "qwen2_vl"

    image_processor = getattr(processor, "image_processor", None)
    image_processor_name = image_processor.__class__.__name__ if image_processor is not None else ""
    if image_processor_name.startswith("Qwen2VLImageProcessor"):
        return "qwen2_vl"
    return None


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    mm_token_type_ids: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    family = get_processor_family(processor)
    if family == "qwen3_vl":
        from verl.models.transformers.qwen3_vl import get_rope_index as get_qwen3_vl_rope_index

        return get_qwen3_vl_rope_index(
            processor,
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
    if family == "qwen2_vl":
        from verl.models.transformers.qwen2_vl import get_rope_index as get_qwen2_vl_rope_index

        return get_qwen2_vl_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )
    raise ValueError(f"Unsupported processor for multimodal mRoPE: {processor.__class__.__name__}")
