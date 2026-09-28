from typing import Optional

import numpy as np
import torch
from PIL.Image import Image

from lmms_engine.mapping_func import register_processor

from .qwen3_vl_processor import Qwen3_VLDataProcessor


def build_qwen3_5_mrope_position_ids(
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build one sample's 3D M-RoPE positions with a zero-based origin."""

    if mm_token_type_ids.ndim != 1:
        raise ValueError(
            f"mm_token_type_ids must be one-dimensional per sample, got {mm_token_type_ids.shape}."
        )

    if video_grid_thw is not None:
        # Qwen3.5 inserts a timestamp and a separate vision span for every frame.
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw,
            video_grid_thw[:, 0],
            dim=0,
        ).clone()
        video_grid_thw[:, 0] = 1

    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else iter(()),
        2: iter(video_grid_thw) if video_grid_thw is not None else iter(()),
    }
    token_types = mm_token_type_ids.tolist()
    groups = []
    start = 0
    while start < len(token_types):
        modality = token_types[start]
        end = start + 1
        while end < len(token_types) and token_types[end] == modality:
            end += 1
        groups.append((modality, start, end))
        start = end

    position_chunks = []
    current_pos = 0
    for modality, start, end in groups:
        group_length = end - start
        if modality == 0:
            positions = torch.arange(
                group_length,
                dtype=torch.long,
                device=mm_token_type_ids.device,
            )
            position_chunks.append(positions.view(1, -1).expand(3, -1) + current_pos)
            current_pos += group_length
            continue
        if modality not in grid_iters:
            raise ValueError(f"Unsupported Qwen3.5 modality type id: {modality}.")

        try:
            grid_thw = next(grid_iters[modality])
        except StopIteration as exc:
            raise ValueError(
                f"Missing grid_thw entry for modality type {modality}."
            ) from exc

        grid_t = int(grid_thw[0].item())
        grid_h = int(grid_thw[1].item()) // spatial_merge_size
        grid_w = int(grid_thw[2].item()) // spatial_merge_size
        temporal = torch.arange(grid_t, device=mm_token_type_ids.device)
        height = torch.arange(grid_h, device=mm_token_type_ids.device) + current_pos
        width = torch.arange(grid_w, device=mm_token_type_ids.device) + current_pos
        t_grid, h_grid, w_grid = torch.meshgrid(
            temporal,
            height,
            width,
            indexing="ij",
        )
        vision_positions = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
        vision_positions[0] += current_pos
        if vision_positions.shape[1] != group_length:
            raise ValueError(
                "Qwen3.5 media token count does not match grid_thw: "
                f"tokens={group_length}, grid_positions={vision_positions.shape[1]}."
            )
        position_chunks.append(vision_positions)
        current_pos += max(int(grid_thw[1].item()), int(grid_thw[2].item())) // spatial_merge_size

    if not position_chunks:
        return torch.empty((3, 0), dtype=torch.long, device=mm_token_type_ids.device)
    return torch.cat(position_chunks, dim=1).to(dtype=torch.long)


@register_processor("qwen3_5")
class Qwen3_5DataProcessor(Qwen3_VLDataProcessor):
    """Qwen3.5 SFT processor built on its Qwen3-VL-compatible frontend."""

    @property
    def chat_template_no_system(self):
        """Use the training page layout ``Page 1<image>Page 2<image>``.

        The base Qwen-VL template historically put a newline after every image
        placeholder. Qwen3.5's DeltaNet decoder is sensitive to this extra
        token, so the Qwen3.5 template keeps adjacent page markers and image
        placeholders contiguous.
        """
        return super().chat_template_no_system.replace(
            "<|vision_end|>\n", "<|vision_end|>"
        )

    def process(
        self,
        images: list[Image],
        hf_messages,
        audios: Optional[list[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
        add_generation_prompt: bool = False,
        think_mode: bool = False,
        **kwargs,
    ):
        # Qwen3.5's bundled tokenizer template rejects a standalone system
        # message. Insert it into the message list so the processor's SFT chat
        # template handles it in the same pass as the other turns.
        if add_system_prompt and hf_messages[0]["role"] != "system":
            hf_messages = [
                # The Transformers processor path expects every message
                # content to be a list of typed blocks (a bare string reaches
                # its visual-content scan and raises ``TypeError``).
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                *hf_messages,
            ]
        add_system_prompt = False

        inputs = super().process(
            images=images,
            hf_messages=hf_messages,
            audios=audios,
            sampling_rate=sampling_rate,
            videos=videos,
            system_message=system_message,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
            think_mode=think_mode,
            **kwargs,
        )

        input_ids = inputs["input_ids"]
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        mm_token_type_ids[input_ids == self.video_token_id] = 2
        inputs["mm_token_type_ids"] = mm_token_type_ids
        inputs["position_ids"] = build_qwen3_5_mrope_position_ids(
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            spatial_merge_size=self.processor.image_processor.merge_size,
        )
        return inputs
