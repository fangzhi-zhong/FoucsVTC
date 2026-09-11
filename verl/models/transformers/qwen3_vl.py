# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-family multimodal position ids for pre-tokenized verl trajectories.

Transformers 5.x changed Qwen3-VL/Qwen3.5 mRoPE to use the processor-produced
``mm_token_type_ids`` stream.  In particular, the position cursor advances by
``max(height, width) // spatial_merge_size`` after an image rather than by the
number of image tokens.  Keeping this implementation here lets verl pass
pre-tokenized, left-padded trajectories to both Qwen3-VL and Qwen3.5 without
instantiating a model just to compute position ids.
"""

import itertools
from typing import Optional

import torch


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    mm_token_type_ids: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return Qwen3-family 3D mRoPE ids for one example.

    ``input_ids`` and ``attention_mask`` may be left padded.  The public verl
    callers use the single-example form (``[seq]``); accepting ``[1, seq]`` as
    a convenience makes the helper less error-prone in small integration
    tests.  The return value is ``[3, seq]`` (the shape expected by the FSDP
    actor), while Transformers' additional mRoPE delta is not needed because
    verl supplies explicit position ids on every forward pass.
    """

    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError("get_rope_index accepts one example at a time")
        input_ids = input_ids[0]
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape={tuple(input_ids.shape)}")

    seq_len = input_ids.shape[0]
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    else:
        attention_mask = attention_mask.reshape(-1).to(device=device).bool()
        if attention_mask.numel() != seq_len:
            raise ValueError("attention_mask and input_ids must have the same length")

    if mm_token_type_ids is None:
        # Older callers/tests did not retain this processor field.  Infer the
        # only reliable information available from token ids; normal text is
        # type 0 and visual pad tokens are type 1/2.
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        mm_token_type_ids[input_ids == image_token_id] = 1
        mm_token_type_ids[input_ids == video_token_id] = 2
    else:
        mm_token_type_ids = mm_token_type_ids.reshape(-1).to(device=device).long()
        if mm_token_type_ids.numel() != seq_len:
            raise ValueError("mm_token_type_ids and input_ids must have the same length")

    # Transformers splits videos into one grid per timestamp/frame.
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
        video_grid_thw[:, 0] = 1

    spatial_merge_size = int(processor.image_processor.merge_size)
    image_grid_thw = image_grid_thw.to(device=device) if image_grid_thw is not None else None
    video_grid_thw = video_grid_thw.to(device=device) if video_grid_thw is not None else None
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    valid_types = mm_token_type_ids[attention_mask]
    # A pure-text trajectory is common after a tool returns a final answer.
    if not torch.any(valid_types != 0):
        text_pos = valid_types.new_zeros((3, int(valid_types.numel())))
        if valid_types.numel():
            text_pos = torch.arange(valid_types.numel(), device=device, dtype=input_ids.dtype).view(1, -1).expand(3, -1)
        out = torch.zeros((3, seq_len), dtype=input_ids.dtype, device=device)
        out[:, attention_mask] = text_pos
        return out

    positions = []
    current_pos = 0
    for modality_type, group in itertools.groupby(enumerate(valid_types.tolist()), lambda item: item[1]):
        group = list(group)
        group_len = len(group)
        if modality_type == 0:
            positions.append(
                torch.arange(group_len, device=device, dtype=input_ids.dtype).view(1, -1).expand(3, -1) + current_pos
            )
            current_pos += group_len
            continue

        if modality_type not in grid_iters or grid_iters[modality_type] is None:
            raise ValueError(f"mm_token_type_ids contains modality {modality_type}, but no matching grid was supplied")
        try:
            grid_thw = next(grid_iters[modality_type])
        except StopIteration as exc:
            raise ValueError(f"more modality-{modality_type} groups than supplied image/video grids") from exc

        grid_t = int(grid_thw[0].item())
        grid_h = int(grid_thw[1].item()) // spatial_merge_size
        grid_w = int(grid_thw[2].item()) // spatial_merge_size
        if min(grid_t, grid_h, grid_w) <= 0:
            raise ValueError(f"invalid visual grid {grid_thw.tolist()}")

        visual_len = grid_t * grid_h * grid_w
        if visual_len != group_len:
            raise ValueError(
                "visual token/grid mismatch: "
                f"modality={modality_type}, token_group={group_len}, grid_tokens={visual_len}, grid={grid_thw.tolist()}"
            )

        # Same construction as Qwen3_5Model.get_vision_position_ids: temporal
        # indices start at zero, spatial indices are offset by current_pos.
        temporal, height, width = torch.meshgrid(
            torch.arange(grid_t, device=device, dtype=input_ids.dtype),
            torch.arange(grid_h, device=device, dtype=input_ids.dtype),
            torch.arange(grid_w, device=device, dtype=input_ids.dtype),
            indexing="ij",
        )
        vision_positions = torch.stack([temporal, height, width]).reshape(3, -1)
        # ``get_vision_position_ids`` offsets all three axes after creating the
        # grid (the temporal axis starts at zero before this offset).
        vision_positions += current_pos
        positions.append(vision_positions)
        current_pos += max(grid_h, grid_w)

    valid_positions = torch.cat(positions, dim=1)
    if valid_positions.shape[1] != int(valid_types.numel()):
        raise ValueError(
            f"mRoPE length mismatch: positions={valid_positions.shape[1]}, valid_tokens={valid_types.numel()}"
        )
    output = torch.zeros((3, seq_len), dtype=input_ids.dtype, device=device)
    output[:, attention_mask] = valid_positions
    return output
