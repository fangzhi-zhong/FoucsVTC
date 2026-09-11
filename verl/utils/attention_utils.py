"""Padding helpers with an optional flash-attention fast path.

The original verl fork imports ``flash_attn.bert_padding`` at module import
time.  That makes otherwise valid SDPA runs fail on Python/CUDA combinations
for which a matching flash-attention wheel is not available (for example the
Qwen3.5 + torch 2.10 environment).  Keep the same public functions, use the
flash-attention implementation when it is importable, and fall back to
equivalent PyTorch operations otherwise.  The fallback is intentionally used
only for remove-padding paths; normal padded execution is unchanged.
"""

from typing import Callable

import torch
import torch.nn.functional as F

_FUNCTIONS = None


def _index_first_axis_fallback(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tensor[indices]


def _pad_input_fallback(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output[indices] = hidden_states
    return output.view(batch, seqlen, *hidden_states.shape[1:])


def _unpad_input_fallback(hidden_states: torch.Tensor, attention_mask: torch.Tensor, unused_mask=None):
    all_masks = attention_mask + unused_mask if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.reshape(-1), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return (
        _index_first_axis_fallback(flat, indices),
        indices,
        cu_seqlens,
        int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() else 0,
        used_seqlens_in_batch,
    )


def _rearrange_fallback(*args, **kwargs):
    from einops import rearrange as einops_rearrange

    return einops_rearrange(*args, **kwargs)


def _get_functions() -> tuple[Callable, Callable, Callable, Callable]:
    global _FUNCTIONS
    if _FUNCTIONS is not None:
        return _FUNCTIONS
    try:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

        _FUNCTIONS = (index_first_axis, pad_input, rearrange, unpad_input)
    except (ImportError, OSError):
        # ImportError covers an absent wheel; OSError covers a wheel compiled
        # against a different torch/CUDA ABI.  Both should use the safe path.
        _FUNCTIONS = (
            _index_first_axis_fallback,
            _pad_input_fallback,
            _rearrange_fallback,
            _unpad_input_fallback,
        )
    return _FUNCTIONS


def index_first_axis(*args, **kwargs):
    return _get_functions()[0](*args, **kwargs)


def pad_input(*args, **kwargs):
    return _get_functions()[1](*args, **kwargs)


def rearrange(*args, **kwargs):
    return _get_functions()[2](*args, **kwargs)


def unpad_input(*args, **kwargs):
    return _get_functions()[3](*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
