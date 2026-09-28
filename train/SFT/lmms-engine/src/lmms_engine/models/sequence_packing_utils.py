from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_outputs import BaseModelOutputWithPast


@dataclass
class BaseModelOutputWithPastAndRmpad(BaseModelOutputWithPast):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    seq_lens: Optional[torch.IntTensor] = None
    word_idx: Optional[torch.IntTensor] = None


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_unpad(q, k, cos, sin, position_ids):
    cos = cos.squeeze().index_select(dim=0, index=position_ids.squeeze()).unsqueeze(1)  # [total_bs_seq, 1, head_dim]
    sin = sin.squeeze().index_select(dim=0, index=position_ids.squeeze()).unsqueeze(1)

    # [total_bs_seq, head_num, head_dim] * [total_bs_seq, 1, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _unpad_input(input_ids, attention_mask):
    valid_mask = attention_mask.squeeze(1).squeeze(1).eq(1)
    seqlens_in_batch = valid_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(valid_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    input_ids = rearrange(input_ids, "b s ... -> (b s) ...")[indices]

    unpad_seq_len = input_ids.shape[0]

    return input_ids, indices, cu_seqlens, max_seqlen_in_batch
    

def _make_batch_and_pos(
    bsz: int,
    cu_seqlens: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    cu_seqlens = cu_seqlens[:bsz+1] # (bsz+1,)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]   # (bsz,)
    max_length = int(lengths.max().item())
    total_seq_len = int(cu_seqlens[-1].item())

    batch_idx = torch.repeat_interleave(
        torch.arange(bsz, device=device),
        lengths
    )  # (total_seq_len,)

    pos = torch.arange(total_seq_len, device=device) - torch.repeat_interleave(
        cu_seqlens[:-1],
        lengths
    )  # (total_seq_len,)

    return lengths, max_length, batch_idx, pos, total_seq_len


def _qkv_to_padded(
    x: torch.Tensor,
    bsz: int,
    cu_seqlens: torch.Tensor,
    pad_value=0,
) -> torch.Tensor:
    # x: (total_seq_len, n_heads, head_dim)
    _, max_length, batch_idx, pos, total_seq_len = _make_batch_and_pos(
        bsz, cu_seqlens, x.device
    )
    x = x[:total_seq_len]

    total_seq_len, n_heads, head_dim = x.shape
    out = x.new_full((bsz, max_length, n_heads, head_dim), pad_value)

    out[batch_idx, pos] = x
    return out


def _cos_sin_to_padded(
    cos: torch.Tensor,
    sin: torch.Tensor,
    bsz: int,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # cos/sin: (channels, 1, total_seq_len, head_dim)
    _, max_length, batch_idx, pos, total_seq_len = _make_batch_and_pos(
        bsz, cu_seqlens, cos.device
    )
    cos = cos[:, :, :total_seq_len]
    sin = sin[:, :, :total_seq_len]
    channels, _, total_seq_len, head_dim = cos.shape
    cos_out = cos.new_zeros((channels, bsz, max_length, head_dim))
    sin_out = sin.new_zeros((channels, bsz, max_length, head_dim))

    cos_src = cos[:, 0]   # (channels, total_seq_len, head_dim)
    sin_src = sin[:, 0]   # (channels, total_seq_len, head_dim)

    cos_out[:, batch_idx, pos, :] = cos_src
    sin_out[:, batch_idx, pos, :] = sin_src

    return cos_out, sin_out


def _dt_to_padded(
    dt: torch.Tensor,
    bsz: int,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    # dt: (total_seq_len, head_dim)
    _, max_length, batch_idx, pos, total_seq_len = _make_batch_and_pos(
        bsz, cu_seqlens, dt.device
    )
    dt = dt[:total_seq_len]
    total_seq_len, head_dim = dt.shape
    out = dt.new_zeros((bsz, max_length, head_dim))

    out[batch_idx, pos] = dt
    return out