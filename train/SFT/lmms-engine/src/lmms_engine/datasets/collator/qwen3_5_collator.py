import collections
from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from ...protocol import Processable


def _flatten_instances(instances: Sequence[Dict]) -> list[Dict]:
    if not instances:
        raise ValueError("Qwen3.5 collator received an empty batch.")
    if isinstance(instances[0], list):
        instances = [sample for pack in instances for sample in pack]
    if not instances:
        raise ValueError("Qwen3.5 collator received an empty packed batch.")
    return list(instances)


def _group_inputs(instances: Sequence[Dict]) -> collections.defaultdict:
    inputs = collections.defaultdict(list)
    for instance in instances:
        for key, value in instance.items():
            inputs[key].append(value)
    return inputs


def _merge_extra_inputs(inputs: dict, batch: dict) -> None:
    ignored_keys = {
        "attention_mask",
        "cu_seqlens",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
        "seq_idx",
    }
    for key, values in inputs.items():
        if key in ignored_keys:
            continue
        first = values[0]
        if isinstance(first, bool) or (
            isinstance(first, (int, float)) and not isinstance(first, torch.Tensor)
        ):
            batch[key] = first
        else:
            batch[key] = torch.cat(values, dim=0)


@dataclass
class Qwen3_5PackingCollator:
    """Build the complete padding-free metadata required by Qwen3.5.

    Qwen3.5 mixes full attention, Gated DeltaNet, and a causal convolution.
    ``cu_seq_lens_*`` isolates full attention and the DeltaNet recurrent state,
    while ``seq_idx`` resets the causal convolution at each sample boundary.
    """

    processor: Processable

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = _flatten_instances(instances)
        inputs = _group_inputs(instances)

        input_ids = inputs.pop("input_ids")
        lengths = torch.tensor(
            [tokens.shape[0] for tokens in input_ids],
            dtype=torch.int32,
            device=input_ids[0].device,
        )
        # torch.cumsum promotes integer inputs to int64 unless dtype is explicit;
        # FlashAttention and FLA both require int32 cumulative sequence lengths.
        cu_seq_lens = torch.nn.functional.pad(
            lengths.cumsum(0, dtype=torch.int32),
            (1, 0),
        )
        max_length = int(lengths.max().item())

        batch = {
            "input_ids": torch.cat(input_ids, dim=0).unsqueeze(0),
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": max_length,
            "max_length_k": max_length,
            "seq_idx": torch.repeat_interleave(
                torch.arange(len(input_ids), dtype=torch.int32, device=lengths.device),
                lengths,
            ).unsqueeze(0),
        }

        if "labels" in inputs:
            labels = torch.cat(inputs.pop("labels"), dim=0)
            # The first token of every flattened sample must not be predicted by
            # the last token of the preceding sample after the causal-LM shift.
            labels[cu_seq_lens[:-1].to(torch.long)] = -100
            batch["labels"] = labels.unsqueeze(0)

        if "position_ids" in inputs:
            # Each sample's processor-built multimodal RoPE starts at zero.
            # Concatenating on the sequence axis preserves the reset boundaries.
            position_ids = inputs.pop("position_ids")
            batch["position_ids"] = torch.cat(position_ids, dim=-1).unsqueeze(1)

        if "mm_token_type_ids" in inputs:
            mm_token_type_ids = inputs.pop("mm_token_type_ids")
            batch["mm_token_type_ids"] = torch.cat(mm_token_type_ids, dim=0).unsqueeze(0)

        _merge_extra_inputs(inputs, batch)
        return batch


@dataclass
class Qwen3_5VisionCollator:
    """Pad ordinary (non-packed) Qwen3.5 multimodal batches."""

    processor: Processable

    def _pad_1d(self, values, padding_value):
        if self.processor.tokenizer.padding_side == "left":
            values = [torch.flip(value, [0]) for value in values]
        padded = torch.nn.utils.rnn.pad_sequence(
            values,
            batch_first=True,
            padding_value=padding_value,
        )
        if self.processor.tokenizer.padding_side == "left":
            padded = torch.flip(padded, [1])
        return padded

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = _flatten_instances(instances)
        inputs = _group_inputs(instances)

        input_ids = inputs.pop("input_ids")
        padded_input_ids = self._pad_1d(
            input_ids,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attention_values = [torch.ones_like(value, dtype=torch.long) for value in input_ids]
        batch = {
            "input_ids": padded_input_ids,
            # Build this from lengths instead of token values: Qwen tokenizers can
            # legitimately use the padding token id inside an unpadded sequence.
            "attention_mask": self._pad_1d(attention_values, padding_value=0),
        }

        if "labels" in inputs:
            batch["labels"] = self._pad_1d(inputs.pop("labels"), padding_value=-100)
        if "mm_token_type_ids" in inputs:
            batch["mm_token_type_ids"] = self._pad_1d(
                inputs.pop("mm_token_type_ids"),
                padding_value=0,
            )
        if "position_ids" in inputs:
            position_ids = inputs.pop("position_ids")
            channels = [
                self._pad_1d([value[channel] for value in position_ids], padding_value=0)
                for channel in range(position_ids[0].shape[0])
            ]
            batch["position_ids"] = torch.stack(channels, dim=0)

        _merge_extra_inputs(inputs, batch)
        return batch
