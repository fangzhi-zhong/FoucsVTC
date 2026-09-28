import collections
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

from ...protocol import Processable
from ...utils.train_utils import TrainUtilities

from transformers import AutoProcessor


@dataclass
class PackingCollator:
    processor: Processable

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        packing_inputs = {}
        if "input_ids" in inputs.keys():
            input_ids = inputs.pop("input_ids")
            cu_seqlens = torch.cat(
                [
                    torch.zeros(1, dtype=input_ids[0].dtype, device=input_ids[0].device),
                    torch.tensor([t.shape[0] for t in input_ids]).cumsum(0),
                ],
                dim = 0
            ).to(torch.int32)
            # print(cu_seqlens)
            packing_inputs["cu_seqlens"] = cu_seqlens
            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            # print(f"input_ids: {input_ids}")
            # processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
            # input_text = processor.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            # print(f"input_text: {input_text}")
            packing_inputs["input_ids"] = input_ids
        if "labels" in inputs.keys():
            labels = inputs.pop("labels")
            labels = torch.cat(labels, dim=0).unsqueeze(0)
            packing_inputs["labels"] = labels

        packing_inputs["attention_mask"] = None

        # for the other keys
        for key, values in inputs.items():
            # Handle scalar/boolean values ( use_audio_in_video)
            if isinstance(values[0], bool) or (
                isinstance(values[0], (int, float)) and not isinstance(values[0], torch.Tensor)
            ):
                packing_inputs[key] = values[0]
            else:
                packing_inputs[key] = torch.concatenate(values, dim=0)

        return packing_inputs


    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
