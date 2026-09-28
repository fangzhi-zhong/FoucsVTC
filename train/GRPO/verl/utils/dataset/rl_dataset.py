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

import copy
import json
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.models.transformers.qwen_vl import get_processor_family, get_rope_index
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


def collate_fn(data_list: list[Optional[dict]]) -> Optional[dict]:
    data_list = [data for data in data_list if data is not None]
    if not data_list:
        return None
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        # Optional parallel high-resolution pages used by visual tools.  They
        # are kept out of the initial model input and attached to
        # ``origin_multi_modal_data`` so the agent rollout can access them
        # without changing the trainer's generation-batch schema.
        self.high_res_image_key = config.get("high_res_image_key", "high_res_images")
        self.video_key = config.get("video_key", "videos")
        self.tools_key = config.get("tools_key", "tools")
        self.tools_enabled_key = config.get("tools_enabled_key", "enable_tools")
        tools_schema_path = config.get("tools_schema_path")
        self.tools_schema_path = (
            os.path.abspath(os.path.expanduser(str(tools_schema_path))) if tools_schema_path else None
        )
        self.default_tools = self._load_tools_schema(self.tools_schema_path)
        self.max_prompt_length = config.get("max_prompt_length", 1024)

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.filter_overlong_prompts_method = config.get("filter_overlong_prompts_method", "exact")
        if self.filter_overlong_prompts_method not in ("exact", "image_size"):
            raise ValueError("filter_overlong_prompts_method must be 'exact' or 'image_size'")
        self.skip_overlong_prompts = config.get("skip_overlong_prompts", False)
        self._skipped_overlong_prompts = 0

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    @staticmethod
    def _load_tools_schema(tools_schema_path: Optional[str]) -> Optional[list[dict]]:
        if tools_schema_path is None:
            return None

        with open(tools_schema_path, encoding="utf-8") as schema_file:
            tools = json.load(schema_file)
        if not isinstance(tools, list) or not tools or not all(isinstance(tool, dict) for tool in tools):
            raise ValueError(
                f"tools schema must be a non-empty JSON list of objects: {tools_schema_path}"
            )
        return tools

    @staticmethod
    def _resolve_tools_for_example(example, tools_key, tools_enabled_key, default_tools):
        if tools_enabled_key and tools_enabled_key in example:
            enabled = example[tools_enabled_key]
            if enabled is not None and not isinstance(enabled, (bool, np.bool_)):
                raise ValueError(
                    f"{tools_enabled_key} must be a boolean or null, got {enabled!r}"
                )
            if enabled is not None and not bool(enabled):
                return None

        row_tools = example.get(tools_key) if tools_key else None
        if row_tools:
            return row_tools
        return default_tools

    def _resolve_tools(self, example: dict) -> Optional[list[dict]]:
        return self._resolve_tools_for_example(
            example,
            self.tools_key,
            self.tools_enabled_key,
            self.default_tools,
        )

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            original_size = len(self.dataframe)
            length_source = (
                "prompts by image dimensions + text tokens"
                if self.filter_overlong_prompts_method == "image_size" else "expanded prompts"
            )
            self.dataframe = self.dataframe.filter(
                self._is_prompt_within_limit,
                num_proc=self.num_workers if self.num_workers > 1 else None,
                desc=f"Filtering {length_source} longer than {self.max_prompt_length} tokens",
            )

            print(
                f"filter dataset len: {len(self.dataframe)} "
                f"(removed {original_size - len(self.dataframe)} prompts exceeding {self.max_prompt_length} tokens)"
            )

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def _render_prompt(self, row_dict: dict):
        """Render text and tool schemas without loading any image pixels."""
        tools = self._resolve_tools(row_dict)
        messages = self._build_messages(row_dict)
        template_kwargs = {"tools": tools} if tools else {}
        template = self.processor if self.processor is not None else self.tokenizer
        raw_prompt = template.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **template_kwargs
        )
        return raw_prompt, messages

    def _prepare_prompt(self, row_dict: dict):
        """Build unpadded model inputs for loading and optional exact filtering."""
        raw_prompt, messages = self._render_prompt(row_dict)
        multi_modal_data = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict[self.image_key]]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict[self.video_key]]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos,
                padding=False, truncation=False, return_tensors="pt",
            )
        else:
            model_inputs = self.tokenizer(
                raw_prompt, return_tensors="pt", add_special_tokens=False,
                padding=False, truncation=False,
            )

        return raw_prompt, messages, model_inputs, multi_modal_data

    def _is_prompt_within_limit(self, example: dict) -> bool:
        if self.filter_overlong_prompts_method == "image_size":
            from verl.utils.dataset.prompt_length import estimate_image_tokens

            if example.get(self.video_key):
                raise ValueError("image_size prompt filtering supports images; use 'exact' for video datasets")
            raw_prompt, _ = self._render_prompt(copy.deepcopy(example))
            length = len(self.tokenizer.encode(raw_prompt, add_special_tokens=False))
            if length > self.max_prompt_length:
                return False
            if self.processor is not None:
                for image in example.get(self.image_key, []):
                    # The rendered prompt already contains one special token
                    # per image. Replace it with the merged visual grid size.
                    length += estimate_image_tokens(image, self.processor.image_processor) - 1
                    if length > self.max_prompt_length:
                        return False
            return True

        # Message conversion and image helpers may mutate nested dictionaries;
        # filtering must preserve the original Parquet row for later loading.
        raw_prompt, _, model_inputs, _ = self._prepare_prompt(copy.deepcopy(example))
        if model_inputs["input_ids"].shape[-1] > self.max_prompt_length:
            return False
        # The rollout's raw text-token stream is also bounded in __getitem__.
        return len(self.tokenizer.encode(raw_prompt, add_special_tokens=False)) <= self.max_prompt_length

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        raw_prompt, messages, model_inputs, multi_modal_data = self._prepare_prompt(row_dict)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if self.skip_overlong_prompts and (
            input_ids.shape[-1] > self.max_prompt_length or len(raw_prompt_ids) > self.max_prompt_length
        ):
            self._skipped_overlong_prompts += 1
            if self._skipped_overlong_prompts <= 5 or self._skipped_overlong_prompts % 100 == 0:
                print(
                    f"Skipping overlong prompt index={item}: input_length={input_ids.shape[-1]}, "
                    f"raw_length={len(raw_prompt_ids)}, max_length={self.max_prompt_length} "
                    f"(skipped {self._skipped_overlong_prompts} in this worker)",
                    flush=True,
                )
            return None

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_raw_image

            origin_multi_modal_data = {}
            if self.image_key in row_dict:
                origin_multi_modal_data["image"] = [
                    process_raw_image(image) for image in row_dict.pop(self.image_key)
                ]
                if self.high_res_image_key and self.high_res_image_key in row_dict:
                    # Tool-only high-resolution pages remain lazy paths and
                    # do not contribute tokens to the initial prompt.
                    high_res_images = row_dict.pop(self.high_res_image_key)
                    if high_res_images is not None:
                        origin_multi_modal_data["high_res_image"] = list(high_res_images)
            row_dict.pop(self.video_key, None)

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict['origin_multi_modal_data'] = origin_multi_modal_data
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

            # Transformers 5.x Qwen3-VL/Qwen3.5 requires this sequence-aligned
            # modality stream for mRoPE.  It must undergo exactly the same
            # left-padding/truncation as input_ids and attention_mask.
            mm_token_type_ids = row_dict["multi_modal_inputs"].get("mm_token_type_ids")
            if mm_token_type_ids is not None:
                if not isinstance(mm_token_type_ids, torch.Tensor):
                    mm_token_type_ids = torch.as_tensor(mm_token_type_ids)
                if mm_token_type_ids.ndim == 1:
                    mm_token_type_ids = mm_token_type_ids.unsqueeze(0)
                mm_token_type_ids, _ = verl_F.postprocess_data(
                    input_ids=mm_token_type_ids,
                    attention_mask=torch.ones_like(mm_token_type_ids),
                    max_length=self.max_prompt_length,
                    pad_token_id=0,
                    left_pad=True,
                    truncation=self.truncation,
                )
                row_dict["multi_modal_inputs"]["mm_token_type_ids"] = mm_token_type_ids

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if get_processor_family(self.processor) is not None:
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    mm_token_type_ids=(
                        row_dict.get("multi_modal_inputs", {}).get("mm_token_type_ids", None)
                    ),
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
