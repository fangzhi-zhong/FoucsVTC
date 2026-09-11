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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.agent import agent_rollout_loop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _dedup_visual_tokens(prompt_token_ids: List[int], tokenizer) -> List[int]:
    """Collapse expanded Qwen image/video pad runs before sending to vLLM.

    HF's processor expands ``<|image_pad|>`` to every visual patch, while vLLM
    performs that expansion itself from ``multi_modal_data``.  Supplying both
    expansions shifts all subsequent token ids.  Raw chat-template prompts
    normally already contain one token, so this is a no-op in the common case.
    """

    visual_ids = set()
    for token in ("<|image_pad|>", "<|video_pad|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if isinstance(token_id, int) and token_id >= 0:
            visual_ids.add(token_id)
    if not visual_ids:
        return prompt_token_ids
    output = []
    previous_visual = False
    for token_id in prompt_token_ids:
        is_visual = token_id in visual_ids
        if is_visual and previous_visual:
            continue
        output.append(token_id)
        previous_visual = is_visual
    return output


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor, tokenizer=None) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return _dedup_visual_tokens(token_ids, tokenizer) if tokenizer is not None else token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), (
            "disable CUDA graph (enforce_eager = False) if free cache engine"
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp
                )
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        # Qwen3.5 keeps the language context length in ``text_config`` rather
        # than on the composite top-level config.  Older models still expose
        # the flat attribute.
        model_max_position_embeddings = getattr(
            model_hf_config,
            "max_position_embeddings",
            getattr(getattr(model_hf_config, "text_config", None), "max_position_embeddings", None),
        )
        if model_max_position_embeddings is not None:
            assert model_max_position_embeddings >= config.prompt_length + config.response_length, (
            "model context length should be greater than total sequence length"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and not self.config.enable_chunked_prefill:
            raise ValueError(
                "Without chunked prefill, max_num_batched_tokens must be at least max_model_len; "
                "increase max_num_batched_tokens or enable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            # limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=int(config.max_num_seqs),
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            # Keep the multimodal processor cache configurable from the
            # rollout config.  vLLM defaults this to 4 GiB; the Qwen3.5-VL
            # agent trajectory can accumulate original pages and zoom crops,
            # so the launcher may raise it without returning raw images
            # through DataProto/Ray.
            mm_processor_cache_gb=float(config.get("mm_processor_cache_gb", 4)),
            
            **({
                "limit_mm_per_prompt": dict(
                    image=self.config.agent.max_vllm_images, 
                    video=self.config.agent.max_vllm_videos,
                ),
            } if self.config.agent.activate_agent and self.config.agent.max_vllm_images else {})
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i], tokenizer=self.tokenizer) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )
            input_data["prompt_token_ids"] = _dedup_visual_tokens(
                input_data["prompt_token_ids"], self.tokenizer
            )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {"top_p": 1.0, "top_k": -1, "min_p": 0.0, "temperature": 0, "n": 1}
            # ``best_of`` was removed from vLLM 0.19's SamplingParams.
            if hasattr(self.sampling_params, "best_of"):
                kwargs["best_of"] = 1
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            if self.config.agent.activate_agent:
                agent_proto = agent_rollout_loop(
                    config=self.config,
                    vllm_engine=self.inference_engine,
                    vllm_inputs=vllm_inputs, 
                    prompts=prompts,
                    multi_modal_inputs=non_tensor_batch.get("multi_modal_inputs", None),
                    sampling_params=self.sampling_params
                )
                response = agent_proto.batch.pop('response')
                # The agent loop pads the complete trajectory to
                # ``prompt_length + response_length`` and returns exactly the
                # response tail.  Fail early with a useful configuration
                # error if data/rollout lengths were overridden
                # inconsistently; otherwise the later model forward would
                # receive mismatched input and mask widths.
                if idx.size(1) != self.config.prompt_length:
                    raise ValueError(
                        "agent prompt length mismatch: "
                        f"data input_ids={idx.size(1)} vs rollout.prompt_length={self.config.prompt_length}"
                    )
                if response.size(1) != self.config.response_length:
                    raise ValueError(
                        "agent response length mismatch: "
                        f"agent response={response.size(1)} vs rollout.response_length={self.config.response_length}"
                    )
            else:
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

                # TODO(sgm): disable logprob when recompute_log_prob is enable
                # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

                response = []
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        response.append(output.outputs[sample_id].token_ids)

                response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                    idx.device
                )

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                        non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                    )
                if "extra_info" in non_tensor_batch.keys():
                    non_tensor_batch["extra_info"] = _repeat_interleave(
                        non_tensor_batch["extra_info"], self.sampling_params.n
                    )

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        if 'raw_prompt' in non_tensor_batch.keys():
            non_tensor_batch.pop('raw_prompt')
        if 'multi_modal_data' in non_tensor_batch.keys():
            non_tensor_batch.pop('multi_modal_data')
        if 'origin_multi_modal_data' in non_tensor_batch.keys():
            non_tensor_batch.pop('origin_multi_modal_data', None)

        if self.config.agent.activate_agent:
            batch = batch.update(agent_proto.batch)
            non_tensor_batch.update(agent_proto.non_tensor_batch)
            tool_name_key = self.config.agent.tool_name_key
            if tool_name_key and tool_name_key in non_tensor_batch.keys():
                non_tensor_batch.pop(tool_name_key)
            print(f' [DEBUG agent output proto] {batch.keys()=}, {non_tensor_batch.keys()=}')

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
