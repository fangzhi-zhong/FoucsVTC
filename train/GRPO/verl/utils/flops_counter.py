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

import torch
from transformers import PretrainedConfig

VALID_CONFIG_TYPE = {"llama", "qwen2", "qwen2_vl", "qwen2_5_vl", "qwen3_5", "deepseek_v3"}


def get_device_flops(unit="T"):
    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device_name = torch.cuda.get_device_name()
    flops = float("inf")  # INF flops for unkown gpu type

    if "MI300X" in device_name:
        flops = 1336e12
    elif "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    flops_unit = unit_convert(flops, unit)
    return flops_unit


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        if config.model_type not in VALID_CONFIG_TYPE:
            print(
                f"Only support config type of {VALID_CONFIG_TYPE}, but got {config.model_type}. "
                f"MFU will always be zero."
            )

        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen2_vl": self._estimate_qwen2_flops,
            "qwen2_5_vl": self._estimate_qwen2_flops,
            "qwen3_5": self._estimate_qwen3_5_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
        }
        self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time, **kwargs):
        return 0

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time, **kwargs):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time, **kwargs):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts

        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts

        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
        # MLA attn
        attn_linear_N = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_N += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_N += hidden_size * self.config.q_lora_rank
            attn_linear_N += num_query_heads * q_head_dim * self.config.q_lora_rank

        attn_linear_N += hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_linear_N += (
            num_query_heads
            * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim)
            * self.config.kv_lora_rank
        )
        attn_linear_N += num_query_heads * self.config.v_head_dim * hidden_size
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        moe_N = (
            (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
            + (hidden_size * self.config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
            + emd_and_lm_head_N
        )
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * moe_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen * num_hidden_layers

        attn_qkv_flops = 12 * seqlen_square_sum * q_head_dim * num_query_heads
        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12

        return flops_achieved

    @staticmethod
    def _qwen3_5_layer_types(config):
        layer_types = getattr(config, "layer_types", None)
        if layer_types:
            return (
                sum(layer_type == "full_attention" for layer_type in layer_types),
                sum(layer_type == "linear_attention" for layer_type in layer_types),
            )
        interval = getattr(config, "full_attention_interval", 4)
        full = sum((idx + 1) % interval == 0 for idx in range(config.num_hidden_layers))
        return full, config.num_hidden_layers - full

    def _estimate_qwen3_5_flops(self, tokens_sum, batch_seqlens, delta_time, lm_head_tokens=None):
        """Estimate Qwen3.5 text FLOPs (the vision tower is excluded).

        Qwen3.5 alternates Gated DeltaNet and full-attention layers.  The
        coefficients follow the usual 6 FLOPs per trainable matrix parameter
        convention used by the other counters in this file.  This keeps the
        reported metric comparable with the existing MFU metrics while
        accounting for Qwen3.5's extra gates, convolution and recurrent rule.
        """
        config = getattr(self.config, "text_config", self.config)
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_layers = config.num_hidden_layers
        full_layers, linear_layers = self._qwen3_5_layer_types(config)
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)

        q_size = config.num_attention_heads * head_dim
        k_size = config.num_key_value_heads * head_dim
        v_size = config.num_key_value_heads * head_dim
        full_attn_linear = hidden_size * (2 * q_size + k_size + v_size + q_size)

        linear_k = config.linear_num_key_heads * config.linear_key_head_dim
        linear_v = config.linear_num_value_heads * config.linear_value_head_dim
        linear_attn_linear = hidden_size * (
            2 * linear_k + 3 * linear_v + 2 * config.linear_num_value_heads
        )
        conv = config.linear_conv_kernel_dim * (2 * linear_k + linear_v)
        attn_linear = full_attn_linear * full_layers + (linear_attn_linear + conv) * linear_layers

        # Qwen3.5-9B is dense SwiGLU. Keep the MoE branch for compatible
        # Qwen3.5 checkpoints without changing the dense model estimate.
        if hasattr(config, "num_experts"):
            mlp = (
                hidden_size * config.num_experts
                + hidden_size * config.moe_intermediate_size * config.num_experts_per_tok * 3
                + hidden_size * config.shared_expert_intermediate_size * 3
                + hidden_size
            )
            mlp *= num_layers
        else:
            mlp = hidden_size * config.intermediate_size * 3 * num_layers

        embedding = vocab_size * hidden_size
        lm_head_tokens = tokens_sum if lm_head_tokens is None else sum(lm_head_tokens)
        dense_flops = 6 * ((mlp + attn_linear + embedding) * tokens_sum + embedding * lm_head_tokens)
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
        full_attention_flops = 6 * seqlen_square_sum * head_dim * config.num_attention_heads * full_layers
        # Chunkwise Gated DeltaNet recurrence, including forward/backward.
        gdn_flops = (
            15
            * config.linear_key_head_dim
            * config.linear_value_head_dim
            * config.linear_num_value_heads
            * tokens_sum
            * linear_layers
        )
        flops_achieved = (dense_flops + full_attention_flops + gdn_flops) / delta_time / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens, delta_time, lm_head_tokens=None):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.
            lm_head_tokens (List[int], optional): Valid positions for the
                vocabulary projection. Used by Qwen3.5's response-only
                ``logits_to_keep`` path; other models ignore it.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time, lm_head_tokens=lm_head_tokens)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
