from typing import Iterable

import torch
from loguru import logger
from transformers import PretrainedConfig

from lmms_engine.utils import TrainUtilities

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


VALID_CONFIG_TYPE = {
    "llama",
    "qwen2",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen2_5_omni",
    "qwen2_5_omni_thinker",
    "qwen2_5_vl_mamba",
    "qwen3",
    "qwen3_dllm",
    "qwen3_moe",
    "qwen3_vl",
    "qwen3_5",
    "deepseek_v3",
    "minicpmv",
    "minicpmo",
    "llava_onevision",
}


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        if config.model_type not in VALID_CONFIG_TYPE:
            logger.warning(
                f"Only support config type of {VALID_CONFIG_TYPE}, but got {config.model_type}. MFU will not be counted."
            )

        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_qwen2_flops,
            "qwen2_5_vl": self._estimate_qwen2_flops,
            "qwen2_5_omni": self._estimate_qwen2_flops,
            "qwen2_5_omni_thinker": self._estimate_qwen2_flops,
            "qwen2_5_vl_mamba": self._estimate_qwen2_flops,
            "qwen3": self._estimate_qwen2_flops,
            "qwen3_dllm": self._estimate_qwen2_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "qwen3_vl": self._estimate_qwen2_flops,
            "qwen3_5": self._estimate_qwen3_5_flops,
            "spb2_vl": self._estimate_spb2vl_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "minicpmv": self._estimate_qwen2_flops,
            "minicpmo": self._estimate_qwen2_flops,
            "llava_onevision": self._estimate_qwen2_flops,
            "bagel": self._estimate_qwen2_flops,
        }
        if config.model_type in [
            "llava_onevision",
            "qwen3_vl",
            "qwen2_5_omni",
            "qwen2_5_omni_thinker",
            "qwen3_5",
        ]:
            self.config = config.text_config
            self.config.model_type = config.model_type
        elif config.model_type == "bagel":
            self.config = config.llm_config
        else:
            self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        return 0

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        config = self.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size

        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
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
    
    def _estimate_qwen3_5_flops(self, tokens_sum, batch_seqlens, delta_time):
        """Estimate Qwen3.5 training FLOPs for its hybrid decoder.

        Qwen3.5 alternates Gated DeltaNet (linear attention) and standard
        full-attention blocks. The generic Qwen2 estimate treats every block
        as full attention, which substantially over-counts the quadratic term
        at long sequence lengths. We count the per-token projections and MLP
        for each block type separately, and add the quadratic QK/AV work only
        for the full-attention blocks. Elementwise operations and the vision
        tower are omitted, matching the existing text-only MFU estimates.
        """
        config = self.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads

        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            full_attention_interval = getattr(config, "full_attention_interval", 4)
            layer_types = [
                "linear_attention" if (idx + 1) % full_attention_interval else "full_attention"
                for idx in range(num_hidden_layers)
            ]
        num_full_attention = sum(layer_type == "full_attention" for layer_type in layer_types)
        num_linear_attention = num_hidden_layers - num_full_attention

        # Qwen3.5 uses a gated SwiGLU MLP in every decoder block.
        mlp_n = hidden_size * intermediate_size * 3

        # Full attention: q_proj emits Q and a gating vector, so its output
        # width is 2 * q_size (unlike the older Qwen2 estimate).
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        full_attention_proj_n = hidden_size * (2 * q_size + k_size + v_size + q_size)

        # Gated DeltaNet projections from Qwen3_5GatedDeltaNet. The recurrent
        # state update is linear in sequence length and is intentionally kept
        # out of the parameter-based term; this mirrors the approximation used
        # for non-matmul kernels elsewhere in this counter.
        key_dim = config.linear_key_head_dim * config.linear_num_key_heads
        value_dim = config.linear_value_head_dim * config.linear_num_value_heads
        delta_net_proj_n = hidden_size * (
            2 * key_dim
            + value_dim
            + value_dim
            + config.linear_num_value_heads
            + config.linear_num_value_heads
        ) + value_dim * hidden_size

        embedding_and_lm_head_n = vocab_size * hidden_size * 2
        dense_n = (
            (mlp_n + full_attention_proj_n) * num_full_attention
            + (mlp_n + delta_net_proj_n) * num_linear_attention
            + embedding_and_lm_head_n
        )
        dense_flops = 6 * dense_n * tokens_sum

        # QK and AV are quadratic only in the full-attention layers. The
        # factor of 12 follows the convention used by _estimate_qwen2_flops
        # (forward + backward for both matrix products).
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
        full_attention_flops = (
            12 * seqlen_square_sum * head_dim * num_attention_heads * num_full_attention
        )

        flops_all_token = dense_flops + full_attention_flops
        return flops_all_token * (1.0 / delta_time) / 1e12

    def _estimate_spb2vl_flops(self, tokens_sum, batch_seqlens, delta_time):
        config = self.config.text_config
        v_config = self.config.vision_config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.attn.get("num_kv_heads", 8)
        num_attention_heads = config.num_heads
        intermediate_size = config.intermediate_size

        head_dim = getattr(
            config,
            "head_dim",
            hidden_size // num_attention_heads,
        )
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        
        attn_conf = getattr(config, 'attn', {})
        if not attn_conf:
             # Fallback to standard
             return self._estimate_qwen2_flops(tokens_sum, batch_seqlens, delta_time)

        moba_layers_indices = set(attn_conf.get('layers', []))
        full_layers_indices = set(attn_conf.get('full_layers', []))
        
        num_full = len(full_layers_indices)
        num_moba = len(moba_layers_indices) - num_full
        num_linear = num_hidden_layers - len(moba_layers_indices)

        mlp_N = hidden_size * intermediate_size * 3

        # moba, swa and full attn
        std_proj_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        
        swa_proj_N = std_proj_N

        sse_dim = num_attention_heads * head_dim
        
        sse_main_N = hidden_size * (sse_dim * 4) # i.e. the qkvs

        sse_lora_N = 2 * (hidden_size * head_dim + head_dim * sse_dim)

        sse_gate_ab_N = 2 * (hidden_size * num_attention_heads * 2)
        
        num_sparse_partition = config.num_sparse_partition

        sse_routing_N = hidden_size * num_sparse_partition

        sse_proj_N = swa_proj_N + sse_main_N + sse_lora_N + sse_gate_ab_N + sse_routing_N

        dense_N_standard = (mlp_N + std_proj_N) * (num_full + num_moba)
        dense_N_sse = (mlp_N + sse_proj_N) * num_linear
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        
        dense_N = dense_N_standard + dense_N_sse + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn flops
        
        moba_chunk = attn_conf.get('moba_chunk_size', 1024)
        moba_topk = attn_conf.get('moba_topk', 4)
        moba_window = moba_chunk * moba_topk
        
        swa_window = attn_conf.get('window_size', 1024)

        attn_flops = 0
        factor = 12 * head_dim * num_attention_heads

        for seqlen in batch_seqlens:
            # Full Attn: L * L
            if num_full > 0:
                attn_flops += factor * num_full * (seqlen * seqlen)
            
            # MoBA Attn: L * window, ignore the compression cost
            if num_moba > 0:
                effective_len = min(seqlen, moba_window)
                attn_flops += factor * num_moba * (seqlen * effective_len)

            # Linear (SSE/SWA): L * window + L * D
            if num_linear > 0:
                effective_len = min(seqlen, swa_window)
                # swa
                attn_flops += factor * num_linear * (seqlen * effective_len)
                # sse
                attn_flops += factor * num_linear * (seqlen * hidden_size * num_sparse_partition)

        flops_achieved = (dense_N_flops + attn_flops) * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time):
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

    def _estimate_qwen2_moe_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_topk = self.config.num_experts_per_tok
        num_experts = self.config.num_experts

        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # gate + moe export
        moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
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

    def estimate_flops(self, batch_seqlens, delta_time):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the
                current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        # Flatten batch_seqlens if it contains nested lists (common for vision models)
        if isinstance(batch_seqlens, (list, tuple)):
            flat_seqlens = []
            for item in batch_seqlens:
                if isinstance(item, (list, tuple)):
                    flat_seqlens.extend(item)
                else:
                    flat_seqlens.append(item)
            batch_seqlens = flat_seqlens
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        promised_flops = TrainUtilities.get_device_flops()
        return estimated_flops, promised_flops


def setup_flops_counter(config: PretrainedConfig) -> FlopsCounter:
    """
    Setup the FlopsCounter based on the provided configuration.

    Args:
        config (PretrainedConfig): The configuration object for the model.

    Returns:
        FlopsCounter: An instance of FlopsCounter initialized with the given configuration.
    """
    global flops_counter
    flops_counter = FlopsCounter(config)
