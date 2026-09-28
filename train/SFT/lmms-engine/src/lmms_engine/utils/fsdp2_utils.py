# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LinearLR
from copy import deepcopy

def apply_fsdp2(model, fsdp_kwargs, fsdp_transformer_layer_cls_to_wrap=None, fp32_layer_cls_to_wrap=None):
    """model: AutoModelForCausalLM"""

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = (
        default_transformer_cls_names_to_wrap
        if fsdp_transformer_layer_cls_to_wrap is None
        else fsdp_transformer_layer_cls_to_wrap
    )

    # print("fsdp_transformer_layer_cls_to_wrap:", fsdp_transformer_layer_cls_to_wrap)

    # wrap fp32 layers first
    if fp32_layer_cls_to_wrap is not None:
        if isinstance(fp32_layer_cls_to_wrap, str):
            fp32_layer_cls_to_wrap = [fp32_layer_cls_to_wrap]

        assert len(fp32_layer_cls_to_wrap) > 0 and fp32_layer_cls_to_wrap[0] is not None

        modules = []
        modules_out_fp16 = []
        for name, module in model.named_modules():
            if module.__class__.__name__ in fp32_layer_cls_to_wrap:
                modules.append(module)
            elif module.__class__.__name__ == "RMSNorm" and not("input_layernorm" in name or "post_attention_layernorm" in name):
                module = module.to(torch.float32)
                modules.append(module)
        
        fp32_fsdp_kwargs = fsdp_kwargs.copy()
        fp32_fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        )

        for idx, module in enumerate(modules):
            fully_shard(module, **fp32_fsdp_kwargs)

        # from torch.distributed.tensor import DTensor
        # def is_dt(x): return isinstance(x, DTensor) or hasattr(x, "to_local")

        # cnt = 0
        # for n,m in model.named_modules():
        #     if hasattr(m, "fp32_params"):
        #         cnt += 1
        #         print("HAS fp32_params:", n,
        #             "dt_bias is_dt=", is_dt(m.fp32_params.dt_bias),
        #             "A_log is_dt=", is_dt(m.fp32_params.A_log))
        # print("layers with fp32_params =", cnt)
    
    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    fsdp_kwargs_copy = deepcopy(fsdp_kwargs)
    # fsdp_kwargs_copy["reshard_after_forward"] = True
    for idx, module in enumerate(modules):
        fully_shard(module, **fsdp_kwargs_copy)
    
    fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    # To broadcast, it needs to be instantiated in the GPU.
    if dist.get_rank() == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        model = model.to_empty(device=torch.cuda.current_device())

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True)
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(torch.cuda.current_device())


"""
Optimizer related
"""


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    min_lr_ratio = 0.0 if min_lr_ratio is None else min_lr_ratio
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * (float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(min_lr_ratio, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_wsd_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    stable_ratio: float = 0.9,
):
    """
    Create a Warmup-Stable-Decay learning rate scheduler.

    The schedule follows three phases:
    1. Warmup: Learning rate increases linearly from 0 to the initial LR
    2. Stable: Learning rate remains constant at the initial LR
    3. Decay: Learning rate decreases following a cosine curve to min_lr_ratio * initial LR

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum learning rate ratio w.r.t the initial learning rate.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule during decay phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        stable_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The ratio of non-warmup steps that should maintain a constant learning rate.
            Set to 0.0 to behave exactly like cosine schedule.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    remaining_steps = max(0, num_training_steps - num_warmup_steps)
    num_stable_steps = int(remaining_steps * stable_ratio)
    num_decay_steps = remaining_steps - num_stable_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        if current_step < num_training_steps:
            progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
            value = max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )
            return (1.0 - min_lr_ratio) * value + min_lr_ratio
        return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    start_factor: float = 0.01,
    end_factor: float = 1,
):
    return LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=num_warmup_steps,
    )


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """Clip grad norm for FSDP2 / DTensor parameters.

    torch.nn.utils.clip_grad_norm_ cannot run on CPU-backed or sharded DTensors directly.
    Only trainable parameters with a non-None grad are included, which avoids NaN norms
    under partial training (e.g. gate-only) where most FSDP shards have no gradient.
    """
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = tuple()

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)

    params_with_grad = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params_with_grad:
        return torch.zeros((), device=torch.cuda.current_device())

    grads = [p.grad for p in params_with_grad]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    total_norm = total_norm.to(torch.cuda.current_device(), non_blocking=True)
    _clip_grads_with_norm_(params_with_grad, max_norm, total_norm, foreach)
    return total_norm
