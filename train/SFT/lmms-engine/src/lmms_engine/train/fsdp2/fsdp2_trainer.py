import fnmatch
import gc
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from functools import partial
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import send_to_device
from loguru import logger
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.utils.data import Dataset, DistributedSampler, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
from transformers.trainer_utils import seed_worker

import lmms_engine.models.utils as model_utils
import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.train.config import TrainingArguments
from lmms_engine.train.registry import TRAINER_REGISTER
from lmms_engine.utils import TrainUtilities
from lmms_engine.utils.fsdp2_utils import (
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)
from lmms_engine.utils.profiler import StepProfiler
from lmms_engine.utils.tracking import Tracking

DatasetType = Union[Dataset, IterableDataset]


def _module_lr_pattern_matches(parameter_name: str, pattern: str) -> bool:
    """Match a parameter name by module prefix or shell-style glob."""
    if any(character in pattern for character in "*?["):
        return fnmatch.fnmatchcase(parameter_name, pattern)
    return parameter_name == pattern or parameter_name.startswith(f"{pattern}.")


def _get_batch_sequence_lengths(batch):
    """Return per-sample lengths for padded and supported packed batches."""
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        return attention_mask.sum(dim=1).detach(), attention_mask.shape[0]

    cu_seqlens = batch.get("cu_seqlens", None)
    if cu_seqlens is None:
        # Qwen3.5 uses the native Transformers name because these boundaries
        # are shared by full attention and its recurrent DeltaNet layers.
        cu_seqlens = batch.get("cu_seq_lens_q", None)
    if cu_seqlens is not None:
        return (cu_seqlens[1:] - cu_seqlens[:-1]).detach(), cu_seqlens.shape[0] - 1

    raise ValueError(
        "attention_mask, cu_seqlens, or cu_seq_lens_q must be present in the "
        "batch to compute sequence lengths"
    )


@TRAINER_REGISTER.register("fsdp2_trainer")
class FSDP2SFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: DatasetType,
        eval_dataset: DatasetType = None,
        processing_class=None,
        data_collator=None,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.data_collator = data_collator
        self.default_backend = []
        if "wandb" in self.args.report_to:
            self.default_backend.append("wandb")
        self.default_backend.append("console")

        # Optional per-step PyTorch profiler configuration
        self.enable_profiler = self.args.enable_profiler
        self.profiler_config = self.args.profiler_config
        self.profiler_dir = os.path.join(self.args.output_dir, "profiler")
        self.step_profiler = StepProfiler(
            enable=self.enable_profiler,
            directory=self.profiler_dir,
            profiler_config=self.profiler_config,
            rank=dist.get_rank(),
        )
        # Initialize gradient accumulation counter
        self.gradient_accumulation_step = 0
        self.prefetch_stream = torch.cuda.Stream()
        # Gradient/parameter stats logging (compact JSONL, optionally rank0-only)
        self.grad_log_enabled = bool(getattr(self.args, "grad_log_enabled", False))
        self.grad_log_rank0_only = bool(getattr(self.args, "grad_log_rank0_only", True))
        self.grad_log_every_n_steps = int(getattr(self.args, "grad_log_every_n_steps", 1) or 1)
        self.grad_log_every_n_steps = max(1, self.grad_log_every_n_steps)
        self.grad_log_groups = bool(getattr(self.args, "grad_log_groups", True))
        patterns = getattr(self.args, "grad_log_param_patterns", None)
        if isinstance(patterns, str):
            patterns = [patterns]
        self.grad_log_param_patterns = patterns

        grad_log_dir = getattr(self.args, "grad_log_dir", None)
        self.grad_log_dir = grad_log_dir or os.path.join(self.args.output_dir, "grad_logs")
        # Previous-step stats cache, keyed by group name or param name
        self._grad_log_prev_stats: dict[str, dict[str, float]] = {}
        self._optimizer_step = 0

    def prepare_dataloader(self, dataset: DatasetType, is_training: bool = True):
        data_collator = self.data_collator
        # print("="*80)
        # print("train_batch_size", self.args.train_batch_size)
        # print("n_gpu", self.args.n_gpu)
        # print("="*80)
        dataloader_params = {
            "batch_size": self.args.train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if isinstance(dataset, IterableDataset):
            sampler = None
        elif getattr(self.args, "group_by_length", False):
            sampler = DistributedLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=dataset,
                lengths=dataset.modality_length,
                model_input_name=None,
                num_replicas=pgm.process_group_manager.dp_world_size,
                rank=pgm.process_group_manager.dp_rank,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=pgm.process_group_manager.dp_world_size,
                rank=pgm.process_group_manager.dp_rank,
            )
        dataloader_params["sampler"] = sampler
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        if is_training:
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=pgm.process_group_manager.dp_rank,
            )
        dataloader = StatefulDataLoader(dataset, **dataloader_params)
        return dataloader

    def prepare_model(self):
        if self.args.bf16:
            param_dtype = torch.bfloat16
        else:
            param_dtype = torch.float16

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        reduce_dtype = getattr(torch, self.args.reduce_dtype)
        output_dtype = getattr(torch, self.args.output_dtype)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=output_dtype,
        )

        fsdp_kwargs = {
            "reshard_after_forward": getattr(self.args, "fsdp_config", {}).get("reshard_after_forward", True),
            "mp_policy": mp_policy,
        }
        
        # Create FSDP2 process group for 8-card sharding groups
        # This reduces communication overhead by sharding within groups instead of across all GPUs
        mesh_group_size = self.args.fsdp_config.get("mesh_group_size", None)
        logger.info(f"mesh_group_size is {mesh_group_size}")
        logger.info(f"fsdp_config is {self.args.fsdp_config}")
        if mesh_group_size is not None and mesh_group_size > 0:
            world_size = dist.get_world_size()
            if world_size % mesh_group_size == 0:
                num_groups = world_size // mesh_group_size
                fsdp_kwargs["mesh"] = init_device_mesh("cuda", mesh_shape=(num_groups, mesh_group_size), mesh_dim_names=("replicate", "shard"))
                logger.info(f"FSDP2 sharding within groups of {mesh_group_size} GPUs with group {num_groups}")
            else:
                logger.warning(f"world_size ({world_size}) is not divisible by mesh_group_size ({mesh_group_size}), using default sharding")

        fp32_layer_cls_to_wrap = self.args.fsdp_config.get("fp32_layer_cls_to_wrap", None)
        transformer_cls_names_to_wrap = self.args.fsdp_config.get("transformer_layer_cls_to_wrap", None)
        full_state = self.model.state_dict()
        logger.info(f"Applying FSDP2 to model")
        # print("transformer_cls_names_to_wrap:", transformer_cls_names_to_wrap)
        apply_fsdp2(self.model, fsdp_kwargs, transformer_cls_names_to_wrap, fp32_layer_cls_to_wrap)
        logger.info(f"Loading full state dict to model")
        fsdp2_load_full_state_dict(self.model, full_state)
        logger.info(f"FSDP2 applied to model")
        self.fsdp2_model = self.model

    def prepare_optimizer(self):
        named_trainable_params = [
            (name, param)
            for name, param in self.fsdp2_model.named_parameters()
            if param.requires_grad
        ]
        if not named_trainable_params:
            raise ValueError("No trainable parameters found for the optimizer.")

        trainable_param_count = sum(
            param.numel() for _, param in named_trainable_params
        )
        logger.info(
            f"Optimizer trainable params: {trainable_param_count / 1e6:.2f}M"
        )

        raw_module_learning_rates = (
            getattr(self.args, "module_learning_rates", None) or {}
        )
        if not isinstance(raw_module_learning_rates, dict):
            raise TypeError(
                "module_learning_rates must be a mapping of "
                "parameter-name patterns to learning rates."
            )

        module_lr_rules = []
        for pattern, raw_learning_rate in raw_module_learning_rates.items():
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(
                    "module_learning_rates patterns must be non-empty strings."
                )
            if isinstance(raw_learning_rate, bool):
                raise ValueError(
                    f"Learning rate for pattern {pattern!r} must be a "
                    "non-negative finite number."
                )
            try:
                learning_rate = float(raw_learning_rate)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Learning rate for pattern {pattern!r} must be a "
                    "non-negative finite number."
                ) from exc
            if learning_rate < 0 or not math.isfinite(learning_rate):
                raise ValueError(
                    f"Learning rate for pattern {pattern!r} must be a "
                    "non-negative finite number."
                )
            module_lr_rules.append((pattern, learning_rate))

        if module_lr_rules:
            default_params = []
            rule_params = [[] for _ in module_lr_rules]
            rule_match_counts = [0 for _ in module_lr_rules]

            for name, param in named_trainable_params:
                selected_rule = None
                for rule_index, (pattern, _) in enumerate(module_lr_rules):
                    if _module_lr_pattern_matches(name, pattern):
                        rule_match_counts[rule_index] += 1
                        selected_rule = rule_index
                if selected_rule is None:
                    default_params.append(param)
                else:
                    rule_params[selected_rule].append(param)

            unmatched_patterns = [
                pattern
                for (pattern, _), match_count in zip(
                    module_lr_rules, rule_match_counts
                )
                if match_count == 0
            ]
            if unmatched_patterns:
                raise ValueError(
                    "module_learning_rates patterns matched no trainable "
                    "parameters: "
                    + ", ".join(repr(pattern) for pattern in unmatched_patterns)
                )

            optimizer_groups = []
            if default_params:
                optimizer_groups.append(
                    {"params": default_params, "lr": self.args.learning_rate}
                )
                logger.info(
                    "Optimizer LR group: default lr={} params={} ({:.2f}M)",
                    self.args.learning_rate,
                    len(default_params),
                    sum(param.numel() for param in default_params) / 1e6,
                )

            for (pattern, learning_rate), params, match_count in zip(
                module_lr_rules, rule_params, rule_match_counts
            ):
                if not params:
                    logger.warning(
                        "Optimizer LR rule {!r} matched {} parameters, but all "
                        "were overridden by later rules.",
                        pattern,
                        match_count,
                    )
                    continue
                optimizer_groups.append(
                    {"params": params, "lr": learning_rate}
                )
                logger.info(
                    "Optimizer LR group: pattern={!r} lr={} params={} ({:.2f}M)",
                    pattern,
                    learning_rate,
                    len(params),
                    sum(param.numel() for param in params) / 1e6,
                )
        else:
            optimizer_groups = [
                {
                    "params": [param for _, param in named_trainable_params],
                    "lr": self.args.learning_rate,
                }
            ]

        optim_name = getattr(
            getattr(self.args, "optim", "adamw_torch"),
            "value",
            getattr(self.args, "optim", "adamw_torch"),
        )
        if optim_name not in {"adamw_torch", "adamw_torch_fused"}:
            raise ValueError(
                "FSDP2SFTTrainer currently supports optim='adamw_torch' and "
                f"optim='adamw_torch_fused'; got {optim_name!r}."
            )

        optimizer_kwargs = {
            "lr": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }
        if optim_name == "adamw_torch_fused":
            optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(optimizer_groups, **optimizer_kwargs)
        logger.info(
            "Using optimizer {} (fused={})",
            optim_name,
            bool(self.optimizer.defaults.get("fused", False)),
        )
        # lm_parameters = list(self.fsdp2_model.model.parameters()) + list(self.fsdp2_model.lm_head.parameters())
        # merge_parameters = []
        # visual_parameters = []
        # for name, param in self.fsdp2_model.visual.named_parameters():
        #     if "merge" in name:
        #         merge_parameters.append(param)
        #     else:
        #         visual_parameters.append(param)
        # lm_lr = 5e-5
        # merge_lr = 5e-5
        # ve_lr = 1e-5
        # self.optimizer = torch.optim.AdamW(
        #     [
        #         {"params": lm_parameters, "lr": lm_lr},
        #         {"params": merge_parameters, "lr": merge_lr},
        #         {"params": visual_parameters, "lr": ve_lr},
        #     ],
        #     weight_decay=self.args.weight_decay,
        #     betas=(self.args.adam_beta1, self.args.adam_beta2),
        #     eps=self.args.adam_epsilon,
        # )

    def prepare_scheduler(
        self,
        num_warmup_steps: int,
        num_training_steps: int,
    ):
        self.args.lr_scheduler_kwargs = self.args.lr_scheduler_kwargs or {}
        if self.args.lr_scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **self.args.lr_scheduler_kwargs,
            )
        elif self.args.lr_scheduler_type == "wsd":
            self.scheduler = get_wsd_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **self.args.lr_scheduler_kwargs,
            )
        elif self.args.lr_scheduler_type == "constant":
            self.scheduler = get_constant_schedule(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                **self.args.lr_scheduler_kwargs,
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {self.args.lr_scheduler_type}")

    def compute_loss(self, batch):
        if self.args.bf16:
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float16
        with torch.autocast(device_type="cuda", dtype=cast_dtype):
            outputs = self.fsdp2_model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return loss

    def training_step(self, batch):
        self.fsdp2_model.train()
        
        # Only zero gradients at the start of accumulation
        if self.gradient_accumulation_step == 0:
            self.optimizer.zero_grad()
        
        loss = self.compute_loss(batch)
        if dist.get_world_size() > 1:
            loss = loss.mean()
        # Scale loss by gradient accumulation steps to maintain correct gradient magnitude
        loss = loss / self.args.gradient_accumulation_steps
        loss_item_pre_step = loss.item()
        loss.backward()
        
        self.gradient_accumulation_step += 1
        
        # Only update optimizer and scheduler when accumulation is complete
        should_update = self.gradient_accumulation_step >= self.args.gradient_accumulation_steps
        
        if should_update:
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp2_model.parameters(), self.args.max_grad_norm)
            self._optimizer_step += 1

            # if grad_norm is not finite, skip the update
            did_step = False
            if not torch.isfinite(grad_norm):
                print(f"WARN: grad_norm is not finite: {grad_norm}")
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
                did_step = True

            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            grad_norm_value = grad_norm.item()

            if self.grad_log_enabled and (self._optimizer_step % self.grad_log_every_n_steps == 0):
                self._log_gradients(grad_norm=grad_norm_value, lr=lr, did_step=did_step)

            self.gradient_accumulation_step = 0
        else:
            # Use previous values when not updating
            lr = self.scheduler.get_last_lr()[0]
            grad_norm_value = 0.0

        # reduce loss across dp ranks (scale back to original magnitude for logging)
        # loss_item = torch.tensor(loss_item * self.args.gradient_accumulation_steps, device=self.args.device)
        # print("="*80)
        # print("self.n_gpu after self.args.device(setup_devices):", self.args.n_gpu)
        # print("="*80)
        # if should_update:
        #     torch.distributed.all_reduce(loss_item, op=torch.distributed.ReduceOp.AVG)
        return {
            "train/loss": loss_item_pre_step,
            "train/lr": lr,
            "train/grad_norm": grad_norm_value,
            "should_update": should_update,  # Flag indicating if optimizer step was performed
        }

    @staticmethod
    def _grad_log_group_name(param_name: str) -> str:
        """
        Create a stable, readable group name for a parameter name.

        Preference is per-layer grouping for common transformer naming patterns:
        - "...layers.<idx>...." -> group up to layers.<idx>
        - "...blocks.<idx>...." -> group up to blocks.<idx>
        Otherwise, fall back to first 2 components.
        """
        parts = param_name.split(".")
        for i in range(len(parts) - 1):
            if parts[i] in ("layers", "layer", "blocks", "block") and parts[i + 1].isdigit():
                return ".".join(parts[: i + 2])
        return ".".join(parts[:2]) if len(parts) >= 2 else param_name

    def _should_log_param(self, param_name: str) -> bool:
        """Return True if this parameter should be logged individually."""
        if not self.grad_log_param_patterns:
            return False
        return any(pat in param_name for pat in self.grad_log_param_patterns)

    def _log_gradients(self, *, grad_norm: float | None, lr: float | None, did_step: bool) -> None:
        """Write compact gradient/parameter stats as JSONL (rank0 optionally aggregates)."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        os.makedirs(self.grad_log_dir, exist_ok=True)

        device = getattr(self.fsdp2_model, "device", None)
        if device is None:
            device = next(self.fsdp2_model.parameters()).device

        # Calculate effective step: global_step * accumulation_steps + current accumulation_step
        effective_step = getattr(self, "global_step", 0) * self.args.gradient_accumulation_steps + self.gradient_accumulation_step

        # Build local per-group stats (deterministic group keys across ranks).
        # Important: keep accumulation on GPU tensors to avoid per-param .item() syncs.
        group_stats: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "numel": 0,
                "grad_numel": 0,
                "param_sqsum": torch.zeros((), device=device, dtype=torch.float64),
                "grad_sqsum": torch.zeros((), device=device, dtype=torch.float64),
                "param_abs_sum": torch.zeros((), device=device, dtype=torch.float64),
                "grad_abs_sum": torch.zeros((), device=device, dtype=torch.float64),
                "param_abs_max": torch.zeros((), device=device, dtype=torch.float32),
                "grad_abs_max": torch.zeros((), device=device, dtype=torch.float32),
                "param_has_nan": torch.zeros((), device=device, dtype=torch.int32),
                "param_has_inf": torch.zeros((), device=device, dtype=torch.int32),
                "grad_has_nan": torch.zeros((), device=device, dtype=torch.int32),
                "grad_has_inf": torch.zeros((), device=device, dtype=torch.int32),
            }
        )
        # Optional: per-parameter stats for selected parameters (matched by name).
        param_stats: dict[str, dict[str, float | int | bool]] = {}

        for name, param in self.fsdp2_model.named_parameters():
            group = self._grad_log_group_name(name)
            gs = group_stats[group]
            log_this_param = self._should_log_param(name)
            ps = None
            if log_this_param:
                ps = param_stats.setdefault(
                    name,
                    {
                        "numel": 0,
                        "grad_numel": 0,
                        "param_sqsum": 0.0,
                        "grad_sqsum": 0.0,
                        "param_abs_sum": 0.0,
                        "grad_abs_sum": 0.0,
                        "param_abs_max": 0.0,
                        "grad_abs_max": 0.0,
                        "param_has_nan": False,
                        "param_has_inf": False,
                        "grad_has_nan": False,
                        "grad_has_inf": False,
                        "param_dtype": None,
                        "grad_dtype": None,
                    },
                )

            param_data = param.data
            param_local = param_data.to_local() if isinstance(param_data, DTensor) else param_data

            if param_local.numel() > 0:
                p32 = param_local.detach().float()
                gs["numel"] = int(gs["numel"]) + int(p32.numel())
                gs["param_sqsum"] = gs["param_sqsum"] + torch.sum(p32 * p32, dtype=torch.float64)
                gs["param_abs_sum"] = gs["param_abs_sum"] + torch.sum(torch.abs(p32), dtype=torch.float64)
                gs["param_abs_max"] = torch.maximum(gs["param_abs_max"], torch.max(torch.abs(p32)))
                gs["param_has_nan"] = torch.maximum(gs["param_has_nan"], torch.isnan(p32).any().to(torch.int32))
                gs["param_has_inf"] = torch.maximum(gs["param_has_inf"], torch.isinf(p32).any().to(torch.int32))
                if ps is not None:
                    if ps["param_dtype"] is None:
                        ps["param_dtype"] = str(param_local.dtype)
                    ps["numel"] += int(p32.numel())
                    sqsum = float(torch.sum(p32 * p32).item())
                    abssum = float(torch.sum(torch.abs(p32)).item())
                    absmax = float(torch.max(torch.abs(p32)).item())
                    ps["param_sqsum"] += sqsum
                    ps["param_abs_sum"] += abssum
                    ps["param_abs_max"] = max(ps["param_abs_max"], absmax)
                    ps["param_has_nan"] = bool(ps["param_has_nan"]) or bool(torch.isnan(p32).any().item())
                    ps["param_has_inf"] = bool(ps["param_has_inf"]) or bool(torch.isinf(p32).any().item())

            if param.grad is None:
                continue

            grad_data = param.grad
            grad_local = grad_data.to_local() if isinstance(grad_data, DTensor) else grad_data
            if grad_local.numel() == 0:
                continue

            g32 = grad_local.detach().float()
            gs["grad_numel"] = int(gs["grad_numel"]) + int(g32.numel())
            gs["grad_sqsum"] = gs["grad_sqsum"] + torch.sum(g32 * g32, dtype=torch.float64)
            gs["grad_abs_sum"] = gs["grad_abs_sum"] + torch.sum(torch.abs(g32), dtype=torch.float64)
            gs["grad_abs_max"] = torch.maximum(gs["grad_abs_max"], torch.max(torch.abs(g32)))
            gs["grad_has_nan"] = torch.maximum(gs["grad_has_nan"], torch.isnan(g32).any().to(torch.int32))
            gs["grad_has_inf"] = torch.maximum(gs["grad_has_inf"], torch.isinf(g32).any().to(torch.int32))
            if ps is not None:
                if ps["grad_dtype"] is None:
                    ps["grad_dtype"] = str(grad_local.dtype)
                ps["grad_numel"] += int(g32.numel())
                sqsum = float(torch.sum(g32 * g32).item())
                abssum = float(torch.sum(torch.abs(g32)).item())
                absmax = float(torch.max(torch.abs(g32)).item())
                ps["grad_sqsum"] += sqsum
                ps["grad_abs_sum"] += abssum
                ps["grad_abs_max"] = max(ps["grad_abs_max"], absmax)
                ps["grad_has_nan"] = bool(ps["grad_has_nan"]) or bool(torch.isnan(g32).any().item())
                ps["grad_has_inf"] = bool(ps["grad_has_inf"]) or bool(torch.isinf(g32).any().item())

        # Materialize per-group stats.
        # - If rank0_only: reduce to rank0 so results reflect all shards.
        # - Else: keep local per-rank stats (write per-rank JSONL).
        reduced_group_stats: dict[str, dict[str, float | int | bool]] = {}
        reduced_param_stats: dict[str, dict[str, float | int | bool]] = {}
        if self.grad_log_rank0_only:
            for group in sorted(group_stats.keys()):
                gs = group_stats[group]

                sum_t = torch.stack(
                    [
                        gs["param_sqsum"],
                        gs["grad_sqsum"],
                        gs["param_abs_sum"],
                        gs["grad_abs_sum"],
                        torch.tensor(float(gs["numel"]), device=device, dtype=torch.float64),
                        torch.tensor(float(gs["grad_numel"]), device=device, dtype=torch.float64),
                    ]
                )
                max_t = torch.stack([gs["param_abs_max"], gs["grad_abs_max"]])
                flags_t = torch.stack([gs["param_has_nan"], gs["param_has_inf"], gs["grad_has_nan"], gs["grad_has_inf"]])

                if world_size > 1:
                    dist.reduce(sum_t, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(max_t, dst=0, op=dist.ReduceOp.MAX)
                    dist.reduce(flags_t, dst=0, op=dist.ReduceOp.SUM)

                if rank == 0:
                    param_sqsum, grad_sqsum, param_abs_sum, grad_abs_sum, numel_f, grad_numel_f = sum_t.tolist()
                    param_abs_max, grad_abs_max = max_t.tolist()
                    p_nan, p_inf, g_nan, g_inf = flags_t.tolist()
                    reduced_group_stats[group] = {
                        "numel": int(numel_f),
                        "grad_numel": int(grad_numel_f),
                        "param_sqsum": float(param_sqsum),
                        "grad_sqsum": float(grad_sqsum),
                        "param_abs_sum": float(param_abs_sum),
                        "grad_abs_sum": float(grad_abs_sum),
                        "param_abs_max": float(param_abs_max),
                        "grad_abs_max": float(grad_abs_max),
                        "param_has_nan": bool(p_nan > 0),
                        "param_has_inf": bool(p_inf > 0),
                        "grad_has_nan": bool(g_nan > 0),
                        "grad_has_inf": bool(g_inf > 0),
                    }

            if rank != 0:
                return
        else:
            for group in sorted(group_stats.keys()):
                gs = group_stats[group]
                reduced_group_stats[group] = {
                    "numel": int(gs["numel"]),
                    "grad_numel": int(gs["grad_numel"]),
                    "param_sqsum": float(gs["param_sqsum"].item()),
                    "grad_sqsum": float(gs["grad_sqsum"].item()),
                    "param_abs_sum": float(gs["param_abs_sum"].item()),
                    "grad_abs_sum": float(gs["grad_abs_sum"].item()),
                    "param_abs_max": float(gs["param_abs_max"].item()),
                    "grad_abs_max": float(gs["grad_abs_max"].item()),
                    "param_has_nan": bool(int(gs["param_has_nan"].item()) > 0),
                    "param_has_inf": bool(int(gs["param_has_inf"].item()) > 0),
                    "grad_has_nan": bool(int(gs["grad_has_nan"].item()) > 0),
                    "grad_has_inf": bool(int(gs["grad_has_inf"].item()) > 0),
                }

        # Materialize selected per-parameter stats.
        # For simplicity and to avoid collective mismatches, we keep per-rank stats:
        # - If grad_log_rank0_only: only rank0 uses its local stats.
        # - Else: each rank writes its own local stats to file.
        if self.grad_log_param_patterns:
            if self.grad_log_rank0_only and rank != 0:
                reduced_param_stats = {}
            else:
                reduced_param_stats = {name: dict(ps) for name, ps in param_stats.items()}

        groups_compact = []
        params_compact = []
        anomaly_groups = []
        global_numel = 0
        global_param_sqsum = 0.0
        global_grad_sqsum = 0.0
        global_param_abs_sum = 0.0
        global_grad_abs_sum = 0.0
        global_param_abs_max = 0.0
        global_grad_abs_max = 0.0

        global_grad_numel = 0

        for group, gs in reduced_group_stats.items():
            numel = int(gs["numel"])  # type: ignore[arg-type]
            grad_numel = int(gs.get("grad_numel", 0))  # type: ignore[arg-type]
            param_sqsum = float(gs["param_sqsum"])  # type: ignore[arg-type]
            grad_sqsum = float(gs["grad_sqsum"])  # type: ignore[arg-type]
            param_abs_sum = float(gs["param_abs_sum"])  # type: ignore[arg-type]
            grad_abs_sum = float(gs["grad_abs_sum"])  # type: ignore[arg-type]
            param_abs_max = float(gs["param_abs_max"])  # type: ignore[arg-type]
            grad_abs_max = float(gs["grad_abs_max"])  # type: ignore[arg-type]

            param_norm = math.sqrt(param_sqsum) if param_sqsum > 0 else 0.0
            grad_norm_group = math.sqrt(grad_sqsum) if grad_sqsum > 0 else 0.0
            param_abs_mean = (param_abs_sum / numel) if numel > 0 else 0.0
            grad_abs_mean = (grad_abs_sum / grad_numel) if grad_numel > 0 else 0.0

            prev = self._grad_log_prev_stats.get(group, {})
            d_param_norm = param_norm - float(prev.get("param_norm", 0.0))
            d_grad_norm = grad_norm_group - float(prev.get("grad_norm", 0.0))
            d_param_abs_mean = param_abs_mean - float(prev.get("param_abs_mean", 0.0))
            d_grad_abs_mean = grad_abs_mean - float(prev.get("grad_abs_mean", 0.0))

            has_anomaly = bool(gs.get("param_has_nan") or gs.get("param_has_inf") or gs.get("grad_has_nan") or gs.get("grad_has_inf"))
            if has_anomaly:
                anomaly_groups.append(group)

            groups_compact.append(
                {
                    "group": group,
                    "numel": numel,
                    "grad_numel": grad_numel,
                    "param_norm": param_norm,
                    "param_abs_mean": param_abs_mean,
                    "param_abs_max": param_abs_max,
                    "grad_norm": grad_norm_group,
                    "grad_abs_mean": grad_abs_mean,
                    "grad_abs_max": grad_abs_max,
                    "d_param_norm": d_param_norm,
                    "d_grad_norm": d_grad_norm,
                    "d_param_abs_mean": d_param_abs_mean,
                    "d_grad_abs_mean": d_grad_abs_mean,
                    "anomaly": has_anomaly,
                }
            )

            global_numel += numel
            global_grad_numel += grad_numel
            global_param_sqsum += param_sqsum
            global_grad_sqsum += grad_sqsum
            global_param_abs_sum += param_abs_sum
            global_grad_abs_sum += grad_abs_sum
            global_param_abs_max = max(global_param_abs_max, param_abs_max)
            global_grad_abs_max = max(global_grad_abs_max, grad_abs_max)

        # Optional per-parameter stats (for selected parameters)
        if reduced_param_stats:
            for name, ps in reduced_param_stats.items():
                numel = int(ps["numel"])
                grad_numel = int(ps["grad_numel"])
                param_sqsum = float(ps["param_sqsum"])
                grad_sqsum = float(ps["grad_sqsum"])
                param_abs_sum = float(ps["param_abs_sum"])
                grad_abs_sum = float(ps["grad_abs_sum"])
                param_abs_max = float(ps["param_abs_max"])
                grad_abs_max = float(ps["grad_abs_max"])
                param_dtype = ps.get("param_dtype")
                grad_dtype = ps.get("grad_dtype")

                param_norm = math.sqrt(param_sqsum) if param_sqsum > 0 else 0.0
                grad_norm_val = math.sqrt(grad_sqsum) if grad_sqsum > 0 else 0.0
                param_abs_mean = (param_abs_sum / numel) if numel > 0 else 0.0
                grad_abs_mean = (grad_abs_sum / grad_numel) if grad_numel > 0 else 0.0

                prev = self._grad_log_prev_stats.get(name, {})
                d_param_norm = param_norm - float(prev.get("param_norm", 0.0))
                d_grad_norm = grad_norm_val - float(prev.get("grad_norm", 0.0))
                d_param_abs_mean = param_abs_mean - float(prev.get("param_abs_mean", 0.0))
                d_grad_abs_mean = grad_abs_mean - float(prev.get("grad_abs_mean", 0.0))

                has_anomaly = bool(
                    ps.get("param_has_nan")
                    or ps.get("param_has_inf")
                    or ps.get("grad_has_nan")
                    or ps.get("grad_has_inf")
                )

                params_compact.append(
                    {
                        "name": name,
                        "numel": numel,
                        "grad_numel": grad_numel,
                        "param_dtype": param_dtype,
                        "grad_dtype": grad_dtype,
                        "param_norm": param_norm,
                        "param_abs_mean": param_abs_mean,
                        "param_abs_max": param_abs_max,
                        "grad_norm": grad_norm_val,
                        "grad_abs_mean": grad_abs_mean,
                        "grad_abs_max": grad_abs_max,
                        "d_param_norm": d_param_norm,
                        "d_grad_norm": d_grad_norm,
                        "d_param_abs_mean": d_param_abs_mean,
                        "d_grad_abs_mean": d_grad_abs_mean,
                        "anomaly": has_anomaly,
                    }
                )

        # Update prev snapshot (rank0 only, or per-rank if rank0_only disabled)
        self._grad_log_prev_stats = {}
        for g in groups_compact:
            self._grad_log_prev_stats[g["group"]] = {
                "param_norm": float(g["param_norm"]),
                "grad_norm": float(g["grad_norm"]),
                "param_abs_mean": float(g["param_abs_mean"]),
                "grad_abs_mean": float(g["grad_abs_mean"]),
            }
        for p in params_compact:
            self._grad_log_prev_stats[p["name"]] = {
                "param_norm": float(p["param_norm"]),
                "grad_norm": float(p["grad_norm"]),
                "param_abs_mean": float(p["param_abs_mean"]),
                "grad_abs_mean": float(p["grad_abs_mean"]),
            }

        # Stable ordering for readability
        groups_compact = sorted(groups_compact, key=lambda x: str(x["group"]))
        params_compact = sorted(params_compact, key=lambda x: str(x["name"]))

        record = {
            "optimizer_step": int(self._optimizer_step),
            "global_step": int(getattr(self, "global_step", 0)),
            "effective_step": int(effective_step),
            "accumulation_step": int(self.gradient_accumulation_step),
            "rank": int(rank),
            "world_size": int(world_size),
            "did_step": bool(did_step),
            "lr": float(lr) if lr is not None else None,
            "clip_grad_norm": float(grad_norm) if grad_norm is not None else None,
            "global": {
                "numel": int(global_numel),
                "grad_numel": int(global_grad_numel),
                "param_norm": float(math.sqrt(global_param_sqsum)) if global_param_sqsum > 0 else 0.0,
                "grad_norm": float(math.sqrt(global_grad_sqsum)) if global_grad_sqsum > 0 else 0.0,
                "param_abs_mean": float(global_param_abs_sum / global_numel) if global_numel > 0 else 0.0,
                "grad_abs_mean": float(global_grad_abs_sum / global_grad_numel) if global_grad_numel > 0 else 0.0,
                "param_abs_max": float(global_param_abs_max),
                "grad_abs_max": float(global_grad_abs_max),
            },
            "anomaly_groups": anomaly_groups[:200],
        }
        if self.grad_log_groups:
            record["groups"] = groups_compact
        if params_compact:
            record["params"] = params_compact

        grad_file = os.path.join(self.grad_log_dir, f"grad_stats_rank{rank}.jsonl")
        with open(grad_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def validation_step(self):
        pass

    def train(self, resume_from_checkpoint: bool = False):
        # print("================================================")
        # print("self.train_dataset.modality_length", self.train_dataset.modality_length)
        # print("================================================")
        # exit()
        self.prepare_model()
        train_dataloader = self.prepare_dataloader(self.train_dataset, is_training=True)
        self.train_dataloader = train_dataloader
        if self.eval_dataset is not None:
            raise NotImplementedError("Evaluation is not implemented")
        self.prepare_optimizer()

        # Validate config for IterableDataset and Dataset
        self.prepare_and_validate_config()

        # Transformers 5 removed warmup_ratio from TrainingArguments. Keep the
        # old behavior for existing configs while allowing Qwen3.5 configs to
        # provide the now-canonical absolute warmup_steps value.
        warmup_ratio = getattr(self.args, "warmup_ratio", 0.0)
        warmup_steps = (
            int(self.total_steps * warmup_ratio)
            if warmup_ratio > 0
            else int(self.args.warmup_steps)
        )
        self.prepare_scheduler(warmup_steps, self.total_steps)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Initialize tracking
        if rank == 0:
            self.tracking = Tracking(
                project_name=os.environ.get("WANDB_PROJECT", self.args.project),
                experiment_name=os.environ.get("WANDB_NAME", self.args.run_name),
                default_backend=self.default_backend,
                config=self.args,
            )

        self.total_tokens = 0
        self.total_sample_num = 0
        # Reset gradient accumulation counter at the start of training
        self.gradient_accumulation_step = 0
        if resume_from_checkpoint:
            # Search for the latest checkpoint in the output_dir
            checkpoints = [f for f in os.listdir(self.args.output_dir) if f.startswith("checkpoint")]
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoints[-1]
            self.load_checkpoints(
                os.path.join(self.args.output_dir, latest_checkpoint),
                int(latest_checkpoint.split("-")[1]),
            )
            start_epoch = int(latest_checkpoint.split("-")[1]) / self.steps_per_epoch
            # start_epoch is a float, we need to convert it to an integer
            start_epoch = int(start_epoch)
            self.global_step = int(latest_checkpoint.split("-")[1])
            need_update_pbar = True
        else:
            start_epoch = 0
            self.global_step = 0
            need_update_pbar = False

        logger.info(f"Training with {self.args.num_train_epochs} epochs with {self.total_steps} steps")
        self.step_profiler.start()

        curr_epoch = start_epoch

        # for debugging
        # print(self.fsdp2_model.lm_head.weight is self.fsdp2_model.model.language_model.embed_tokens.weight)

        def nparam(m):
            return sum(p.numel() for p in m.parameters())

        # print("text_layers", nparam(self.fsdp2_model.model.language_model.layers))
        # print("embed", nparam(self.fsdp2_model.model.language_model.embed_tokens))
        # print("lm_head", nparam(self.fsdp2_model.lm_head))
        # print("vision", nparam(self.fsdp2_model.model.visual))

        pbar = tqdm(total=self.total_steps, desc="Training", disable=dist.get_rank() != 0)
        delta_time = 0.0
        seq_sum_len = []
        loss_item = 0.0
        total_batch_size = 0
        micro_seq_len_min = torch.tensor(1e6, device=self.fsdp2_model.device, dtype=torch.int32)
        micro_seq_len_max = torch.tensor(0, device=self.fsdp2_model.device, dtype=torch.int32)
        while not self.should_stop():
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(curr_epoch)

            # if the checkpoint is loaded, we need to update the pbar
            # but we only need to update the pbar once
            if need_update_pbar:
                update_step = self.global_step
                pbar.update(update_step)
                need_update_pbar = False
            
            it = iter(self.train_dataloader)
            batch = next(it, None)
            if batch is not None:
                with torch.cuda.stream(self.prefetch_stream):
                    batch = send_to_device(batch, self.fsdp2_model.device, non_blocking=True)
            while batch is not None:
                if self.should_stop():
                    break
                
                # torch.distributed.barrier()

                # for debug

                # torch.cuda.reset_peak_memory_stats()

                # wait for prefetching to finish
                torch.cuda.current_stream(device=self.fsdp2_model.device).wait_stream(self.prefetch_stream)

                start_time = time.perf_counter()

                train_metrics = self.training_step(batch)

                should_update = train_metrics.pop("should_update", False)

                # print(torch.cuda.max_memory_allocated() / 1024**3, "GiB allocated")
                # print(torch.cuda.max_memory_reserved() / 1024**3, "GiB reserved")

                end_time = time.perf_counter()
                step_delta_time = end_time - start_time

                # overlap batch to device
                next_batch = next(it, None)
                if next_batch is not None:
                    with torch.cuda.stream(self.prefetch_stream):
                        next_batch = send_to_device(next_batch, self.fsdp2_model.device, non_blocking=True)

                step_seq_len, batch_size = _get_batch_sequence_lengths(batch)

                loss_item_pre_step = train_metrics.get("train/loss")

                delta_time += step_delta_time
                step_seq_len_sum = step_seq_len.sum()
                step_seq_len_min = step_seq_len.min()
                step_seq_len_max = step_seq_len.max()
                seq_sum_len.append(step_seq_len_sum)
                micro_seq_len_min = torch.minimum(micro_seq_len_min, step_seq_len_min)
                micro_seq_len_max = torch.maximum(micro_seq_len_max, step_seq_len_max)
                loss_item += loss_item_pre_step
                total_batch_size += batch_size

                self.print_batch_input(batch)

                batch = next_batch
                
                if should_update:
                # Calculate flops per rank
                    if self.global_step % self.args.logging_steps == 0:
                        seq_sum_len = torch.stack(seq_sum_len)
                        seq_sum_len = seq_sum_len.cpu().tolist()
                        micro_seq_len_max = micro_seq_len_max.cpu().item()
                        micro_seq_len_min = micro_seq_len_min.cpu().item()

                        flops, promised_flops = model_utils.flops_counter.estimate_flops(seq_sum_len, delta_time=delta_time)
                        device = self.fsdp2_model.device
                        sp_size = pgm.process_group_manager.cp_world_size

                        # Calculate training metrics (MFU, token stats, throughput)
                        perf_metrics, self.total_tokens, self.total_sample_num = self.calculate_training_metrics(
                            loss_item = loss_item,
                            flops = flops,
                            sp_size=sp_size,
                            promised_flops=promised_flops,
                            device=device,
                            seq_sum_len=seq_sum_len,
                            total_batch_size = total_batch_size,
                            micro_seq_len_min=micro_seq_len_min,
                            micro_seq_len_max=micro_seq_len_max,
                            log_accumulation_steps=self.args.gradient_accumulation_steps*self.args.logging_steps,
                            total_tokens=self.total_tokens,
                            total_sample_num=self.total_sample_num,
                            delta_time=delta_time,
                            world_size=world_size,
                            rank=rank,
                        )
                        train_metrics.update(perf_metrics)
                        seq_sum_len = []
                        loss_item = 0.0
                        delta_time = 0.0
                        total_batch_size = 0
                        micro_seq_len_min = torch.tensor(1e6, device=self.fsdp2_model.device, dtype=torch.int32)
                        micro_seq_len_max = torch.tensor(0, device=self.fsdp2_model.device, dtype=torch.int32)

                        # Only update global_step and perform step-related operations when gradient accumulation is complete
                        self.step_profiler.step()
                        if self.step_profiler.should_save(self.global_step + 1):
                            self.step_profiler.stop_and_save()
                            self.step_profiler.stop_trace()

                        if self.steps_per_epoch is not None and self.steps_per_epoch > 0:
                            epoch_progress = f"{self.global_step / self.steps_per_epoch:.2f}"
                            train_metrics["train/epoch"] = float(epoch_progress)
                        if rank == 0:
                            self.tracking.log(train_metrics, step=self.global_step)
                    
                    self.global_step += 1
                    
                    if self.should_save:
                        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                        self.save_checkpoints(
                            output_dir,
                            self.global_step,
                            total_limit=self.args.save_total_limit,
                        )

                    if (
                        self.args.torch_empty_cache_steps is not None
                        and self.global_step % self.args.torch_empty_cache_steps == 0
                    ):
                        self.empty_cache()
                    pbar.update(1)
            curr_epoch += 1

            if self.eval_dataset is not None:
                raise NotImplementedError("Evaluation is not implemented")

        pbar.close()
        # Save the final checkpoint
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        self.save_checkpoints(output_dir, self.global_step, total_limit=self.args.save_total_limit)

    def evaluate(self):
        raise NotImplementedError("Evaluation is not implemented")

    def remove_old_checkpoints(self, output_path: str, total_limit: int = None):
        if total_limit is None:
            return
        # get all checkpoints in output_path
        checkpoints = [f for f in os.listdir(output_path) if f.startswith("checkpoint")]
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        if len(checkpoints) > total_limit:
            for checkpoint in checkpoints[:-total_limit]:
                logger.info(f"Removing checkpoint {checkpoint}")
                shutil.rmtree(os.path.join(output_path, checkpoint))

    def save_checkpoints(self, output_path: str, step: int, total_limit: int = None):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            os.makedirs(output_path, exist_ok=True)

        dist.barrier()
        model_path = os.path.join(
            output_path,
            "pytorch_model_fsdp_0",
            f"model_world_size_{world_size}_rank_{rank}.pt",
        )
        optim_path = os.path.join(
            output_path,
            "optimizer",
            f"optimizer_world_size_{world_size}_rank_{rank}.pt",
        )
        extra_state_path = os.path.join(
            output_path,
            "extra_state",
            f"extra_state_world_size_{world_size}_rank_{rank}.pt",
        )
        dataloader_state_path = os.path.join(
            output_path,
            "dataloader_state",
            f"dataloader_state_world_size_{world_size}_rank_{rank}.pt",
        )
        if rank == 0:
            os.makedirs(os.path.join(output_path, "pytorch_model_fsdp_0"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "optimizer"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "extra_state"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "dataloader_state"), exist_ok=True)

        dist.barrier()

        torch.save(self.fsdp2_model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)
        extra_state = {
            "lr_scheduler_state": self.scheduler.state_dict(),
            "rng": self.get_rng_state(),
            "total_tokens": self.total_tokens,
        }
        torch.save(extra_state, extra_state_path)
        torch.save(self.train_dataloader.state_dict(), dataloader_state_path)
        logger.info(f"Saved checkpoint to {output_path} at step {step}")

        if rank == 0:
            self.processing_class.save_pretrained(output_path)
            self.model.config.save_pretrained(output_path)
            self.remove_old_checkpoints(self.args.output_dir, total_limit=self.args.save_total_limit)

        dist.barrier()

    @property
    def should_save(self):
        return self.global_step % self.args.save_steps == 0 and self.global_step > 0

    def load_checkpoints(self, output_path: str, step: int):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        model_path = os.path.join(
            output_path,
            "pytorch_model_fsdp_0",
            f"model_world_size_{world_size}_rank_{rank}.pt",
        )
        optim_path = os.path.join(
            output_path,
            "optimizer",
            f"optimizer_world_size_{world_size}_rank_{rank}.pt",
        )
        extra_state_path = os.path.join(
            output_path,
            "extra_state",
            f"extra_state_world_size_{world_size}_rank_{rank}.pt",
        )
        dataloader_state_path = os.path.join(
            output_path,
            "dataloader_state",
            f"dataloader_state_world_size_{world_size}_rank_{rank}.pt",
        )

        model_state_dict = torch.load(model_path, weights_only=False)
        self.fsdp2_model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(torch.load(optim_path, weights_only=False))
        extra_state = torch.load(extra_state_path, weights_only=False)
        self.total_tokens = extra_state["total_tokens"]
        self.load_rng_state(extra_state["rng"])
        self.scheduler.load_state_dict(extra_state["lr_scheduler_state"])
        self.train_dataloader.load_state_dict(torch.load(dataloader_state_path, weights_only=False))
        # Reset gradient accumulation counter when loading checkpoint
        self.gradient_accumulation_step = 0
        logger.info(f"Loaded checkpoint from {output_path} at step {step}")

    def get_rng_state(self):
        return {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

    def prepare_and_validate_config(self):
        if isinstance(self.train_dataset, IterableDataset):
            is_iterable_dataset = True
        else:
            is_iterable_dataset = False

        if is_iterable_dataset:
            assert self.args.max_steps > 0, "max_steps must be set for IterableDataset"
            if self.args.num_train_epochs > 1:
                logger.warning("num_train_epochs will be ignored for IterableDataset")
                self.args.num_train_epochs = 1
            self.steps_per_epoch = self.args.max_steps
            self.total_steps = self.args.max_steps
        else:
            self.steps_per_epoch = len(self.train_dataloader)
            self.total_steps = (
                self.steps_per_epoch * self.args.num_train_epochs if self.args.max_steps < 0 else self.args.max_steps
            )

    def should_stop(self):
        if self.global_step >= self.total_steps and self.total_steps > 0:
            return True
        return False

    def load_rng_state(self, rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

    def empty_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def print_batch_input(self, batch):
        if self.args.print_batch_input_steps > 0 and self.global_step % self.args.print_batch_input_steps == 0:
            try:
                input_ids = batch.get("input_ids", torch.tensor(0))
                logger.info(self.processing_class.processor.batch_decode(input_ids, skip_special_tokens=True)[0])
            except Exception as e:
                logger.error(f"Error printing batch input: {e}")

    @staticmethod
    def calculate_training_metrics(
        loss_item: float,
        flops: int,
        sp_size: int,
        promised_flops: float,
        device: torch.device,
        seq_sum_len: list,
        total_batch_size: int, 
        micro_seq_len_min: int,
        micro_seq_len_max: int,
        log_accumulation_steps: int,
        total_tokens: int,
        total_sample_num: int,
        delta_time: float,
        world_size: int,
        rank: int,
    ) -> tuple[dict, int]:
        """
        Calculate training performance metrics including MFU, token statistics, and throughput.

        Args:
            flops: Containing FLOPs count
            sp_size: Sequence parallel size
            promised_flops: Promised FLOPs capacity
            device: Device to perform computations on
            seq_len: List of sequence lengths per batch
            total_tokens: Current total token count
            delta_time: Time taken for the training step
            world_size: Distributed training world size

        Returns:
            tuple: (metrics_dict, updated_total_tokens)
        """
        metrics = {}

        # Calculate train_loss
        loss_t = torch.tensor(loss_item, device=device)
        # Calculate mfu per rank
        # Divide by sp size because attention mask we use to calculate are unsplitted
        mfu_t = flops / sp_size / promised_flops
        mfu_t = torch.tensor(mfu_t, device=device)
        # Calculating token stats
        seq_sum_len = torch.tensor(seq_sum_len, device=device, dtype=torch.float32)  
        # Divide total seq len by sp size if sp is enabled since we split the seq len
        total_seq_len = seq_sum_len.sum() / sp_size
        # Avg seq len won't be effected by sp since we perform all reduce
        # across world size
        global_seq_len_avg_t = seq_sum_len.sum() / log_accumulation_steps
        global_seq_len_min = seq_sum_len.min()
        global_seq_len_max = seq_sum_len.max()
        micro_seq_len_min = torch.tensor(micro_seq_len_min, device=device, dtype=torch.int32)
        micro_seq_len_max = torch.tensor(micro_seq_len_max, device=device, dtype=torch.int32)
        total_batch_size = torch.tensor(total_batch_size, device=device)

        # pack sum,min,max tensors for reduce
        sum_group = torch.stack([loss_t, mfu_t, global_seq_len_avg_t, total_seq_len, total_batch_size])
        min_group = torch.stack([global_seq_len_min, micro_seq_len_min])
        max_group = torch.stack([global_seq_len_max, micro_seq_len_max])

        torch.distributed.reduce(sum_group, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(min_group, dst=0, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.reduce(max_group, dst=0, op=torch.distributed.ReduceOp.MAX)
        
        # torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, async_op=True)        
        # torch.distributed.all_reduce(mfu, op=torch.distributed.ReduceOp.AVG, async_op=True)    
        # torch.distributed.all_reduce(total_seq_len, op=torch.distributed.ReduceOp.SUM, async_op=True)
        # torch.distributed.all_reduce(global_seq_len_avg, op=torch.distributed.ReduceOp.AVG, async_op=True)
        # torch.distributed.all_reduce(global_seq_len_min, op=torch.distributed.ReduceOp.MIN, async_op=True)
        # torch.distributed.all_reduce(global_seq_len_max, op=torch.distributed.ReduceOp.MAX, async_op=True)
        # torch.distributed.all_reduce(micro_seq_len_min, op=torch.distributed.ReduceOp.MIN, async_op=True)
        # torch.distributed.all_reduce(micro_seq_len_max, op=torch.distributed.ReduceOp.MAX, async_op=True)
        # torch.distributed.all_reduce(total_batch_size, op=torch.distributed.ReduceOp.SUM, async_op=True)

        # rank 0 collects and computes final metrics
        if rank != 0:
            return {}, 0, 0
        
        loss_t, mfu_t, global_seq_len_avg_t, total_seq_len, total_batch_size = sum_group
        loss, mfu, global_seq_len_avg = loss_t/world_size, mfu_t/world_size, global_seq_len_avg_t/world_size
        global_seq_len_min, micro_seq_len_min = min_group
        global_seq_len_max, micro_seq_len_max = max_group

        mfu = mfu.item()
        total_batch_size = total_batch_size.item()
        total_seq_len = total_seq_len.item()

        total_tokens += total_seq_len
        total_sample_num += total_batch_size
        tokens_per_second = total_seq_len / delta_time
        tokens_per_gpu = tokens_per_second / sp_size / world_size

        metrics["train/loss"] = loss.item()
        metrics["perf/global_seq_len_avg"] = global_seq_len_avg.item()
        metrics["perf/global_seq_len_min"] = global_seq_len_min.item()
        metrics["perf/global_seq_len_max"] = global_seq_len_max.item()
        metrics["perf/micro_seq_len_min"] = micro_seq_len_min.item()
        metrics["perf/micro_seq_len_max"] = micro_seq_len_max.item()
        metrics["perf/micro_seq_len_avg"] = total_seq_len / total_batch_size
        metrics["train/mfu"] = round(mfu, 2)
        metrics["train/delta_time"] = round(delta_time, 2)

        # Log total tokens and total tokens per second
        metrics["train/total_tokens"] = total_tokens
        metrics["train/total_sample_num"] = total_sample_num
        metrics["train/tokens_per_second"] = round(tokens_per_second)
        metrics["train/tokens_per_gpu"] = round(tokens_per_gpu)

        return metrics, total_tokens, total_sample_num
