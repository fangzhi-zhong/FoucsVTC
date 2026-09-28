from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

import transformers

from ..datasets import DatasetConfig
from ..models import ModelConfig


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_muon: Optional[bool] = False
    freeze_modules: Optional[List[str]] = None
    # If set, only parameters whose names contain any of these substrings are trainable.
    # Example: ["self_attn.gate_proj"] for Glyph attention scaled gate only.
    trainable_param_patterns: Optional[List[str]] = None
    # Optional parameter-name module prefixes/globs mapped to custom learning rates.
    # Later entries take precedence when multiple patterns match.
    module_learning_rates: Optional[Dict[str, float]] = None
    use_rmpad: Optional[bool] = False
    fsdp2: Optional[bool] = False
    sp_ulysses_degree: Optional[int] = 1
    reduce_dtype: Optional[str] = "bfloat16"
    output_dtype: Optional[str] = "bfloat16"
    print_batch_input_steps: Optional[int] = -1
    enable_profiler: Optional[bool] = False
    profiler_config: Optional[Dict[str, Any]] = None
    # Gradient/parameter statistics logging (FSDP2 trainer)
    grad_log_enabled: Optional[bool] = False
    grad_log_dir: Optional[str] = None
    grad_log_rank0_only: Optional[bool] = True
    # Log once every N optimizer steps (i.e., when accumulation completes)
    grad_log_every_n_steps: Optional[int] = 1
    # If true, include per-group stats (per transformer block). If false, only global stats.
    grad_log_groups: Optional[bool] = True
    # Optional: per-parameter logging for selected parameters (matched by name substring).
    # Example: ["norm", "ln_", "proj"] to focus on norm / projection layers.
    grad_log_param_patterns: Optional[List[str]] = None


@dataclass
class TrainerConfig:
    trainer_type: Literal["hf_trainer", "fsdp2_trainer"]
    dataset_config: DatasetConfig
    trainer_args: TrainingArguments
    model_config: ModelConfig
    extra_kwargs: Dict[str, Any] = None

    def to_dict(self):
        trainer_args_dict = self.trainer_args.to_dict()
        model_config_dict = self.model_config.to_dict()
        dataset_config_dict = self.dataset_config.to_dict()
        final_dict = asdict(self)
        final_dict["trainer_args"] = trainer_args_dict
        final_dict["model_config"] = model_config_dict
        final_dict["dataset_config"] = dataset_config_dict
        return final_dict
