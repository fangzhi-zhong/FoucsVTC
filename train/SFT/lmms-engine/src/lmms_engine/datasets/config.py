from typing import List, Literal, Optional, Union

from pydantic import field_validator

from lmms_engine.protocol import Args

from .processor import ProcessorConfig


class DatasetConfig(Args):
    dataset_type: str
    dataset_format: Literal["json", "jsonl", "csv", "yaml", "hf_dataset", "arrow"]
    processor_config: Union[dict, ProcessorConfig]

    # Dataset configuration
    dataset_path: Optional[str] = None  # Optional - used for external files
    datasets: Optional[List[dict]] = None  # Optional - used for inline YAML definitions
    shuffle: bool = True
    data_seed: Optional[int] = 42
    eval_dataset_path: Optional[str] = None

    # Chat-template configuration
    #
    # Qwen3-VL-Instruct's own chat template emits no system block unless the
    # caller supplies one, so leaving add_system_prompt on trains the model on a
    # prefix that no standard client ever sends.
    add_system_prompt: Optional[bool] = True
    # Follow the Qwen3-VL-Thinking convention: the assistant turn opens with
    # `<think>` as part of the *prompt* rather than something the model has to
    # emit. Requires targets that already start with `<think>`, and a serving
    # chat template whose generation prompt ends in `<|im_start|>assistant\n<think>\n`.
    think_mode: Optional[bool] = False
    # Optional instruction appended to the end of every user turn. A newline is
    # inserted before the suffix when the user turn already contains text.
    user_prompt_suffix: Optional[str] = None
    # Qwen3.5-only weak grounding augmentation. When enabled, one prompt from
    # lmms_engine.datasets.grounding_prompts is appended to every user turn.
    random_grounding_prompt: bool = False
    grounding_prompt_seed: Optional[int] = 42

    # Object storage configuration
    object_storage: Optional[Literal["azure", "gcs", "none"]] = "none"
    bucket_name: Optional[str] = None

    # Packing configuration
    packing: Optional[bool] = False
    packing_strategy: Optional[str] = None
    packing_length: Optional[int] = 32000
    filter_overlong: Optional[bool] = True
    filter_overlong_workers: Optional[int] = 8
    max_length: Optional[int] = None
    packing_lookahead: Optional[int] = None
    packing_max_items_per_pack: Optional[int] = None
    # How `length_grouped` packing measures a pack against `packing_length`.
    # "padded" bounds `num_items * max_len`, matching the [B, max_len] tensor the
    # collator emits -- required when the model consumes it directly.
    # "tokens" bounds `sum(len)`, which is the real cost once `use_rmpad` strips
    # padding, and fits substantially more data per step.
    packing_cost: Optional[Literal["padded", "tokens"]] = "padded"
    # Look-ahead pool for `length_grouped` packing, in tokens; defaults to
    # 8 * packing_length. Larger pools group lengths better but hold more
    # decoded samples in host RAM, per dataloader worker.
    packing_pool_tokens: Optional[int] = None

    # Video configuration
    video_sampling_strategy: Optional[Literal["fps", "frame_num"]] = "fps"
    video_max_pixels: Optional[int] = 768 * 28 * 28
    video_max_frames: Optional[int] = 768
    video_min_pixels: Optional[int] = 3136
    frame_num: Optional[int] = 64
    fps: Optional[int] = 1
    video_backend: Optional[Literal["decord", "qwen_vl_utils", "qwen_omni_utils"]] = "qwen_vl_utils"

    @field_validator(
        "video_max_pixels",
        "video_max_frames",
        "frame_num",
        "fps",
        "packing_length",
        "max_length",
        "filter_overlong_workers",
        "packing_pool_tokens",
    )
    @classmethod
    def validate_positive_values(cls, v, info):
        """Validate that numeric video and packing parameters are positive."""
        if v is not None and v <= 0:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v

    @field_validator("video_backend")
    @classmethod
    def validate_video_backend_migration(cls, v):
        """Provide migration warning for deprecated torchvision backend."""
        if v == "torchvision":
            raise ValueError(
                "The 'torchvision' video backend has been removed. "
                "Please use 'decord', 'qwen_vl_utils', or 'qwen_omni_utils' instead. "
                "Migration guide: If you were using torchvision, 'decord' provides "
                "similar functionality with better performance."
            )
        return v
