import random

from torch.utils.data import get_worker_info

from lmms_engine.datasets.collator import (
    Qwen3_5PackingCollator,
    Qwen3_5VisionCollator,
)
from lmms_engine.mapping_func import register_dataset
from lmms_engine.datasets.grounding_prompts import GROUNDING_PROMPTS

from .qwen3_vl_iterable_dataset import Qwen3VLIterableDataset


@register_dataset("qwen3_5_iterable")
class Qwen3_5IterableDataset(Qwen3VLIterableDataset):
    """Qwen3.5 data pipeline with DeltaNet-safe sequence packing."""

    def _random_grounding_prompt(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        key = (getattr(self, "rank", 0), worker_id)
        if getattr(self, "_grounding_rng_key", None) != key:
            base_seed = self.config.grounding_prompt_seed
            if base_seed is None:
                base_seed = 42
            self._grounding_rng = random.Random(
                int(base_seed) + 1_000_003 * key[0] + 9_973 * key[1]
            )
            self._grounding_rng_key = key
        return self._grounding_rng.choice(GROUNDING_PROMPTS)

    def get_collator(self):
        if self.config.packing:
            return Qwen3_5PackingCollator(self.processor)
        return Qwen3_5VisionCollator(self.processor)
