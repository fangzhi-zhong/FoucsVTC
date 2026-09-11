"""Exercise rollout cache cleanup without importing CUDA, Ray, or vLLM."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def load_sharding_manager():
    def memory_logger(**kwargs):
        return lambda method: method

    dependencies = {
        "torch": {"cuda": types.SimpleNamespace(empty_cache=Mock())},
        "torch.distributed.device_mesh": {"DeviceMesh": object},
        "torch.distributed.fsdp.api": {
            "FullStateDictConfig": object,
            "ShardedStateDictConfig": object,
            "StateDictType": object,
        },
        "torch.distributed.fsdp.fully_sharded_data_parallel": {"FullyShardedDataParallel": object},
        "verl": {"DataProto": object},
        "verl.protocol": {"all_gather_data_proto": Mock()},
        "verl.third_party.vllm": {
            "LLM": object,
            "vllm_version": "0.19.1",
            "parallel_state": object(),
        },
        "verl.utils.debug": {"GPUMemoryLogger": memory_logger, "log_gpu_memory_usage": Mock()},
        "cache_test.base": {"BaseShardingManager": object},
        "cache_test.patch": {"patched_ds_v3_load_weights": Mock(), "patched_qwen_moe_load_weights": Mock()},
    }
    modules = {}
    for name, attributes in dependencies.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module

    path = Path(__file__).resolve().parents[2] / "verl/workers/sharding_manager/fsdp_vllm.py"
    spec = importlib.util.spec_from_file_location("cache_test.fsdp_vllm", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class SplitCacheEngine:
    """Model vLLM's sender hit suppressing data and sleep clearing only P1."""

    def __init__(self):
        self.sender = set()
        self.receiver = {}
        self.events = []

    def generate(self, key, image):
        data = None if key in self.sender else image
        self.sender.add(key)
        if key not in self.receiver:
            if data is None:
                raise AssertionError("Expected a cached item")
            self.receiver[key] = data
        return self.receiver[key]

    def sleep(self, level):
        self.events.append(("sleep", level))
        self.receiver.clear()

    def reset_mm_cache(self):
        self.events.append(("reset_mm_cache",))
        self.sender.clear()
        self.receiver.clear()


class FSDPVLLMCacheTest(unittest.TestCase):
    def setUp(self):
        self.source = load_sharding_manager()

    def exit_rollout(self, engine):
        manager = self.source.FSDPVLLMShardingManager.__new__(self.source.FSDPVLLMShardingManager)
        manager.inference_engine = engine
        manager.module = Mock()
        manager.device_mesh = None
        manager.__exit__(None, None, None)
        manager.module.train.assert_called_once_with()

    def test_next_rollout_can_reuse_an_image_after_sleep(self):
        image = object()
        engine = SplitCacheEngine()
        self.assertIs(engine.generate("image", image), image)

        self.exit_rollout(engine)

        self.assertEqual(engine.events, [("sleep", 1), ("reset_mm_cache",)])
        self.assertIs(engine.generate("image", image), image)

    def test_older_sleep_engine_without_reset_api(self):
        self.source.vllm_version = "0.7.0"
        engine = types.SimpleNamespace(sleep=Mock())
        self.exit_rollout(engine)
        engine.sleep.assert_called_once_with(level=1)

    def test_wrapped_engine_reset_api_is_supported(self):
        self.source.vllm_version = "0.7.0"
        engine = types.SimpleNamespace(sleep=Mock(), llm_engine=types.SimpleNamespace(reset_mm_cache=Mock()))
        self.exit_rollout(engine)
        engine.llm_engine.reset_mm_cache.assert_called_once_with()

    def test_legacy_offload_versions_keep_the_existing_api(self):
        for version in ("0.5.4", "0.6.3"):
            with self.subTest(version=version):
                self.source.vllm_version = version
                engine = types.SimpleNamespace(offload_model_weights=Mock())
                self.exit_rollout(engine)
                engine.offload_model_weights.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
