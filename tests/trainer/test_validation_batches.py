"""Exercise validation batching and metric grouping without models or Ray workers."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _ValidationDataset(Dataset):
    def __init__(self, data_files, tokenizer, processor, config):
        self.size = 32 if data_files == "train" else 17

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Identical questions about different images must remain distinct samples.
        return {
            "input_ids": torch.tensor([10, 11]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "position_ids": torch.arange(2),
            "raw_prompt_ids": [10, 11],
            "raw_prompt": {"role": "user", "content": "What is in the image?"},
            "multi_modal_data": {"image": f"image-{index}"},
            "origin_multi_modal_data": {"image": f"original-{index}"},
            "multi_modal_inputs": {"pixels": index},
            "extra_info": {"sample_id": index, "enable_tools": True},
            "tools": {"name": "zoom"},
            "data_source": "images",
        }


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens):
        return "What is in the image?"


class _RolloutWorker:
    world_size = 8

    def __init__(self):
        self.batch_sizes = []
        self.seen = Counter()

    def generate_sequences(self, batch):
        self.batch_sizes.append(len(batch))
        if len(batch) % self.world_size:
            raise AssertionError("Rollout input must be padded to the worker count")
        if not batch.meta_info["validate"]:
            raise AssertionError("Validation metadata was lost")
        responses = []
        for index, extra_info in enumerate(batch.non_tensor_batch["extra_info"]):
            sample_id = extra_info["sample_id"]
            if not extra_info["enable_tools"]:
                raise AssertionError("Agent extra_info was lost")
            expected = {
                "multi_modal_data": {"image": f"image-{sample_id}"},
                "origin_multi_modal_data": {"image": f"original-{sample_id}"},
                "multi_modal_inputs": {"pixels": sample_id},
                "tools": {"name": "zoom"},
                "raw_prompt": {"role": "user", "content": "What is in the image?"},
            }
            for key, value in expected.items():
                if batch.non_tensor_batch[key][index] != value:
                    raise AssertionError(f"{key} no longer matches its sample")
            responses.append([sample_id % 2 + self.seen[sample_id]])
            self.seen[sample_id] += 1
        return DataProto.from_dict(tensors={"responses": torch.tensor(responses)})


class _Reward:
    def __init__(self):
        self.sample_ids = []
        self.batch_sizes = []

    def __call__(self, batch, return_dict):
        if not return_dict:
            raise AssertionError("Validation must request reward details")
        self.batch_sizes.append(len(batch))
        self.sample_ids.extend(info["sample_id"] for info in batch.non_tensor_batch["extra_info"])
        scores = batch.batch["responses"].float()
        return {"reward_tensor": scores, "reward_extra_info": {"acc": scores[:, 0].tolist()}}


class ValidationBatchesTest(unittest.TestCase):
    def make_trainer(self, batch_size=16, n=1, val_num_workers=None):
        trainer = object.__new__(RayPPOTrainer)
        trainer.config = OmegaConf.create(
            {
                "data": {
                    "train_files": "train",
                    "val_files": "val",
                    "train_batch_size": 16,
                    "val_batch_size": batch_size,
                    "shuffle": False,
                },
                "trainer": {"total_epochs": 1, "total_training_steps": None, "log_val_generations": 0},
                "actor_rollout_ref": {
                    "actor": {"optim": {}},
                    "rollout": {
                        "val_kwargs": {"n": n, "do_sample": False},
                        "agent": {"activate_agent": True, "tool_name_key": "tools"},
                    },
                },
                "critic": {"optim": {}},
                "reward_model": {"enable": False},
            }
        )
        if val_num_workers is not None:
            trainer.config.data.val_num_workers = val_num_workers
        trainer.tokenizer = _Tokenizer()
        trainer.processor = None
        trainer.actor_rollout_wg = _RolloutWorker()
        trainer.val_reward_fn = _Reward()
        with patch("verl.trainer.ppo.ray_trainer.RLHFDataset", _ValidationDataset):
            trainer._create_dataloader()
        return trainer

    def test_loader_uses_bounded_batches_and_keeps_tail(self):
        trainer = self.make_trainer()
        self.assertEqual(trainer.val_dataloader.batch_size, 16)
        self.assertEqual(trainer.val_dataloader.num_workers, 0)
        self.assertEqual([len(batch["input_ids"]) for batch in trainer.val_dataloader], [16, 1])

    def test_null_batch_size_falls_back_to_train_batch_size(self):
        trainer = self.make_trainer(batch_size=None)
        self.assertEqual(trainer.val_dataloader.batch_size, 16)
        self.assertEqual(len(trainer.val_dataloader), 2)

    def test_validation_worker_count_is_configurable(self):
        trainer = self.make_trainer(val_num_workers=2)
        # Do not start subprocesses; this checks the actual loader configuration.
        self.assertEqual(trainer.val_dataloader.num_workers, 2)

    def test_padding_is_removed_before_reward_and_prompts_do_not_merge(self):
        trainer = self.make_trainer()
        metrics = trainer._validate()
        self.assertEqual(trainer.actor_rollout_wg.batch_sizes, [16, 8])
        self.assertEqual(trainer.val_reward_fn.batch_sizes, [16, 1])
        self.assertEqual(trainer.val_reward_fn.sample_ids, list(range(17)))
        self.assertAlmostEqual(metrics["val-core/images/acc/mean@1"], 8 / 17)
        self.assertEqual(metrics["val-aux/images/acc/std@1"], 0)
        self.assertTrue(all("@17" not in key for key in metrics))

    def test_multi_batch_metrics_match_single_batch_with_repeated_samples(self):
        for n in (1, 3):
            with self.subTest(n=n):
                multi = self.make_trainer(batch_size=16, n=n)
                single = self.make_trainer(batch_size=17, n=n)
                multi_metrics = multi._validate()
                single_metrics = single._validate()
                self.assertEqual(multi_metrics.keys(), single_metrics.keys())
                for key in multi_metrics:
                    self.assertAlmostEqual(multi_metrics[key], single_metrics[key], msg=key)
                self.assertEqual(multi.val_reward_fn.sample_ids, np.repeat(np.arange(17), n).tolist())
                self.assertEqual(multi.val_reward_fn.batch_sizes, [16 * n, n])
                self.assertAlmostEqual(multi_metrics[f"val-core/images/acc/mean@{n}"], 8 / 17 + (n - 1) / 2)
                self.assertAlmostEqual(multi_metrics[f"val-aux/images/acc/std@{n}"], np.std(np.arange(n)))
                self.assertTrue(all(f"@{17 * n}" not in key for key in multi_metrics))


if __name__ == "__main__":
    unittest.main()
