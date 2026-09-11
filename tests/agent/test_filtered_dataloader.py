"""CPU regressions for skipping rows without changing batch size or weights."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.filtered_dataloader import FilteredDataLoader
from verl.utils.dataset.rl_dataset import collate_fn


class _Rows(Dataset):
    def __init__(self, size, skipped=()):
        self.size = size
        self.skipped = set(skipped)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index in self.skipped:
            return None
        return {
            "input_ids": torch.tensor([index]),
            "image": {"id": index},
            # Homogeneous within each source batch, with different widths in
            # adjacent batches: collate produces incompatible 2-D arrays.
            "raw_prompt_ids": [index] * (1 + index // 4),
        }


def make_loader(size, skipped=(), batch_size=4, source_batch_size=None, num_workers=0):
    source = StatefulDataLoader(
        _Rows(size, skipped), batch_size=source_batch_size or batch_size,
        num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=collate_fn,
    )
    return FilteredDataLoader(source, batch_size)


def batch_ids(batch):
    ids = batch["input_ids"][:, 0].tolist()
    assert [image["id"] for image in batch["image"]] == ids
    assert [list(value) for value in batch["raw_prompt_ids"]] == [
        [index] * (1 + index // 4) for index in ids
    ]
    return ids


class FilteredDataLoaderTest(unittest.TestCase):
    def test_rebatch_keeps_order_and_uses_partial_source_tail(self):
        loader = make_loader(11, skipped={0, 1, 10})
        self.assertEqual([batch_ids(batch) for batch in loader], [[2, 3, 4, 5], [6, 7, 8, 9]])

    def test_only_final_incomplete_valid_batch_is_dropped(self):
        loader = make_loader(13, skipped={1, 4, 8, 12})
        self.assertEqual(len(loader), 3)  # Only an upper bound before actual skips.
        self.assertEqual([batch_ids(batch) for batch in loader], [[0, 2, 3, 5], [6, 7, 9, 10]])
        self.assertIsNone(loader.state_dict()["buffer"])

    def test_checkpoint_restores_pending_rows_and_worker_progress(self):
        for workers in (0, 2):
            with self.subTest(num_workers=workers):
                loader = make_loader(14, skipped={1, 4, 12}, num_workers=workers)
                iterator = iter(loader)
                self.assertEqual(batch_ids(next(iterator)), [0, 2, 3, 5])
                state_file = io.BytesIO()
                torch.save(loader.state_dict(), state_file)
                expected = [batch_ids(batch) for batch in iterator]
                state_file.seek(0)
                restored = make_loader(14, skipped={1, 4, 12}, num_workers=workers)
                restored.load_state_dict(torch.load(state_file, weights_only=False))
                self.assertEqual([batch_ids(batch) for batch in restored], expected)
                self.assertEqual(expected, [[6, 7, 8, 9]])

    def test_resume_can_drain_full_buffer_before_reading_source(self):
        loader = make_loader(8, batch_size=2, source_batch_size=8)
        iterator = iter(loader)
        self.assertEqual(batch_ids(next(iterator)), [0, 1])
        restored = make_loader(8, batch_size=2, source_batch_size=8)
        restored.load_state_dict(loader.state_dict())
        self.assertEqual([batch_ids(batch) for batch in restored], [[2, 3], [4, 5], [6, 7]])

    def test_old_loader_checkpoints_are_accepted(self):
        old_loader = StatefulDataLoader(
            _Rows(12), batch_size=4, num_workers=0, shuffle=False,
            drop_last=True, collate_fn=collate_fn,
        )
        source_iterator = iter(old_loader)
        self.assertEqual(batch_ids(next(source_iterator)), [0, 1, 2, 3])
        restored = make_loader(12)
        restored.load_state_dict(old_loader.state_dict())
        self.assertEqual([batch_ids(batch) for batch in restored], [[4, 5, 6, 7], [8, 9, 10, 11]])

    def test_empty_all_skipped_and_next_epoch(self):
        self.assertEqual(list(make_loader(0)), [])
        self.assertEqual(list(make_loader(8, skipped=range(8))), [])
        loader = make_loader(7, skipped={1})
        for _ in range(2):
            self.assertEqual([batch_ids(batch) for batch in loader], [[0, 2, 3, 4]])

    def test_validation_ignores_empty_batches_without_replacing_rows(self):
        trainer = object.__new__(RayPPOTrainer)
        trainer.config = OmegaConf.create({
            "actor_rollout_ref": {"rollout": {
                "val_kwargs": {"n": 1, "do_sample": False}, "agent": {"activate_agent": False},
            }},
            "reward_model": {"enable": False},
            "trainer": {"log_val_generations": 0},
        })
        trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0, decode=lambda *a, **kw: "question")
        trainer.val_dataloader = [None, {
            "input_ids": torch.tensor([[10], [11]]),
            "attention_mask": torch.ones((2, 1), dtype=torch.long),
            "position_ids": torch.zeros((2, 1), dtype=torch.long),
            "raw_prompt_ids": np.array([[10], [11]], dtype=object),
            "data_source": np.array(["test", "test"], dtype=object),
        }, None]
        generate = Mock(side_effect=lambda batch: DataProto.from_dict(
            tensors={"responses": torch.ones((len(batch), 1), dtype=torch.long)}
        ))
        trainer.actor_rollout_wg = SimpleNamespace(world_size=8, generate_sequences=generate)
        trainer.val_reward_fn = Mock(side_effect=lambda batch, return_dict: {
            "reward_tensor": batch.batch["responses"].float(),
        })
        metrics = trainer._validate()
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(len(generate.call_args.args[0]), 8)
        self.assertEqual(len(trainer.val_reward_fn.call_args.args[0]), 2)
        self.assertEqual(metrics["val-core/test/reward/mean@1"], 1)

    def test_validation_with_no_valid_samples_returns_no_metrics(self):
        trainer = object.__new__(RayPPOTrainer)
        trainer.val_dataloader = [None, None]
        self.assertEqual(trainer._validate(), {})


if __name__ == "__main__":
    unittest.main()
