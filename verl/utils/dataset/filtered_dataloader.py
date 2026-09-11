"""Keep training batches full when dataset items are skipped at load time."""

import numpy as np
import torch

from verl.protocol import _concat_non_tensor_batch_values


class FilteredDataLoader:
    """Rebatch collated, valid rows in sampler order without replacing samples.

    The source loader must retain its final partial batch. ``len`` is an upper
    bound because the number of runtime skips is unknown until iteration.
    """

    def __init__(self, dataloader, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if dataloader.drop_last:
            raise ValueError("FilteredDataLoader requires source drop_last=False")
        self.dataloader = dataloader
        self.batch_size = batch_size
        self._buffer = None

    def __len__(self):
        return len(self.dataloader.sampler) // self.batch_size

    @staticmethod
    def _size(batch):
        return len(next(iter(batch.values()))) if batch else 0

    @staticmethod
    def _slice(batch, start, end=None):
        # Copy object-array storage so the small pending tail does not retain
        # all decoded images from the batch that has already been consumed.
        return {
            key: value[start:end].copy() if isinstance(value, np.ndarray) else value[start:end]
            for key, value in batch.items()
        }

    @staticmethod
    def _concat(left, right):
        if left.keys() != right.keys():
            raise ValueError("Filtered batches must have matching columns")
        return {
            key: torch.cat((value, right[key]), dim=0)
            if isinstance(value, torch.Tensor)
            else _concat_non_tensor_batch_values([value, right[key]], key)
            for key, value in left.items()
        }

    def __iter__(self):
        source = iter(self.dataloader)
        while True:
            if self._size(self._buffer) >= self.batch_size:
                result = self._slice(self._buffer, 0, self.batch_size)
                remaining = self._slice(self._buffer, self.batch_size)
                self._buffer = remaining if self._size(remaining) else None
                yield result
                continue
            try:
                batch = next(source)
            except StopIteration:
                break
            if self._size(batch):
                self._buffer = batch if self._buffer is None else self._concat(self._buffer, batch)

        if self._buffer is not None:
            print(f"Dropping {self._size(self._buffer)} valid samples from the final incomplete training batch")
            self._buffer = None

    def state_dict(self):
        # The source loader may have advanced past rows still in our buffer.
        # Save them together so checkpoint resume neither loses nor duplicates
        # those rows. Iteration replaces buffers instead of mutating them.
        return {
            "filtered_dataloader_version": 1,
            "dataloader": self.dataloader.state_dict(),
            "buffer": self._buffer,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state):
        if "filtered_dataloader_version" not in state:
            # Checkpoints created before runtime filtering contain just the
            # underlying StatefulDataLoader state.
            self._buffer = None
            self.dataloader.load_state_dict(state)
            return
        if state["filtered_dataloader_version"] != 1:
            raise ValueError("Unsupported filtered dataloader checkpoint version")
        if state["batch_size"] != self.batch_size:
            raise ValueError("Cannot resume a filtered dataloader with a different batch_size")
        self.dataloader.load_state_dict(state["dataloader"])
        self._buffer = state["buffer"]
