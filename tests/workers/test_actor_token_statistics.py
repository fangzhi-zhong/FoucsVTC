"""CPU checks for response-logit selection and bounded monitoring entropy."""

from contextlib import nullcontext
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

import verl.utils.torch_functional as verl_F
import verl.workers.actor.dp_actor as actor_module


class _LanguageHead(nn.Module):
    """Exercise the same pre-head tensor indexing used by HF and Liger Qwen3.5."""

    def __init__(self, dtype, model_type="qwen3_5"):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.hidden = nn.Parameter(torch.randn(2, 12, 5, dtype=dtype))
        self.weight = nn.Parameter(torch.randn(11, 5, dtype=dtype))
        self.logits_to_keep = None

    def forward(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep=0):
        self.logits_to_keep = logits_to_keep
        positions = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return types.SimpleNamespace(logits=self.hidden[:, positions, :] @ self.weight.T)


class ActorTokenStatisticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_monitoring_entropy_matches_dense_for_noncontiguous_microbatch_two(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.randn(2, 8, 11, dtype=dtype)[:, :-1, :]
                self.assertFalse(logits.is_contiguous())
                dense_entropy = verl_F.entropy_from_logits
                with torch.no_grad(), patch.object(verl_F, "entropy_from_logits", wraps=dense_entropy) as entropy:
                    actual = verl_F.entropy_from_logits_chunked(logits, chunk_size=3)
                torch.testing.assert_close(actual, dense_entropy(logits))
                self.assertEqual([call.args[0].shape[0] for call in entropy.call_args_list], [3, 3, 1, 3, 3, 1])
                self.assertTrue(all(call.args[0].ndim == 2 for call in entropy.call_args_list))

    def test_zero_chunk_size_retains_dense_monitoring(self):
        logits = torch.randn(2, 7, 11)
        dense_entropy = verl_F.entropy_from_logits
        with torch.no_grad(), patch.object(verl_F, "entropy_from_logits", wraps=dense_entropy) as entropy:
            actual = verl_F.entropy_from_logits_chunked(logits, chunk_size=0)
        entropy.assert_called_once_with(logits)
        torch.testing.assert_close(actual, dense_entropy(logits))

    def test_differentiable_entropy_retains_dense_gradients_and_higher_derivatives(self):
        logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(2, 3, dtype=torch.float64)
        dense_entropy = verl_F.entropy_from_logits
        with patch.object(verl_F, "entropy_from_logits", wraps=dense_entropy) as entropy:
            actual = verl_F.entropy_from_logits_chunked(logits, chunk_size=2)
        entropy.assert_called_once_with(logits)
        expected = dense_entropy(logits)
        actual_grad = torch.autograd.grad((actual * weights).sum(), logits, create_graph=True)[0]
        expected_grad = torch.autograd.grad((expected * weights).sum(), logits, create_graph=True)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)
        torch.testing.assert_close(
            torch.autograd.grad(actual_grad.square().sum(), logits)[0],
            torch.autograd.grad(expected_grad.square().sum(), logits)[0],
        )

    def _check_actor(self, dtype, calculate_entropy, model_type="qwen3_5"):
        model = _LanguageHead(dtype, model_type)
        actor = object.__new__(actor_module.DataParallelPPOActor)
        actor.actor_module = model
        actor.use_remove_padding = False
        actor.entropy_chunk_size = 3
        labels = torch.randint(0, 11, (2, 7))
        micro_batch = {
            "responses": labels,
            "input_ids": torch.zeros(2, 12, dtype=torch.long),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
            "position_ids": torch.arange(12).expand(2, -1),
        }
        captured = {}

        def cpu_logprobs(logits, labels, inplace_backward):
            captured["shape"] = tuple(logits.shape)
            captured["contiguous"] = logits.is_contiguous()
            captured["inplace_backward"] = inplace_backward
            return verl_F.logprobs_from_logits_naive(logits, labels)

        # The actor's production context is CUDA autocast and its CE uses
        # Triton. Keep these tests on CPU while testing the actual actor's
        # position selection, tensor layout, entropy routing and gradients.
        with patch.object(actor_module.torch, "autocast", return_value=nullcontext()), patch.object(
            actor_module, "logprobs_from_logits", side_effect=cpu_logprobs
        ):
            entropy, logprobs = actor._forward_micro_batch(micro_batch, 0.7, calculate_entropy)

        dense_logits = (model.hidden @ model.weight.T) / 0.7
        dense_logits = dense_logits[:, -8:-1, :]
        expected_logprobs = verl_F.logprobs_from_logits_naive(dense_logits, labels)
        tolerance = {"rtol": 0.05, "atol": 0.08} if dtype == torch.bfloat16 else {}
        torch.testing.assert_close(logprobs, expected_logprobs, **tolerance)
        self.assertEqual(captured["shape"], (2, 7, 11))
        self.assertEqual(captured["inplace_backward"], not calculate_entropy)
        if model_type == "qwen3_5":
            self.assertTrue(captured["contiguous"])
            torch.testing.assert_close(model.logits_to_keep, torch.arange(-8, -1))
        else:
            self.assertEqual(model.logits_to_keep, 0)

        weights = torch.randn_like(logprobs)
        actual_loss = (logprobs * weights).sum()
        expected_loss = (expected_logprobs * weights).sum()
        if calculate_entropy:
            expected_entropy = verl_F.entropy_from_logits(dense_logits)
            torch.testing.assert_close(entropy, expected_entropy, **tolerance)
            actual_loss = actual_loss + 0.1 * entropy.sum()
            expected_loss = expected_loss + 0.1 * expected_entropy.sum()
        else:
            self.assertIsNone(entropy)
        actual_grads = torch.autograd.grad(actual_loss, (model.hidden, model.weight))
        expected_grads = torch.autograd.grad(expected_loss, (model.hidden, model.weight))
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, **tolerance)

    def test_exact_response_positions_preserve_values_and_gradients(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            for calculate_entropy in (False, True):
                with self.subTest(dtype=dtype, calculate_entropy=calculate_entropy):
                    self._check_actor(dtype, calculate_entropy)

    def test_other_model_keeps_original_causal_slice(self):
        self._check_actor(torch.float64, calculate_entropy=True, model_type="qwen2_5_vl")


if __name__ == "__main__":
    unittest.main()
