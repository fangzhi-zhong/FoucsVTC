"""CPU regressions for Qwen chat prefills, EOS markers, and reward context."""

import importlib.util
import unittest
from pathlib import Path

import torch

from verl import DataProto
from verl.workers.reward_manager.naive import NaiveRewardManager


ROOT = Path(__file__).resolve().parents[2]
REWARD_PATH = ROOT / "examples/reward_function/qwen35_vtc_reward.py"
SPEC = importlib.util.spec_from_file_location("qwen35_vtc_reward_under_test", REWARD_PATH)
REWARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REWARD)

PREFILL_PROMPT = (
    "<|im_start|>system\nRead the document and answer.<|im_end|>\n"
    "<|im_start|>user\nWhat number is on the page?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)
TOOL_CALL = (
    '<tool_call>{"name":"zoom_region","arguments":'
    '{"page":1,"bbox_2d":[0,0,80,80]}}</tool_call>'
)
PREFILLED_RESPONSE = "Inspect the page.</think>\n" + TOOL_CALL + "\n<answer>42</answer>"


class CharacterTokenizer:
    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids.tolist())


def make_reward_batch(prompts, responses):
    """Use asymmetric padding to verify the manager decodes only valid tokens."""
    prompt_width = max(map(len, prompts)) + 3
    response_width = max(map(len, responses)) + 2
    prompt_ids, response_ids, masks = [], [], []
    for prompt, response in zip(prompts, responses):
        prompt_ids.append([0] * (prompt_width - len(prompt)) + [ord(char) for char in prompt])
        response_ids.append([ord(char) for char in response] + [0] * (response_width - len(response)))
        masks.append(
            [0] * (prompt_width - len(prompt)) + [1] * len(prompt)
            + [1] * len(response) + [0] * (response_width - len(response))
        )
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor(prompt_ids, dtype=torch.long),
            "responses": torch.tensor(response_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        },
        non_tensors={
            "data_source": ["vtc"] * len(prompts),
            "reward_model": [{"ground_truth": "42"} for _ in prompts],
            "extra_info": [{"dpi": 72, "index": index} for index in range(len(prompts))],
        },
    )


class Qwen35VTCRewardFormatTest(unittest.TestCase):
    def score(self, response, prompt=None):
        return REWARD.compute_score(
            data_source="vtc", solution_str=response, ground_truth="42",
            extra_info={"dpi": 72, "evidence_locations": [{"page": 1, "bbox": [0, 0, 10, 10]}]}, prompt_str=prompt,
        )

    def test_prefilled_think_and_terminal_eos_receive_full_reward(self):
        for suffix in ("<|im_end|>", "<|endoftext|>", " \n<|im_end|>\t<|endoftext|>\n"):
            with self.subTest(suffix=suffix):
                score = self.score(PREFILLED_RESPONSE + suffix, PREFILL_PROMPT)
                self.assertEqual(score["format_reward"], 1.0)
                self.assertEqual(score["acc_reward"], 1.0)
                self.assertEqual(score["tool_reward"], 1.0)
                self.assertEqual(score["score"], 2.0)

    def test_extra_closing_think_without_prefill_remains_invalid(self):
        score = self.score(PREFILLED_RESPONSE + "<|im_end|>")
        self.assertEqual(score["format_reward"], 0.0)
        self.assertAlmostEqual(score["score"], 1.8)

    def test_complete_think_without_prompt_context_is_valid(self):
        score = self.score("<think>Read the page.</think><answer>42</answer><|im_end|>")
        self.assertEqual(score["format_reward"], 1.0)
        self.assertEqual(score["score"], 1.0)

    def test_missing_empty_unclosed_and_trailing_content_still_fail(self):
        responses = (
            "<think>No answer.</think><|im_end|>",
            "<answer> \n </answer><|im_end|>",
            "<think>Still thinking.<answer>42</answer><|im_end|>",
            "<answer>42</answer>extra text",
            "<answer>42</answer>extra text<|im_end|>",
            "<answer>42</answer><|im_end|>extra text",
            "<answer>42</answer><|eot_id|>",
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertEqual(REWARD._format_reward(response), 0.0)
        self.assertEqual(REWARD._format_reward("<answer>42</answer>", PREFILL_PROMPT), 0.0)

    def test_only_final_assistant_prefill_is_used_from_prompt(self):
        # A user-provided opening tag cannot repair an unmatched response close.
        prompt = "<|im_start|>user\nThe literal tag is <think>.<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(REWARD._format_reward("</think><answer>42</answer>", prompt), 0.0)
        # A second opening tag from the response is not silently repaired.
        self.assertEqual(
            REWARD._format_reward("<think>Again.</think><answer>42</answer>", PREFILL_PROMPT), 0.0
        )
        # Unmatched tags elsewhere in the prompt cannot spoil the valid prefill.
        prompt = "<|im_start|>user\nExample closing tag: </think><|im_end|>\n" + PREFILL_PROMPT
        self.assertEqual(REWARD._format_reward("Done.</think><answer>42</answer><|im_end|>", prompt), 1.0)

    def test_multiturn_tool_trajectory_and_historical_prompt_context(self):
        response = (
            "Need to zoom.</think>" + TOOL_CALL + "<|im_end|>\n"
            "<|im_start|>tool\n<tool_response>Visible number: 42</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>Now I can read it.</think>"
            "<answer>42</answer><|im_end|>"
        )
        self.assertEqual(self.score(response, PREFILL_PROMPT)["score"], 2.0)
        historical_prompt = (
            PREFILL_PROMPT + "Zoom first.</think>" + TOOL_CALL + "<|im_end|>\n"
            "<|im_start|>tool\nObservation includes </think> and <answer>42</answer>.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        score = self.score("Finished.</think><answer>42</answer><|endoftext|>", historical_prompt)
        self.assertEqual(score["format_reward"], 1.0)
        self.assertEqual(score["tool_calls"], 0)
        self.assertEqual(score["score"], 1.0)
        wrong = self.score("Finished.</think><answer>wrong</answer><|im_end|>", historical_prompt)
        self.assertEqual(wrong["acc_reward"], 0.0)
        self.assertEqual(wrong["score"], 0.2)


class NaiveRewardPromptContextTest(unittest.TestCase):
    def test_custom_reward_loader_preserves_prompt_opt_in_through_wrapper(self):
        from verl.trainer.main_ppo import get_custom_reward_fn

        wrapped_score = get_custom_reward_fn({
            "custom_reward_function": {"path": str(REWARD_PATH), "name": "compute_score"}
        })
        manager = NaiveRewardManager(CharacterTokenizer(), num_examine=0, compute_score=wrapped_score)
        response = PREFILLED_RESPONSE + "<|im_end|>"
        result = manager(make_reward_batch([PREFILL_PROMPT], [response]), return_dict=True)
        self.assertEqual(result["reward_extra_info"]["format_reward"], [1.0])
        self.assertAlmostEqual(result["reward_extra_info"]["score"][0], 0.8)
        self.assertAlmostEqual(result["reward_tensor"].sum().item(), 0.8)

    def test_explicit_prompt_parameter_receives_each_unpadded_context(self):
        prompts = [PREFILL_PROMPT, "<|im_start|>assistant\n"]
        responses = [PREFILLED_RESPONSE + "<|im_end|>", PREFILLED_RESPONSE + "<|endoftext|>"]
        seen = []

        def score(data_source, solution_str, ground_truth, extra_info=None, prompt_str=None):
            seen.append((data_source, solution_str, ground_truth, extra_info, prompt_str))
            return REWARD.compute_score(data_source, solution_str, ground_truth, extra_info, prompt_str=prompt_str)

        manager = NaiveRewardManager(CharacterTokenizer(), num_examine=0, compute_score=score)
        result = manager(make_reward_batch(prompts, responses), return_dict=True)
        self.assertEqual([item[4] for item in seen], prompts)
        self.assertEqual([item[1] for item in seen], responses)
        self.assertEqual([item[0] for item in seen], ["vtc", "vtc"])
        self.assertEqual([item[2] for item in seen], ["42", "42"])
        self.assertEqual([item[3] for item in seen], [{"dpi": 72, "index": 0}, {"dpi": 72, "index": 1}])
        self.assertEqual(result["reward_extra_info"]["format_reward"], [1.0, 0.0])
        expected = torch.zeros_like(result["reward_tensor"])
        expected[0, len(responses[0]) - 1] = 0.8
        expected[1, len(responses[1]) - 1] = 0.6
        torch.testing.assert_close(result["reward_tensor"], expected)

    def test_legacy_four_argument_score_remains_compatible(self):
        seen = []

        def legacy_score(data_source, solution_str, ground_truth, extra_info):
            seen.append((data_source, solution_str, ground_truth, extra_info))
            return 0.75

        response = "<answer>42</answer><|im_end|>"
        manager = NaiveRewardManager(CharacterTokenizer(), num_examine=0, compute_score=legacy_score)
        reward = manager(make_reward_batch([PREFILL_PROMPT], [response]))
        self.assertEqual(seen, [("vtc", response, "42", {"dpi": 72, "index": 0})])
        self.assertEqual(reward.sum().item(), 0.75)
        self.assertEqual(reward[0, len(response) - 1].item(), 0.75)

    def test_kwargs_alone_does_not_opt_in_to_prompt_context(self):
        seen_kwargs = []

        def kwargs_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
            seen_kwargs.append(kwargs)
            return 0.0

        manager = NaiveRewardManager(CharacterTokenizer(), num_examine=0, compute_score=kwargs_score)
        manager(make_reward_batch([PREFILL_PROMPT], ["<answer>42</answer>"]))
        self.assertEqual(seen_kwargs, [{}])


if __name__ == "__main__":
    unittest.main()
