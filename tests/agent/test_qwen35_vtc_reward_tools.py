"""Reward contracts for useful, bounded zoom trajectories (CPU only)."""

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[2] / "examples/reward_function/qwen35_vtc_reward.py"
SPEC = importlib.util.spec_from_file_location("vtc_reward_tools", PATH)
REWARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REWARD)

BOX = [100, 100, 300, 300]
OTHER = [600, 600, 800, 800]


def call(box=BOX, page=1, name="zoom_region"):
    box = REWARD._expand_evidence_box(box)
    return '<tool_call>' + json.dumps({"name": name, "arguments": {"page": page, "bbox_2d": box}}) + '</tool_call>'


def score(calls="", answer="42", evidence=None, gold="42", dpi=72):
    metadata = {"dpi": dpi, "num_pages": 2, "evidence_locations": evidence if evidence is not None else [{"page": 1, "bbox": BOX}]}
    final = f"<answer>{answer}</answer>" if answer is not None else ""
    return REWARD.compute_score("vtc", calls + final, gold, metadata)


class ToolRewardTest(unittest.TestCase):
    def test_evidence_expands_by_twenty_percent_and_has_minimum_size(self):
        self.assertEqual(REWARD._expand_evidence_box([100, 100, 300, 300]), [80.0, 80.0, 320.0, 320.0])
        self.assertEqual(REWARD._expand_evidence_box([10, 10, 20, 20]), [0.0, 0.0, 80.0, 80.0])
        self.assertEqual(REWARD._expand_evidence_box([980, 980, 990, 990]), [920.0, 920.0, 1000.0, 1000.0])

    def test_useful_call_has_small_bonus_only_after_correct_final(self):
        self.assertAlmostEqual(score()["score"], 1.0)
        self.assertAlmostEqual(score(call())["score"], 2.0)
        for answer in ("wrong", None):
            self.assertEqual(score(call(), answer)["tool_bonus"], 0)

    def test_partial_multianswer_is_not_success(self):
        result = score(call(), answer="42", gold=np.array(["42", "24"]))
        self.assertEqual(result["acc_reward"], 0.5)
        self.assertEqual(result["tool_bonus"], 0)
        self.assertEqual(result["answer_correct"], 0)

    def test_answer_requires_boundaries(self):
        for gold, answer in (("4", "42"), ("no", "unknown"), ("red", "hundred")):
            self.assertEqual(score(call(), answer=answer, gold=gold)["tool_bonus"], 0)
        self.assertEqual(score(call(), answer="2,521", gold="2521")["answer_correct"], 1)

    def test_iou_and_page_matter(self):
        exact = score(call())
        broad = score(call([50, 50, 350, 350]))
        self.assertGreater(exact["tool_bonus"], broad["tool_bonus"])
        self.assertGreater(broad["tool_bonus"], 0)
        self.assertEqual(score(call(page=2))["tool_bonus"], 0)
        self.assertEqual(score(call(OTHER))["tool_bonus"], 0)

    def test_large_crops_decay_and_full_page_has_no_bonus(self):
        for box, expected in (([0, 0, 800, 1000], 0.08), ([0, 0, 1000, 1000], 0)):
            result = score(call(box), evidence=[{"page": 1, "bbox": box}])
            self.assertAlmostEqual(result["iou_reward"], expected)

    def test_extra_calls_scale_bonus_by_evidence_over_attempts(self):
        evidence = [{"page": 1, "bbox": BOX}, {"page": 2, "bbox": OTHER}]
        two = call() + call(OTHER, page=2)
        baseline = score(two, evidence=evidence)
        for count in (3, 4, 8):
            result = score(two + call() * (count - 2), evidence=evidence)
            self.assertAlmostEqual(result["tool_bonus"], baseline["tool_bonus"] * 2 / count)
            self.assertAlmostEqual(result["tool_penalty"], 0.2 * (count - 2) / count)
            self.assertLess(result["score"], baseline["score"])

    def test_fewer_calls_are_allowed_and_repeated_evidence_is_not_reused(self):
        evidence = [{"page": 1, "bbox": BOX}, {"page": 1, "bbox": OTHER}]
        one = score(call(), evidence=evidence)
        repeated = score(call() * 2, evidence=evidence)
        distinct = score(call() + call(OTHER), evidence=evidence)
        self.assertEqual(one["tool_bonus"], distinct["tool_bonus"])
        self.assertEqual(repeated["matched_evidence"], 1)
        self.assertLess(repeated["tool_bonus"], one["tool_bonus"])

    def test_duplicate_metadata_does_not_inflate_allowance(self):
        evidence = [{"page": 1, "bbox": BOX}] * 4
        result = score(call() * 2, evidence=evidence)
        self.assertEqual(result["evidence_count"], 1)
        self.assertEqual(result["call_efficiency"], 0.5)

    def test_no_evidence_has_no_positive_tool_signal(self):
        self.assertEqual(score(evidence=[])["score"], 1.0)
        result = score(call(), evidence=[])
        self.assertEqual(result["tool_bonus"], 0)
        self.assertAlmostEqual(result["tool_penalty"], 0.2)

    def test_invalid_and_unfinished_calls_count_towards_cost(self):
        invalid = [call(page=0), call(page=True), call(page=3), call(name="other"),
                   '<tool_call>{"name":"zoom_region","arguments":{"page":1,"bbox_2d":[0,0,NaN,500]}}</tool_call>',
                   '<tool_call>{"name":"zoom_region","arguments":{"page":1,"bbox_2d":[0,0,1001,500]}}</tool_call>',
                   '<tool_call>{"name":"zoom_region","arguments":{"page":1,"bbox_2d":[0,0,0,500]}}</tool_call>', '<tool_call>{bad}</tool_call>',
                   '<tool_call><function=zoom_region><parameter=page>1</parameter><parameter=bbox_2d>(100,100,300,300)</parameter></function></tool_call>',
                   '<tool_call>{"name":"zoom_region"']
        for text in invalid:
            with self.subTest(text=text):
                result = score(call() + text)
                self.assertEqual(result["tool_calls"], 2)
                self.assertEqual(result["invalid_tool_calls"], 1)
                self.assertAlmostEqual(result["tool_penalty"], 0.15)

    def test_reasoning_observation_and_nonterminal_answers_cannot_earn_bonus(self):
        samples = [
            '<think>The answer is 42.</think>' + call(),
            '<think><answer>42</answer></think>',
            '<think><answer>42</answer>',
            '<answer>42</answer>' + call(),
            call() + '<|im_end|><|im_start|>user\n<answer>42</answer><|im_end|>',
            call() + '<|im_end|><|im_start|>tool\n<answer>42</answer><|im_end|><|im_start|>assistant\n',
        ]
        for solution in samples:
            with self.subTest(solution=solution):
                result = REWARD.compute_score('vtc', solution, '42', {"evidence_locations": [{"page": 1, "bbox": BOX}]})
                self.assertEqual(result["acc_reward"], 0)
                self.assertEqual(result["tool_bonus"], 0)

    def test_observation_format_examples_are_not_actions(self):
        solution = ('<think>Zoom.</think>' + call() + '<|im_end|>\n'
                    '<|im_start|>user\n<tool_response>crop</tool_response>\n'
                    'Format: <think>...</think> <tool_call>...</tool_call> <answer>...</answer>.'
                    '<|im_end|>\n<|im_start|>assistant\n<think>Read.</think><answer>42</answer><|im_end|>')
        result = REWARD.compute_score('vtc', solution, '42', {"evidence_locations": [{"page": 1, "bbox": BOX}]})
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["invalid_tool_calls"], 0)
        self.assertEqual(result["format_reward"], 1)
        self.assertAlmostEqual(result["score"], 2.0)

    def test_json_metadata_numpy_boxes_and_dpi_variants(self):
        metadata = {"dpi": 96, "evidence_bboxes_by_dpi": {
            "72": [{"page": 2, "bbox": OTHER}], "144": [{"page": 1, "bbox": BOX}],
        }}
        result = REWARD.compute_score('vtc', call() + '<answer>42</answer>', '42', json.dumps(metadata))
        self.assertEqual(result["evidence_count"], 1)
        self.assertAlmostEqual(result["tool_bonus"], 1.0 * (2 / 3) ** 2)
        pairs = REWARD._evidence_pairs({"evidence_locations": np.array([{"page": 1, "bbox": np.array(BOX)}])})
        self.assertEqual(pairs, [(1, REWARD._expand_evidence_box(BOX))])
        self.assertEqual(score(call(), dpi=144)["tool_bonus"], 0)

    def test_xml_protocol(self):
        text = '<tool_call><function=zoom_region><parameter=page>1</parameter><parameter=bbox_2d>[80,80,320,320]</parameter></function></tool_call>'
        self.assertAlmostEqual(score(text)["score"], 2.0)


if __name__ == '__main__':
    unittest.main()
