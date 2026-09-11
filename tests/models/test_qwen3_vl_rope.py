import importlib.util
import types
import unittest
from pathlib import Path

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


MODULE_PATH = Path(__file__).resolve().parents[2] / "verl/models/transformers/qwen3_vl.py"
SPEC = importlib.util.spec_from_file_location("qwen3_vl_rope_under_test", MODULE_PATH)
ROPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROPE)


class FakeTokenizer:
    TOKENS = {"<|vision_start|>": 10, "<|image_pad|>": 11, "<|video_pad|>": 13}

    def convert_tokens_to_ids(self, token):
        return self.TOKENS[token]


class FakeProcessor:
    tokenizer = FakeTokenizer()
    image_processor = types.SimpleNamespace(merge_size=2)


class Qwen3VLRopeTest(unittest.TestCase):
    def test_matches_transformers_reference_for_one_image(self):
        input_ids = torch.tensor([101, 10, 11, 11, 11, 11, 12, 102], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
        mm_token_type_ids = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.long)

        actual = ROPE.get_rope_index(
            FakeProcessor(),
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        fake_model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                vision_config=types.SimpleNamespace(spatial_merge_size=2),
                image_token_id=11,
                video_token_id=13,
                vision_start_token_id=10,
            )
        )
        fake_model.get_vision_position_ids = Qwen3VLModel.get_vision_position_ids.__get__(fake_model, type(fake_model))
        expected, _ = Qwen3VLModel.get_rope_index(
            fake_model,
            input_ids=input_ids.unsqueeze(0),
            mm_token_type_ids=mm_token_type_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask.unsqueeze(0),
        )
        torch.testing.assert_close(actual, expected[:, 0])


if __name__ == "__main__":
    unittest.main()
