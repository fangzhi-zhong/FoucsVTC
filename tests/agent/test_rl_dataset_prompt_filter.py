"""CPU regressions for prompt filtering and skipping actual overlong inputs."""

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import datasets
import torch
from omegaconf import OmegaConf
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def render_prompt(messages, tools=None, extra_tokens=0):
    content = messages[-1]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    length = int(re.search(r"tokens=(\d+)", content).group(1))
    return "x" * (length + 32 * len(tools or []) + extra_tokens)


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, template_special_tokens=0):
        self.template_special_tokens = template_special_tokens

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, tools=None):
        assert add_generation_prompt
        text = render_prompt(messages, tools)
        if tokenize:
            # Some templates' tokenize=True path differs from rendering text
            # and explicitly tokenizing with add_special_tokens=False.
            return [1] * (len(text) + self.template_special_tokens)
        return text

    def __call__(self, text, return_tensors, add_special_tokens, padding, truncation):
        assert return_tensors == "pt" and not add_special_tokens
        assert not padding and not truncation
        input_ids = torch.ones((1, len(text)), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [1] * len(text)


class FakeProcessor:
    """Expand each decoded image/video by 8000 tokens, without a GPU/model."""

    image_token = "x"
    image_processor = object()

    def __init__(self, template_extra_tokens=0, text_shrink=0):
        self.template_extra_tokens = template_extra_tokens
        self.text_shrink = text_shrink

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, tools=None):
        assert add_generation_prompt and not tokenize
        return render_prompt(messages, tools, self.template_extra_tokens)

    def __call__(self, text, images, videos, return_tensors, padding, truncation):
        assert return_tensors == "pt"
        assert not padding and not truncation
        assert all(isinstance(image, Image.Image) for image in images or [])
        assert all(isinstance(video, torch.Tensor) for video in videos or [])
        media_count = len(images or []) + len(videos or [])
        length = len(text[0]) - self.text_shrink + 8000 * media_count
        input_ids = torch.ones((1, length), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": torch.ones_like(input_ids) * int(media_count > 0),
            "pixel_values": torch.ones((media_count, 3)),
        }


class RLHFDatasetPromptFilterTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "page.png"
        Image.new("RGB", (56, 56), "white").save(self.image_path)
        self.high_res_path = str(self.root / "unopened-high-res.png")
        self.cache_patch = patch.object(datasets.config, "HF_DATASETS_CACHE", self.root / "hf-cache")
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.parquet_index = 0

    def row(self, index, text_tokens, image=False, **fields):
        row = {
            "prompt": [{"role": "user", "content": f"tokens={text_tokens}"}],
            "extra_info": {"index": index},
            **fields,
        }
        if image:
            row["prompt"][0]["content"] += " <image>"
            row["images"] = [str(self.image_path)]
            row["high_res_images"] = [self.high_res_path]
        return row

    def make_dataset(self, rows, processor=None, tokenizer=None, **config):
        path = self.root / f"prompts-{self.parquet_index}.parquet"
        self.parquet_index += 1
        datasets.Dataset.from_list(rows).to_parquet(str(path))
        options = {
            "max_prompt_length": 8192,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_method": "exact",
            "skip_overlong_prompts": False,
            "filter_overlong_prompts_workers": 1,
            "truncation": "error",
            "return_raw_chat": True,
            "cache_dir": str(self.root / "verl-cache"),
            **config,
        }
        return RLHFDataset(str(path), tokenizer or FakeTokenizer(), OmegaConf.create(options), processor)

    def test_image_size_boundary_without_pixel_processing(self):
        rows = [self.row(i, length, image=True) for i, length in enumerate((4193, 4194, 194, 195))]
        for row in rows[2:]:
            row["images"] *= 2
            row["prompt"][0]["content"] += " <image>"
        original = copy.deepcopy(rows)
        processor = FakeProcessor()
        with (
            patch("verl.utils.dataset.prompt_length.estimate_image_tokens", return_value=4000) as estimate,
            patch.object(RLHFDataset, "_prepare_prompt", side_effect=AssertionError("No input preparation")),
            patch.object(FakeProcessor, "__call__", side_effect=AssertionError("No processor forward")),
            patch("verl.utils.dataset.vision_utils.process_image", side_effect=AssertionError("No pixel decoding")),
        ):
            dataset = self.make_dataset(rows, processor, filter_overlong_prompts_method="image_size")
        # Each image replaces one existing placeholder: 4193+3999 and
        # 194+2*3999 both fit exactly; the next token is rejected.
        self.assertEqual(list(dataset.dataframe), [original[0], original[2]])
        self.assertEqual(estimate.call_count, 6)
        self.assertTrue(all(call.args[1] is processor.image_processor for call in estimate.call_args_list))
        self.assertEqual(rows, original)

    def test_image_size_includes_processor_template_and_tools(self):
        schema_path = self.root / "size-tools.json"
        schema_path.write_text(json.dumps([{"type": "function", "function": {"name": "zoom"}}]), encoding="utf-8")
        rows = [
            self.row(0, 185, image=True, enable_tools=False),
            self.row(1, 153, image=True, enable_tools=True),
            self.row(2, 154, image=True, enable_tools=True),
        ]
        with patch("verl.utils.dataset.prompt_length.estimate_image_tokens", return_value=8000):
            dataset = self.make_dataset(
                rows,
                FakeProcessor(template_extra_tokens=8),
                filter_overlong_prompts_method="image_size",
                tools_schema_path=str(schema_path),
            )
        self.assertEqual([row["extra_info"]["index"] for row in dataset.dataframe], [0, 1])

    def test_image_size_supports_text_only_without_tensor_tokenization(self):
        with patch.object(FakeTokenizer, "__call__", side_effect=AssertionError("Only encode is needed")):
            dataset = self.make_dataset(
                [self.row(0, 8192), self.row(1, 8193)], filter_overlong_prompts_method="image_size"
            )
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["attention_mask"].sum().item(), 8192)

    def test_image_size_rejects_video_with_exact_method_guidance(self):
        row = self.row(0, 1, videos=[{"video": ["frame-1.png", "frame-2.png"]}])
        row["prompt"][0]["content"] += " <video>"
        with patch("verl.utils.dataset.vision_utils.process_video", side_effect=AssertionError("No video decoding")):
            with self.assertRaisesRegex(ValueError, "exact"):
                self.make_dataset([row], FakeProcessor(), filter_overlong_prompts_method="image_size")

    def test_actual_overlength_is_skipped_after_underestimated_filter(self):
        rows = [self.row(i, length, image=True) for i, length in enumerate((192, 193, 1196))]
        with patch("verl.utils.dataset.prompt_length.estimate_image_tokens", return_value=1):
            dataset = self.make_dataset(
                rows, FakeProcessor(), filter_overlong_prompts_method="image_size", skip_overlong_prompts=True
            )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0]["attention_mask"].sum().item(), 8192)
        with patch("verl.utils.dataset.vision_utils.process_raw_image", side_effect=AssertionError("Skip before tool images")):
            self.assertIsNone(dataset[1])  # 8193 real tokens
            self.assertIsNone(dataset[2])  # the original 9196-token failure

    def test_actual_raw_prompt_overlength_is_skipped(self):
        dataset = self.make_dataset(
            [self.row(0, 8192), self.row(1, 8193)],
            FakeProcessor(text_shrink=8184),
            filter_overlong_prompts=False,
            skip_overlong_prompts=True,
        )
        self.assertEqual(dataset[0]["attention_mask"].sum().item(), 8)
        self.assertIsNone(dataset[1])

    def test_collate_discards_skipped_examples_and_handles_all_skipped(self):
        first = {"input_ids": torch.tensor([1, 2]), "index": 3}
        second = {"input_ids": torch.tensor([4, 5]), "index": 6}
        batch = collate_fn([None, first, None, second])
        self.assertTrue(torch.equal(batch["input_ids"], torch.tensor([[1, 2], [4, 5]])))
        self.assertEqual(batch["index"].tolist(), [3, 6])
        self.assertIsNone(collate_fn([None, None]))

    def test_image_expansion_filters_exact_boundary_in_multiprocess(self):
        rows = [
            self.row(0, 100, image=True, estimated_prompt_tokens=20000),
            self.row(1, 192, image=True, estimated_prompt_tokens=20000),
            self.row(2, 193, image=True, estimated_prompt_tokens=1),
            self.row(3, 1196, image=True, estimated_prompt_tokens=1),
        ]
        dataset = self.make_dataset(rows, FakeProcessor(), filter_overlong_prompts_workers=2)
        self.assertEqual([row["extra_info"]["index"] for row in dataset.dataframe], [0, 1])
        for offset, expected_length in enumerate((8100, 8192)):
            before = copy.deepcopy(dataset.dataframe[offset])
            sample = dataset[offset]
            self.assertEqual(sample["input_ids"].shape, (8192,))
            self.assertEqual(sample["attention_mask"].sum().item(), expected_length)
            mm_types = sample["multi_modal_inputs"]["mm_token_type_ids"][0]
            self.assertTrue(torch.equal(mm_types, sample["attention_mask"]))
            self.assertEqual(sample["origin_multi_modal_data"]["high_res_image"], [self.high_res_path])
            self.assertEqual(dataset.dataframe[offset], before)

    def test_filter_does_not_decode_raw_or_high_resolution_tool_images(self):
        rows = [self.row(0, 192, image=True)]
        rows[0]["images"] = [{"bytes": self.image_path.read_bytes()}]
        original = copy.deepcopy(rows[0])
        with patch(
            "verl.utils.dataset.vision_utils.process_raw_image",
            side_effect=AssertionError("Filtering must not decode tool-only raw images"),
        ):
            dataset = self.make_dataset(rows, FakeProcessor())
            self.assertTrue(dataset._is_prompt_within_limit(rows[0]))
        # process_image mutates byte-backed dictionaries internally, but the
        # filter callback must preserve both those dictionaries and messages.
        self.assertEqual(rows[0], original)
        self.assertEqual(len(dataset), 1)
        self.assertFalse(Path(self.high_res_path).exists())
        sample = dataset[0]
        self.assertEqual(sample["origin_multi_modal_data"]["high_res_image"], [self.high_res_path])
        self.assertIsInstance(sample["origin_multi_modal_data"]["image"][0], Image.Image)

    def test_processor_template_and_enabled_tools_count_toward_budget(self):
        schema = [{"type": "function", "function": {"name": "zoom"}}]
        schema_path = self.root / "tools.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        # Processor adds 8 tokens; enabled tools add 32 more. The disabled
        # example fits exactly even though its text is longer.
        rows = [
            self.row(0, 184, image=True, enable_tools=False),
            self.row(1, 152, image=True, enable_tools=True),
            self.row(2, 153, image=True, enable_tools=True),
        ]
        dataset = self.make_dataset(rows, FakeProcessor(template_extra_tokens=8), tools_schema_path=str(schema_path))
        self.assertEqual([row["extra_info"]["index"] for row in dataset.dataframe], [0, 1])
        for offset in range(len(dataset)):
            self.assertEqual(dataset[offset]["attention_mask"].sum().item(), 8192)

    def test_text_only_uses_same_tokenizer_call_as_getitem(self):
        rows = [self.row(0, 8192), self.row(1, 8193)]
        dataset = self.make_dataset(rows, tokenizer=FakeTokenizer(template_special_tokens=1))
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample["index"], 0)
        self.assertEqual(sample["attention_mask"].sum().item(), 8192)
        self.assertEqual(len(sample["raw_prompt_ids"]), 8192)

    def test_raw_rollout_prompt_ids_also_stay_within_budget(self):
        rows = [self.row(0, 8192), self.row(1, 8193)]
        # Both processor outputs fit, but raw_prompt_ids is independently
        # consumed by rollout and must not exceed max_prompt_length.
        dataset = self.make_dataset(rows, FakeProcessor(text_shrink=8184))
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample["attention_mask"].sum().item(), 8)
        self.assertEqual(len(sample["raw_prompt_ids"]), 8192)

    def test_disabling_filter_preserves_overlength_error(self):
        dataset = self.make_dataset(
            [self.row(0, 1196, image=True)], FakeProcessor(), filter_overlong_prompts=False
        )
        self.assertEqual(len(dataset), 1)
        with self.assertRaisesRegex(NotImplementedError, "sequence_length=9196 is larger than max_length=8192"):
            dataset[0]

    def test_video_expansion_uses_processed_frames(self):
        rows = [self.row(0, 192), self.row(1, 193)]
        for row in rows:
            row["prompt"][0]["content"] += " <video>"
            row["videos"] = [{"video": ["frame-1.png", "frame-2.png"]}]
        frames = torch.ones((2, 3, 28, 28))
        with patch("verl.utils.dataset.vision_utils.process_video", return_value=frames) as process_video:
            dataset = self.make_dataset(rows, FakeProcessor())
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(process_video.call_count, 3)
        self.assertEqual(sample["attention_mask"].sum().item(), 8192)
        self.assertEqual(sample["multi_modal_data"]["video"][0].shape, (2, 3, 28, 28))


if __name__ == "__main__":
    unittest.main()
