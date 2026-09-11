"""Header-only visual-token counts must agree with Qwen resizing semantics."""

from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from verl.utils.dataset.prompt_length import estimate_image_tokens


class PromptLengthEstimateTest(unittest.TestCase):
    def setUp(self):
        self.processor = SimpleNamespace(
            patch_size=16, merge_size=2, do_resize=True,
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        )

    def test_local_inputs_read_headers_only(self):
        image = Image.new("RGB", (1000, 700))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            image.save(path)
            encoded = path.read_bytes()

            def forbid_pixels(*args, **kwargs):
                raise AssertionError("Length estimation must not decode/resize image pixels")

            with patch.object(Image.Image, "load", forbid_pixels), patch.object(Image.Image, "resize", forbid_pixels), patch.object(Image.Image, "convert", forbid_pixels):
                for value in [path, str(path), image, encoded, {"image": str(path), "resized_height": 224, "resized_width": 224}]:
                    # Direct process_image inputs: 1000x700 -> 992x704 after factor32 rounding.
                    self.assertEqual(estimate_image_tokens(value, self.processor), 31 * 22)

    def test_matches_processor_resize_for_pixel_limits(self):
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        for dimensions in [(31, 32), (1178, 1684), (9000, 9000), (800, 400)]:
            with self.subTest(dimensions=dimensions):
                width, height = dimensions
                expected_h, expected_w = smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=16777216)
                # Expose header dimensions without allocating a large pixel buffer.
                image = Image.new("1", (1, 1))
                image._size = dimensions
                self.assertEqual(estimate_image_tokens(image, self.processor), expected_h * expected_w // 1024)

    def test_dict_fetch_image_resize_is_counted(self):
        from qwen_vl_utils.vision_process import smart_resize as fetch_resize
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as processor_resize

        image = Image.new("RGB", (1000, 700))
        stream = BytesIO()
        image.save(stream, format="PNG")
        for options in [{}, {"resized_height": 300, "resized_width": 420}, {"min_pixels": 100000, "max_pixels": 200000}]:
            with self.subTest(options=options):
                if "resized_height" in options:
                    first_h, first_w = fetch_resize(options["resized_height"], options["resized_width"], factor=28)
                else:
                    first_h, first_w = fetch_resize(700, 1000, factor=28, **options)
                final_h, final_w = processor_resize(first_h, first_w, factor=32, min_pixels=65536, max_pixels=16777216)
                expected = final_h * final_w // 1024
                for source in [{"bytes": stream.getvalue()}, {"image": image}]:
                    self.assertEqual(estimate_image_tokens({**source, **options}, self.processor), expected)

    def test_resize_disabled_and_unsupported_input(self):
        self.processor.do_resize = False
        self.assertEqual(estimate_image_tokens(Image.new("RGB", (64, 96)), self.processor), 6)
        with self.assertRaisesRegex(ValueError, "must align"):
            estimate_image_tokens(Image.new("RGB", (63, 96)), self.processor)
        with self.assertRaisesRegex(ValueError, "remote URLs"):
            estimate_image_tokens("https://example.com/page.png", self.processor)
        with self.assertRaisesRegex(TypeError, "Unsupported image"):
            estimate_image_tokens(object(), self.processor)


if __name__ == "__main__":
    unittest.main()
