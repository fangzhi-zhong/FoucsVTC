import importlib.util
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "verl/workers/agent/envs/mm_process_engine/vtc_zoom.py"
)
SPEC = importlib.util.spec_from_file_location("vtc_zoom_under_test", MODULE_PATH)
ZOOM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ZOOM)


class VTCZoomTest(unittest.TestCase):
    def test_normalized_bbox_to_pixels(self):
        actual = ZOOM.normalized_bbox_to_pixels([100, 200, 500, 600], 100, 200, padding_ratio=0)
        self.assertEqual(actual, (10, 40, 50, 120))

    def test_execute_zoom_uses_one_based_page(self):
        first = Image.new("RGB", (100, 100), "red")
        second = Image.new("RGB", (100, 100), "blue")
        action = (
            '<tool_call>{"name":"zoom_region","arguments":'
            '{"page":2,"bbox_2d":[100,100,500,500]}}</tool_call>'
        )
        crop, info = ZOOM.execute_zoom(action, [first, second], padding_ratio=0)
        self.assertEqual(info["page"], 2)
        self.assertEqual(crop.size, (40, 40))
        self.assertEqual(crop.getpixel((0, 0)), (0, 0, 255))

    def test_execute_zoom_pads_extreme_aspect_ratio(self):
        # A very thin model-generated box can produce a crop that qwen_vl_utils
        # refuses before it gets to the processor (MAX_RATIO is 200).
        page = Image.new("RGB", (1200, 1000), "white")
        action = (
            '<tool_call>{"name":"zoom_region","arguments":'
            '{"page":1,"bbox_2d":[0,0,1000,1]}}</tool_call>'
        )
        crop, info = ZOOM.execute_zoom(action, [page], padding_ratio=0)
        self.assertLess(max(crop.size) / min(crop.size), 200)
        self.assertEqual(info["crop_size"], list(crop.size))

        vertical_action = (
            '<tool_call>{"name":"zoom_region","arguments":'
            '{"page":1,"bbox_2d":[0,0,1,1000]}}</tool_call>'
        )
        vertical_crop, _ = ZOOM.execute_zoom(vertical_action, [page], padding_ratio=0)
        self.assertLess(max(vertical_crop.size) / min(vertical_crop.size), 200)

    def test_padded_extreme_crop_is_accepted_by_qwen_processor(self):
        try:
            from qwen_vl_utils import fetch_image
        except ImportError:
            self.skipTest("qwen_vl_utils is not installed")

        page = Image.new("RGB", (1200, 1000), "white")
        action = (
            '<tool_call>{"name":"zoom_region","arguments":'
            '{"page":1,"bbox_2d":[0,0,1000,1]}}</tool_call>'
        )
        crop, _ = ZOOM.execute_zoom(action, [page], padding_ratio=0)
        processed = fetch_image({"image": crop})
        self.assertGreater(processed.width, 0)
        self.assertGreater(processed.height, 0)

    def test_invalid_page_is_rejected(self):
        action = (
            '<tool_call>{"name":"zoom_region","arguments":'
            '{"page":0,"bbox_2d":[100,100,500,500]}}</tool_call>'
        )
        with self.assertRaisesRegex(ZOOM.ZoomToolError, "page must be in"):
            ZOOM.execute_zoom(action, [Image.new("RGB", (100, 100))])

    def test_incomplete_tool_call_is_rejected(self):
        with self.assertRaisesRegex(ZOOM.ZoomToolError, "no complete"):
            ZOOM.execute_zoom("<tool_call>{}", [Image.new("RGB", (100, 100))])


if __name__ == "__main__":
    unittest.main()
