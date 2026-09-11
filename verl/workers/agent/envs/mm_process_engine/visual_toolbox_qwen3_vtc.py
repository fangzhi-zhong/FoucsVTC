"""Qwen3.5-VL VTC zoom tool used by the verl agent rollout.

The model starts with low-resolution document pages. A tool call selects a
page and normalized box; the tool crops the corresponding high-resolution page
and appends it as a new visual observation. Invalid or low-IoU calls return
feedback instead of terminating immediately, which gives GRPO a chance to
learn retry behaviour. Reward is computed by the terminal custom reward
function; this environment only returns observations and diagnostic IoU.
"""

import os
from copy import deepcopy

import numpy as np
from PIL import Image

from verl.workers.agent.envs.mm_process_engine.vtc_zoom import (
    TOOL_NAME,
    TOOL_SCHEMA,
    ZoomToolError,
    execute_zoom,
    extract_tool_call,
)
from verl.workers.agent.tool_envs import ToolBase


class _LazyPages:
    """Sequence facade that decodes only the page selected by the action."""

    def __init__(self, refs):
        self.refs = refs

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        return Qwen3VLVTCZoomTool._load_page(self.refs[index])


class Qwen3VLVTCZoomTool(ToolBase):
    name = "qwen3_vl_vtc_zoom"

    def __init__(self, _name=None, _description="", _parameters=None, **kwargs):
        function = TOOL_SCHEMA["function"]
        super().__init__(name=self.name, description=function["description"], parameters=function["parameters"])
        self.pages = []
        self.high_res_pages = []
        self.evidence = []
        self.call_count = 0
        self.max_calls = 3
        self.history = []

    def reset(self, raw_prompt=None, multi_modal_data=None, origin_multi_modal_data=None, **kwargs):
        source = origin_multi_modal_data or multi_modal_data or {}
        self.pages = list(source.get("image", []) or [])
        self.high_res_pages = list(source.get("high_res_image", source.get("high_res_images", [])) or [])
        if not self.high_res_pages:
            self.high_res_pages = list(self.pages)

        metadata = kwargs.get("tool_meta") or kwargs.get("extra_info") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        elif isinstance(metadata.get("metadata"), dict):
            metadata = metadata["metadata"]
        self.evidence = self._extract_evidence(metadata)
        # The rollout configuration takes precedence over dataset metadata.
        self.max_calls = max(1, int(kwargs.get("max_tool_calls", metadata.get("max_tool_calls", 3))))
        self.call_count = 0
        self.history = []
        if not self.pages:
            raise ValueError("qwen3_vl_vtc_zoom requires at least one original page image")

    @staticmethod
    def _extract_evidence(metadata):
        """Normalize common RULER evidence layouts to ``(page, bbox)`` pairs."""

        pairs = []
        by_dpi = metadata.get("evidence_bboxes_by_dpi", {})
        if isinstance(by_dpi, dict):
            candidates = []
            for key, value in by_dpi.items():
                digits = "".join(ch for ch in str(key) if ch.isdigit())
                candidates.append((int(digits or 0), value))
            for _, value in sorted(candidates, reverse=True):
                pairs = Qwen3VLVTCZoomTool._pairs_from_layout(value)
                if pairs:
                    break
        if not pairs:
            pairs = Qwen3VLVTCZoomTool._pairs_from_layout(metadata.get("evidence_locations"))
        return pairs

    @staticmethod
    def _pairs_from_layout(layout):
        if isinstance(layout, np.ndarray):
            layout = layout.tolist()
        pairs = []
        if isinstance(layout, dict):
            if "page" in layout or "page_id" in layout:
                page = layout.get("page", layout.get("page_id"))
                boxes = layout.get("bbox_2d", layout.get("bbox", layout.get("bboxes", [])))
                if isinstance(boxes, np.ndarray):
                    boxes = boxes.tolist()
                if isinstance(boxes, (list, tuple)) and len(boxes) == 4 and all(
                    isinstance(x, (int, float)) for x in boxes
                ):
                    boxes = [boxes]
                for box in boxes or []:
                    if isinstance(box, np.ndarray):
                        box = box.tolist()
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        pairs.append((int(page), [float(x) for x in box]))
            else:
                for page, boxes in layout.items():
                    if isinstance(boxes, np.ndarray):
                        boxes = boxes.tolist()
                    if isinstance(boxes, (list, tuple)) and len(boxes) == 4 and all(
                        isinstance(x, (int, float)) for x in boxes
                    ):
                        boxes = [boxes]
                    for box in boxes or []:
                        if isinstance(box, np.ndarray):
                            box = box.tolist()
                        if isinstance(box, (list, tuple)) and len(box) == 4:
                            pairs.append((int(page), [float(x) for x in box]))
        elif isinstance(layout, (list, tuple)):
            for item in layout:
                if isinstance(item, np.ndarray):
                    item = item.tolist()
                if isinstance(item, dict):
                    pairs.extend(Qwen3VLVTCZoomTool._pairs_from_layout(item))
                elif isinstance(item, (list, tuple)) and len(item) == 5:
                    pairs.append((int(item[0]), [float(x) for x in item[1:]]))
        return pairs

    @staticmethod
    def _load_page(page):
        if isinstance(page, Image.Image):
            return page.convert("RGB")
        if isinstance(page, (str, os.PathLike)):
            with Image.open(page) as image:
                return image.convert("RGB").copy()
        if isinstance(page, dict):
            if isinstance(page.get("image"), (str, os.PathLike)):
                return Qwen3VLVTCZoomTool._load_page(page["image"])
            if "bytes" in page:
                from io import BytesIO

                with Image.open(BytesIO(page["bytes"])) as image:
                    return image.convert("RGB").copy()
        raise ZoomToolError(f"unsupported page image type: {type(page)!r}")

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _local_reward(self, page, bbox):
        if not self.evidence:
            return 0.0, 0.0
        page_ious = [self._iou(bbox, target) for target_page, target in self.evidence if target_page == page]
        best_iou = max(page_ious, default=0.0)
        # Keep IoU for diagnostics; the terminal reward owns all reward terms.
        return 0.0, best_iou

    @staticmethod
    def _observation_prompt(action_text, text, has_image):
        # Pass a prompt directly rather than applying a standalone chat
        # template: Qwen3.5 rejects a conversation beginning with only a tool
        # role. ``<image>`` is replaced by the processor helper and remains a
        # single placeholder for vLLM.
        close_assistant = "" if action_text.rstrip().endswith("<|im_end|>") else "<|im_end|>\n"
        image_part = "<image>\n" if has_image else ""
        return (
            f"{close_assistant}<|im_start|>user\n<tool_response>\n"
            f"{image_part}{text}\n</tool_response>\n"
            "Think first. During thinking, point out the relevant page or evidence location when needed. "
            "Then call zoom_region if needed, and answer.\n\n"
            "Format strictly as: <think>...</think> <tool_call>...</tool_call> "
            "(if tools are needed) <answer>...</answer>.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )

    def execute(self, action_string: str, **kwargs):
        if "<tool_call>" not in action_string:
            return "", 0.0, True, {"status": "final"}
        if self.call_count >= self.max_calls:
            return "Error: maximum zoom attempts reached; provide your best final answer.", 0.0, True, {
                "status": "failed",
                "error": "tool_call_limit",
            }

        try:
            payload = extract_tool_call(action_string)
            if payload is None:
                raise ZoomToolError("no complete tool call found")
            page_number = payload.get("arguments", {}).get("page") if isinstance(payload, dict) else None
            source_pages = self.high_res_pages if page_number and page_number <= len(self.high_res_pages) else self.pages
            crop, info = execute_zoom(action_string, _LazyPages(source_pages))
        except (ZoomToolError, TypeError, ValueError, OSError) as exc:
            self.call_count += 1
            done = self.call_count >= self.max_calls
            feedback = self._observation_prompt(
                action_string, f"Zoom failed: {exc}. Try a different page or box.", False
            )
            info = {"status": "failed", "error": str(exc), "call_count": self.call_count}
            self.history.append(info)
            return {"prompt": feedback}, 0.0, done, info

        self.call_count += 1
        shaping_reward, best_iou = self._local_reward(info["page"], info["bbox_2d"])
        info.update({"best_iou": best_iou, "call_count": self.call_count})
        self.history.append(deepcopy(info))
        text = (
            f"Result from {TOOL_NAME} on page {info['page']} (crop {info['crop_size'][0]}x{info['crop_size'][1]}). "
            "Inspect the crop and answer the original question; retry zoom_region if it is still unreadable."
        )
        observation = {
            "prompt": self._observation_prompt(action_string, text, True),
            "multi_modal_data": {"image": [crop]},
        }
        # Do not terminate after a successful crop: the model must be able to
        # emit a final answer or request another crop.
        return observation, shaping_reward, False, info
