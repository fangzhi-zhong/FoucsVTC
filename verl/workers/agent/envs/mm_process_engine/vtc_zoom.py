"""Pure document zoom protocol shared by the verl env and standalone smoke test."""

import json
import math
import re
import ast
from typing import Any, Sequence

from PIL import Image


TOOL_NAME = "zoom_region"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Crop a potentially unreadable region from a document page and return it as a new image.",
        "parameters": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "1-based page number."},
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "[x1, y1, x2, y2] normalized to [0, 1000].",
                },
            },
            "required": ["page", "bbox_2d"],
        },
    },
}


class ZoomToolError(ValueError):
    """Raised when a model action violates the frozen zoom protocol."""


def extract_tool_call(action_text: str) -> dict[str, Any] | None:
    """Parse the last Qwen3/Qwen3.5 ``<tool_call>`` block.

    Qwen3.5's native chat template serializes tools as XML-like tags such as
    ``<function=zoom_region><parameter=page>1`` rather than the JSON payload
    used by older Qwen checkpoints.  We accept both wire formats and normalize
    them to ``{"name": ..., "arguments": ...}`` for the validator below.
    """

    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", action_text, flags=re.DOTALL)
    if not matches:
        return None
    block = matches[-1].strip()
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        function_match = re.search(r"<function\s*=\s*([A-Za-z0-9_.:-]+)\s*>", block)
        if function_match is None:
            raise ZoomToolError("tool call is neither valid JSON nor Qwen XML")
        arguments: dict[str, Any] = {}
        for parameter_match in re.finditer(
            r"<parameter\s*=\s*([A-Za-z0-9_.:-]+)\s*>\s*(.*?)\s*</parameter>",
            block,
            flags=re.DOTALL,
        ):
            key = parameter_match.group(1)
            raw_value = parameter_match.group(2).strip()
            try:
                # JSON handles arrays/numbers and is stricter than Python eval.
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(raw_value)
                except (SyntaxError, ValueError):
                    value = raw_value
            arguments[key] = value
        payload = {"name": function_match.group(1), "arguments": arguments}
    if not isinstance(payload, dict):
        raise ZoomToolError("tool call must decode to an object")
    return payload


def _validate_payload(payload: dict[str, Any], page_count: int) -> tuple[int, list[float]]:
    if payload.get("name") != TOOL_NAME:
        raise ZoomToolError(f"unknown tool name: {payload.get('name')!r}")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ZoomToolError("arguments must be a JSON object")

    page = arguments.get("page")
    if isinstance(page, bool) or not isinstance(page, int):
        raise ZoomToolError("page must be an integer")
    if page < 1 or page > page_count:
        raise ZoomToolError(f"page must be in [1, {page_count}], got {page}")

    bbox = arguments.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ZoomToolError("bbox_2d must contain exactly four numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        raise ZoomToolError("bbox_2d values must be numbers")
    bbox = [float(value) for value in bbox]
    if any(not math.isfinite(value) or value < 0 or value > 1000 for value in bbox):
        raise ZoomToolError("bbox_2d values must be finite and within [0, 1000]")
    left, top, right, bottom = bbox
    if left >= right or top >= bottom:
        raise ZoomToolError("bbox_2d must satisfy x1 < x2 and y1 < y2")
    return page, bbox


def normalized_bbox_to_pixels(
    bbox: Sequence[float], width: int, height: int, padding_ratio: float = 0.05
) -> tuple[int, int, int, int]:
    """Convert a [0, 1000] bbox to a clipped pixel bbox with context padding."""

    left, top, right, bottom = bbox
    left_px = left * width / 1000.0
    top_px = top * height / 1000.0
    right_px = right * width / 1000.0
    bottom_px = bottom * height / 1000.0
    pad_x = (right_px - left_px) * padding_ratio
    pad_y = (bottom_px - top_px) * padding_ratio

    pixel_bbox = (
        max(0, math.floor(left_px - pad_x)),
        max(0, math.floor(top_px - pad_y)),
        min(width, math.ceil(right_px + pad_x)),
        min(height, math.ceil(bottom_px + pad_y)),
    )
    if pixel_bbox[0] >= pixel_bbox[2] or pixel_bbox[1] >= pixel_bbox[3]:
        raise ZoomToolError(f"bbox becomes empty after pixel conversion: {pixel_bbox}")
    return pixel_bbox


def _pad_to_max_aspect_ratio(
    image: Image.Image, max_aspect_ratio: float = 199.0
) -> Image.Image:
    """Pad an extreme crop so Qwen's image processor can resize it.

    ``qwen_vl_utils.smart_resize`` rejects images whose absolute aspect ratio
    is 200 or larger.  A model-generated bbox can be only a few pixels tall or
    wide after conversion, even when the source page itself is well behaved.
    Keep the crop centered and add white context only when it is necessary.
    """

    if max_aspect_ratio <= 1:
        raise ValueError(f"max_aspect_ratio must be greater than 1, got {max_aspect_ratio}")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ZoomToolError(f"crop has invalid size: {image.size}")
    if max(width / height, height / width) < max_aspect_ratio:
        return image

    if width >= height:
        padded_height = max(height, math.ceil(width / max_aspect_ratio))
        canvas = Image.new(image.mode, (width, padded_height), "white")
        canvas.paste(image, (0, (padded_height - height) // 2))
    else:
        padded_width = max(width, math.ceil(height / max_aspect_ratio))
        canvas = Image.new(image.mode, (padded_width, height), "white")
        canvas.paste(image, ((padded_width - width) // 2, 0))
    return canvas


def execute_zoom(
    action_text: str, pages: Sequence[Image.Image], padding_ratio: float = 0.05
) -> tuple[Image.Image, dict[str, Any]]:
    """Execute one strict Qwen tool action against the original document pages."""

    if not pages:
        raise ZoomToolError("no document pages are available")
    payload = extract_tool_call(action_text)
    if payload is None:
        raise ZoomToolError("no complete <tool_call> block found")
    if not isinstance(payload, dict) or not payload.get("name") or "arguments" not in payload:
        raise ZoomToolError("no complete tool call found")
    page, normalized_bbox = _validate_payload(payload, len(pages))
    source = pages[page - 1].convert("RGB")
    pixel_bbox = normalized_bbox_to_pixels(normalized_bbox, source.width, source.height, padding_ratio)
    crop = _pad_to_max_aspect_ratio(source.crop(pixel_bbox).copy())
    info = {
        "status": "success",
        "tool_used": TOOL_NAME,
        "page": page,
        "bbox_2d": normalized_bbox,
        "bbox_pixels": list(pixel_bbox),
        "source_size": [source.width, source.height],
        "crop_size": [crop.width, crop.height],
    }
    return crop, info
