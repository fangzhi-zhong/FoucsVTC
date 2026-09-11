"""Count Qwen image tokens from headers without decoding image pixels."""

import base64
import math
import os
from collections.abc import Mapping
from io import BytesIO

from PIL import Image


def _image_size(image):
    """Return (width, height); opening a PIL header does not decode pixels."""
    if isinstance(image, Image.Image):
        return image.size
    if isinstance(image, (bytes, bytearray, memoryview)):
        with BytesIO(bytes(image)) as stream, Image.open(stream) as opened:
            return opened.size
    if isinstance(image, (str, os.PathLike)):
        path = os.fspath(image)
        if isinstance(path, str) and path.startswith(("http://", "https://")):
            raise ValueError("Image length estimation requires a local image; remote URLs are unsupported")
        with Image.open(path) as opened:
            return opened.size
    raise TypeError(f"Unsupported image input for header-only length estimation: {type(image).__name__}")


def _smart_resize(height, width, factor, min_pixels, max_pixels, *, fetch_image=False):
    """Match the size arithmetic in Qwen's processor and fetch_image.

    qwen_vl_utils.fetch_image clamps the initial rounded dimensions, whereas
    Transformers' Qwen2VLImageProcessor clamps dimensions after downscaling.
    Keep that distinction without importing either pixel-processing module.
    """
    if height <= 0 or width <= 0 or factor <= 0 or not 0 < min_pixels <= max_pixels:
        raise ValueError("Image dimensions, patch factor and pixel limits must be positive and valid")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must not exceed 200")
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if fetch_image:
        h_bar, w_bar = max(factor, h_bar), max(factor, w_bar)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
        if not fetch_image:
            h_bar, w_bar = max(factor, h_bar), max(factor, w_bar)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _size_after_process_image(image):
    """Mirror vision_utils.process_image's direct versus fetch_image paths."""
    if not isinstance(image, Mapping):
        return _image_size(image)
    if isinstance(image.get("image"), (str, os.PathLike)):
        # process_image opens these directly, ignoring per-dict resize options.
        return _image_size(image["image"])
    if "bytes" in image:
        if "image" in image:
            raise ValueError("Cannot have both `bytes` and `image` in an image record")
        source = image["bytes"]
    else:
        source = image.get("image", image.get("image_url"))
        if isinstance(source, str) and source.startswith("file://"):
            source = source[7:]
        elif isinstance(source, str) and source.startswith("data:image"):
            if "base64," not in source:
                raise ValueError("Only base64 image data URLs are supported")
            source = base64.b64decode(source.split("base64,", 1)[1], validate=True)
    width, height = _image_size(source)
    # vision_utils calls fetch_image without image_patch_size: its default is
    # 14, with merge factor 2, even when the subsequent processor uses patch16.
    factor = 28
    if "resized_height" in image and "resized_width" in image:
        height, width = image["resized_height"], image["resized_width"]
        min_pixels, max_pixels = 4 * factor**2, 16384 * factor**2
    else:
        min_pixels = image.get("min_pixels", 4 * factor**2)
        max_pixels = image.get("max_pixels", 16384 * factor**2)
    height, width = _smart_resize(height, width, factor, min_pixels, max_pixels, fetch_image=True)
    return width, height


def _size_option(size, name):
    return size.get(name) if isinstance(size, Mapping) else getattr(size, name, None)


def estimate_image_tokens(image, image_processor) -> int:
    """Return merged Qwen visual tokens using only image header dimensions.

    ``image_processor`` is the processor's image processor, with ``patch_size``,
    ``merge_size``, ``do_resize`` and ``size`` (or legacy pixel-limit fields).
    Paths, PIL images, encoded bytes and vision_utils image dictionaries are
    supported. Dicts taking the fetch_image path include its first resize.
    Bare encoded bytes are interpreted as an unresized image; dataset records
    using ``{"bytes": ...}`` retain their actual fetch_image preprocessing.

    This returns image-pad tokens only: callers must also count text, chat
    template and image boundary tokens. Unknown formats fail explicitly;
    there is no processor/decode fallback or remote image download.
    """
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if not isinstance(patch_size, int) or not isinstance(merge_size, int) or min(patch_size, merge_size) <= 0:
        raise ValueError("Image token estimation requires a Qwen image processor with patch_size and merge_size")
    factor = patch_size * merge_size
    width, height = _size_after_process_image(image)
    if getattr(image_processor, "do_resize", True):
        size = getattr(image_processor, "size", None)
        min_pixels = _size_option(size, "shortest_edge")
        max_pixels = _size_option(size, "longest_edge")
        if min_pixels is None:
            min_pixels = getattr(image_processor, "min_pixels", None)
        if max_pixels is None:
            max_pixels = getattr(image_processor, "max_pixels", None)
        if min_pixels is None or max_pixels is None:
            raise ValueError("Image processor must provide shortest_edge/longest_edge or min_pixels/max_pixels")
        height, width = _smart_resize(height, width, factor, min_pixels, max_pixels)
    if height <= 0 or width <= 0 or height % factor or width % factor:
        raise ValueError(f"Image size {(width, height)} must align to patch_size * merge_size = {factor}")
    return (height // factor) * (width // factor)
