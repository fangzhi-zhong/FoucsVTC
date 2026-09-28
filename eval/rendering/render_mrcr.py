#!/usr/bin/env python3
"""Render OpenAI MRCR conversation histories as VTC page images.

The long context is rendered
to A4 PNG pages and replaced by one ``<image>`` placeholder per page, while the
final retrieval request remains text.  Output is written to::

    MRCR_VTC/dpi_72/{2needle,4needle,8needle}/test.jsonl
    MRCR_VTC/dpi_72/images/<subset>/<index>/page_0001.png

MRCR contains very long examples (up to one million tokens).  Rasterization is
therefore done in small PDF page chunks, and every completed sample receives a
marker so ``--recover`` can safely resume an interrupted render.

The main font defaults to system DejaVu Sans; override it with ``--font-path``
or ``FOCUSVTC_FONT_PATH``. Use
``--cjk-font-path`` or ``FOCUSVTC_CJK_FONT_PATH`` for a separate CJK font;
otherwise the main font must cover all characters in the source conversations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path
from statistics import median
from xml.sax.saxutils import escape

# Keep CLI help available before loading optional rendering dependencies.
def load_dependencies() -> None:
    global np, pq, Image, convert_from_path, pdfinfo_from_path
    global colors, TA_LEFT, ParagraphStyle, getSampleStyleSheet
    global pdfmetrics, TTFont, Paragraph, SimpleDocTemplate, tqdm

    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    from pdf2image import convert_from_path, pdfinfo_from_path
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate
    from tqdm import tqdm

    Image.MAX_IMAGE_PIXELS = None


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FOCUSVTC_DATA_ROOT", str(ROOT / "datasets"))).expanduser()
DEFAULT_SRC_ROOT = DATA_ROOT / "sources" / "mrcr"
DEFAULT_DST_ROOT = DATA_ROOT / "MRCR_VTC"
FONTS: dict[str, str] = {}
SUBSETS = ("2needle", "4needle", "8needle")
TOKEN_BINS = (
    "4K-8K",
    "8K-16K",
    "16K-32K",
    "32K-64K",
    "64K-128K",
    "128K-256K",
    "256K-512K",
    "512K-1M",
)

PAGE_SIZE = (595.0, 842.0)
MARGIN_X = 10.0
MARGIN_Y = 10.0
FONT_SIZE = 9.0
LINE_HEIGHT = 10.0
PAGE_BG = "#FFFFFF"
FONT_COLOR = "#000000"
CROP_TOLERANCE = 5
RENDER_VERSION = 1

# Qwen3-VL visual-token accounting, identical to LongBench_VTC.
VISION_FACTOR = 32
VISION_MIN_PIXELS = 65536
VISION_MAX_PIXELS = 16777216

PROMPT_HEADER = (
    "The following images contain a conversation history in chronological order. "
    "Read it and answer the final user request.\n\nConversation history:\n"
)
PROMPT_TAIL = "\n\nFinal user request:\n{request}"

_RE_MULTISPACE = re.compile(r" {2,}")
_RE_CJK = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
_REGISTERED_FONTS: set[str] = set()


def register_font(font_key: str) -> str:
    font_name = f"vtc_{font_key}"
    if font_name not in _REGISTERED_FONTS:
        pdfmetrics.registerFont(TTFont(font_name, FONTS[font_key]))
        _REGISTERED_FONTS.add(font_name)
    return font_name


def pick_font(text: str) -> str:
    return "cjk" if _RE_CJK.search(text) else "primary"


def serialize_history(messages: list[dict]) -> tuple[str, str]:
    """Return a role-labelled history and the final user request."""
    if len(messages) < 2:
        raise ValueError("prompt must contain history plus a final request")
    if messages[-1].get("role") != "user":
        raise ValueError("final MRCR message is not a user request")

    rendered = []
    for message in messages[:-1]:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("MRCR messages must have string role/content fields")
        rendered.append(f"{role.capitalize()}: {content}")

    request = messages[-1].get("content")
    if not isinstance(request, str):
        raise TypeError("final MRCR request is not text")
    return "\n".join(rendered), request


def build_pdf(text: str, font_key: str, path: Path) -> None:
    font_name = register_font(font_key)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_Y,
        bottomMargin=MARGIN_Y,
    )
    style = ParagraphStyle(
        name="MRCRVTC",
        parent=getSampleStyleSheet()["Normal"],
        fontName=font_name,
        fontSize=FONT_SIZE,
        leading=LINE_HEIGHT,
        textColor=colors.HexColor(FONT_COLOR),
        backColor=colors.HexColor(PAGE_BG),
        alignment=TA_LEFT,
        wordWrap="CJK" if font_key == "cjk" else None,
    )

    text = text.replace("\xad", "").replace("\u200b", "")
    body = _RE_MULTISPACE.sub(lambda match: "&nbsp;" * len(match.group()), escape(text))
    body = body.replace("\n", "<br/>").replace("\t", "&nbsp;" * 4)

    def draw_bg(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(PAGE_BG))
        canvas.rect(0, 0, PAGE_SIZE[0], PAGE_SIZE[1], stroke=0, fill=1)
        canvas.restoreState()

    doc.build([Paragraph(body, style)], onFirstPage=draw_bg, onLaterPages=draw_bg)


def autocrop(image: Image.Image, is_last_page: bool) -> Image.Image:
    gray = np.array(image.convert("L"))
    background = np.median(gray[:2, :2])
    mask = np.abs(gray - background) > CROP_TOLERANCE
    columns = np.where(mask.any(axis=0))[0]
    if columns.size:
        right = min(image.width, int(columns[-1] + 1 + MARGIN_X))
        image = image.crop((0, 0, right, image.height))
    if is_last_page:
        rows = np.where(mask.any(axis=1))[0]
        if rows.size:
            lower = min(image.height, int(rows[-1] + MARGIN_Y))
            image = image.crop((0, 0, image.width, lower))
    return image


def smart_resize(
    width: int,
    height: int,
    factor: int = VISION_FACTOR,
    min_pixels: int = VISION_MIN_PIXELS,
    max_pixels: int = VISION_MAX_PIXELS,
) -> tuple[int, int]:
    height_bar = max(factor, round(height / factor) * factor)
    width_bar = max(factor, round(width / factor) * factor)
    if height_bar * width_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        height_bar = max(factor, math.floor(height / beta / factor) * factor)
        width_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif height_bar * width_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        height_bar = math.ceil(height * beta / factor) * factor
        width_bar = math.ceil(width * beta / factor) * factor
    return width_bar, height_bar


def count_vision_tokens(sizes: list[tuple[int, int]]) -> int:
    total = 0
    for width, height in sizes:
        width_bar, height_bar = smart_resize(width, height)
        total += (width_bar * height_bar) // (VISION_FACTOR * VISION_FACTOR)
    return total


def marker_is_valid(marker_path: Path, dpi: int, font_key: str) -> tuple[list[str], list[tuple[int, int]]] | None:
    if not marker_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker != {
            **marker,
            "render_version": RENDER_VERSION,
            "dpi": dpi,
            "font": font_key,
            "font_path": FONTS[font_key],
        }:
            return None
        pages = marker["pages"]
        paths = [str(marker_path.parent / page["name"]) for page in pages]
        sizes = [tuple(page["size"]) for page in pages]
        if not pages or not all(Path(path).is_file() for path in paths):
            return None
        return paths, sizes
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def write_marker(marker_path: Path, dpi: int, font_key: str, paths: list[str], sizes: list[tuple[int, int]]) -> None:
    payload = {
        "render_version": RENDER_VERSION,
        "dpi": dpi,
        "font": font_key,
        "font_path": FONTS[font_key],
        "pages": [
            {"name": Path(path).name, "size": list(size)}
            for path, size in zip(paths, sizes, strict=True)
        ],
    }
    temporary = marker_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(marker_path)


def render_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    poppler_path: str | None,
    chunk_pages: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    for stale in out_dir.glob("page_*.png"):
        stale.unlink()
    marker_path = out_dir / "_complete.json"
    marker_path.unlink(missing_ok=True)

    page_count = int(pdfinfo_from_path(str(pdf_path), poppler_path=poppler_path)["Pages"])
    paths: list[str] = []
    sizes: list[tuple[int, int]] = []
    for first_page in range(1, page_count + 1, chunk_pages):
        last_page = min(page_count, first_page + chunk_pages - 1)
        pages = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
            poppler_path=poppler_path,
            thread_count=1,
        )
        for offset, page in enumerate(pages):
            page_number = first_page + offset
            image = autocrop(page, is_last_page=(page_number == page_count))
            if min(image.size) <= 1:
                raise RuntimeError(f"bad rendered page {page_number}")
            path = out_dir / f"page_{page_number:04d}.png"
            image.save(path, "PNG")
            paths.append(str(path))
            sizes.append(image.size)
            image.close()
            page.close()
        del pages
        gc.collect()
    return paths, sizes


def make_output_row(job: dict, dpi: int, font_key: str, history: str, request: str, paths: list[str], sizes: list[tuple[int, int]]) -> dict:
    source = job["row"]
    question = PROMPT_HEADER + "<image>" * len(paths) + PROMPT_TAIL.format(request=request)
    return {
        "index": job["index"],
        "_id": f"{job['subset']}-{job['index']:04d}",
        "dataset": "mrcr",
        "subset": job["subset"],
        "length_bin": job["length_bin"],
        "dpi": dpi,
        "font": font_key,
        "question": question,
        "answers": [source["answer"]],
        "answer": source["answer"],
        "random_string_to_prepend": source["random_string_to_prepend"],
        "n_needles": source["n_needles"],
        "desired_msg_index": source["desired_msg_index"],
        "total_messages": source["total_messages"],
        "rendered_messages": source["total_messages"] - 1,
        "n_chars": source["n_chars"],
        "context_chars": len(history),
        "date_added": source["date_added"],
        "source_file": job["source_file"],
        "source_row": job["source_row"],
        "images": paths,
        "num_pages": len(paths),
        "image_sizes": [list(size) for size in sizes],
        "vision_tokens": count_vision_tokens(sizes),
    }


def process_one(job: dict) -> dict | None:
    source = job["row"]
    dst_root = Path(job["dst_root"])
    dpi = job["dpi"]
    try:
        messages = json.loads(source["prompt"])
        if source["total_messages"] != len(messages):
            raise ValueError(f"total_messages={source['total_messages']} but parsed {len(messages)}")
        history, request = serialize_history(messages)
        font_key = pick_font(history)
        out_dir = dst_root / f"dpi_{dpi}" / "images" / job["subset"] / str(job["index"])
        out_dir.mkdir(parents=True, exist_ok=True)
        marker_path = out_dir / "_complete.json"

        recovered = marker_is_valid(marker_path, dpi, font_key) if job["recover"] else None
        if recovered is not None:
            paths, sizes = recovered
        else:
            with tempfile.NamedTemporaryFile(prefix="mrcr_vtc_", suffix=".pdf", delete=False) as handle:
                pdf_path = Path(handle.name)
            try:
                build_pdf(history, font_key, pdf_path)
                paths, sizes = render_pdf(
                    pdf_path,
                    out_dir,
                    dpi,
                    job["poppler_path"],
                    job["chunk_pages"],
                )
                write_marker(marker_path, dpi, font_key, paths, sizes)
            finally:
                pdf_path.unlink(missing_ok=True)

        return make_output_row(job, dpi, font_key, history, request, paths, sizes)
    except Exception as error:  # one bad sample must not discard completed work
        print(
            f"[ERROR] subset={job['subset']} index={job['index']} "
            f"source={job['source_file']}#{job['source_row']}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return None


def build_bin_map(src_root: Path, subset: str) -> dict[tuple[str, int], str]:
    """Map shuffled 100-row source blocks back to the eight official bins."""
    blocks = []
    for path in sorted((src_root / subset).glob("*.parquet")):
        lengths = pq.read_table(path, columns=["n_chars"])["n_chars"].to_pylist()
        if len(lengths) % 100:
            raise ValueError(f"{path}: row count is not divisible by 100")
        for start in range(0, len(lengths), 100):
            block = lengths[start : start + 100]
            blocks.append((median(block), path.name, start))
    if len(blocks) != len(TOKEN_BINS):
        raise ValueError(f"{subset}: expected 8 length blocks, found {len(blocks)}")
    blocks.sort()
    return {(filename, start): token_bin for token_bin, (_, filename, start) in zip(TOKEN_BINS, blocks, strict=True)}


def iter_jobs(
    src_root: Path,
    dst_root: Path,
    subset: str,
    dpi: int,
    recover: bool,
    poppler_path: str | None,
    chunk_pages: int,
    num_samples: int,
):
    bin_map = build_bin_map(src_root, subset)
    index = 0
    for path in sorted((src_root / subset).glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        source_row = 0
        for batch in parquet.iter_batches(batch_size=1):
            if num_samples and index >= num_samples:
                return
            block_start = (source_row // 100) * 100
            yield {
                "index": index,
                "subset": subset,
                "length_bin": bin_map[(path.name, block_start)],
                "source_file": f"{subset}/{path.name}",
                "source_row": source_row,
                "row": batch.to_pylist()[0],
                "dst_root": str(dst_root),
                "dpi": dpi,
                "recover": recover,
                "poppler_path": poppler_path,
                "chunk_pages": chunk_pages,
            }
            index += 1
            source_row += 1


def initialize_worker(fonts: dict[str, str], pdf_builder, render_version: int) -> None:
    global FONTS, build_pdf, RENDER_VERSION
    load_dependencies()
    FONTS = fonts
    build_pdf = pdf_builder
    RENDER_VERSION = render_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--out-root", "--dst-root", dest="dst_root", type=Path, default=DEFAULT_DST_ROOT)
    parser.add_argument("--font-path", type=Path,
                        default=os.environ.get("FOCUSVTC_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                        help="TrueType font; defaults to FOCUSVTC_FONT_PATH or system DejaVu Sans")
    parser.add_argument("--cjk-font-path", type=Path, default=os.environ.get("FOCUSVTC_CJK_FONT_PATH"),
                        help="CJK TrueType font; defaults to FOCUSVTC_CJK_FONT_PATH or the main font")
    parser.add_argument("--dpi", type=int, default=72)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--num-samples", type=int, default=0, help="cap each subset; 0 renders all")
    parser.add_argument("--processes", type=int, default=48)
    parser.add_argument("--chunk-pages", type=int, default=16)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    if args.font_path is None or not args.font_path.expanduser().is_file():
        parser.error("set --font-path or FOCUSVTC_FONT_PATH to a readable .ttf file")
    font_path = args.font_path.expanduser().resolve()
    cjk_font_path = (args.cjk_font_path or font_path).expanduser().resolve()
    if not cjk_font_path.is_file():
        parser.error("--cjk-font-path or FOCUSVTC_CJK_FONT_PATH must name a readable .ttf file")
    if args.dpi <= 0 or args.processes <= 0 or args.chunk_pages <= 0 or args.num_samples < 0:
        parser.error("DPI, processes and chunk-pages must be positive; num-samples must be nonnegative")
    args.src_root = args.src_root.expanduser().resolve()
    args.dst_root = args.dst_root.expanduser().resolve()
    FONTS.update(primary=str(font_path), cjk=str(cjk_font_path))
    load_dependencies()

    env_bin = Path(sys.executable).parent
    poppler_path = str(env_bin) if (env_bin / "pdftoppm").exists() else None
    if not args.src_root.is_dir():
        raise SystemExit(f"source dataset not found: {args.src_root}")

    for subset in args.subsets:
        output_dir = args.dst_root / f"dpi_{args.dpi}" / subset
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test.jsonl"
        temporary_path = output_path.with_suffix(".jsonl.tmp")
        jobs = iter_jobs(
            args.src_root,
            args.dst_root,
            subset,
            args.dpi,
            args.recover,
            poppler_path,
            args.chunk_pages,
            args.num_samples,
        )
        expected = args.num_samples or 800
        completed = []
        with Pool(
            processes=args.processes,
            initializer=initialize_worker,
            initargs=(FONTS, build_pdf, RENDER_VERSION),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(process_one, jobs, chunksize=1),
                total=expected,
                desc=subset,
            ):
                if result is not None:
                    completed.append(result)

        completed.sort(key=lambda row: row["index"])
        with temporary_path.open("w", encoding="utf-8") as output:
            for row in completed:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
        print(f"{subset}: {len(completed)}/{expected} rendered -> {output_path}", flush=True)
        if len(completed) != expected:
            raise SystemExit(f"{subset}: {expected - len(completed)} sample(s) failed")


if __name__ == "__main__":
    main()
