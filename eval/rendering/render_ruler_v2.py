#!/usr/bin/env python3
"""Render the haystack of RULER v2 prompts to PNG pages at several DPIs.

Layout uses A4 pages, 10pt margins, a user-provided 9pt TrueType font, 10pt
leading, left alignment, a white background, and automatic cropping.

The PDF is built once per sample and rasterized at every DPI, so page breaks are
identical across DPIs and DPI stays the only independent variable.

Input: <src-root>/<task>/test.jsonl, with index, question, expected_answer,
and length fields. Only the haystack is rendered; the exact instruction header
and query tail are retained around the page placeholders in each output row.

Output layout:
    RULER_v2_VTC/dpi_{dpi}/{task}/test.jsonl
    RULER_v2_VTC/dpi_{dpi}/images/{task}/{index}/page_001.png
"""

import argparse
import gc
import importlib
import io
import json
import math
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path
from xml.sax.saxutils import escape

if __package__:
    from .ruler2_split import SPEC, split_prompt
else:
    from ruler2_split import SPEC, split_prompt

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FOCUSVTC_DATA_ROOT", ROOT / "datasets")).expanduser()
FONT_NAME = "vtc_font"

PAGE_SIZE = (595.0, 842.0)
MARGIN_X = 10.0
MARGIN_Y = 10.0
FONT_SIZE = 9.0
LINE_HEIGHT = 10.0
PAGE_BG = "#FFFFFF"
FONT_COLOR = "#000000"
CROP_TOLERANCE = 5

# Qwen3-VL: patch_size=16, merge_size=2 -> one visual token per 32x32 px, and
# the processor snaps both sides to a multiple of 32 within [min, max] pixels
# (preprocessor_config.json: shortest_edge=65536, longest_edge=16777216).
VISION_FACTOR = 32
VISION_MIN_PIXELS = 65536
VISION_MAX_PIXELS = 16777216

SRC_ROOT = DATA_ROOT / "sources/ruler_v2"
DST_ROOT = DATA_ROOT / "RULER_v2_VTC"

# Prefer Poppler next to the interpreter when installed in a Conda environment.
_env_bin = Path(sys.executable).parent
POPPLER_PATH = str(_env_bin) if (_env_bin / "pdftoppm").exists() else None

_RE_MULTISPACE = re.compile(r" {2,}")


def build_pdf(text, font_path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    pdfmetrics.registerFont(TTFont(FONT_NAME, str(font_path)))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_Y,
        bottomMargin=MARGIN_Y,
    )
    style = ParagraphStyle(
        name="Custom",
        parent=getSampleStyleSheet()["Normal"],
        fontName=FONT_NAME,
        fontSize=FONT_SIZE,
        leading=LINE_HEIGHT,
        textColor=colors.HexColor(FONT_COLOR),
        backColor=colors.HexColor(PAGE_BG),
        alignment=TA_LEFT,
    )

    text = text.replace("\xad", "").replace("\u200b", "")
    body = _RE_MULTISPACE.sub(lambda m: "&nbsp;" * len(m.group()), escape(text))
    body = body.replace("\n", "<br/>").replace("\t", "&nbsp;" * 4)

    def draw_bg(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(PAGE_BG))
        canvas.rect(0, 0, PAGE_SIZE[0], PAGE_SIZE[1], stroke=0, fill=1)
        canvas.restoreState()

    doc.build([Paragraph(body, style)], onFirstPage=draw_bg, onLaterPages=draw_bg)
    pdf = buf.getvalue()
    buf.close()
    return pdf


def autocrop(img, is_last_page):
    import numpy as np

    gray = np.array(img.convert("L"))
    bg = np.median(gray[:2, :2])
    mask = np.abs(gray - bg) > CROP_TOLERANCE
    cols = np.where(mask.any(axis=0))[0]
    if cols.size:
        right = min(img.width, int(cols[-1] + 1 + MARGIN_X))
        img = img.crop((0, 0, right, img.height))
    if is_last_page:
        rows = np.where(mask.any(axis=1))[0]
        if rows.size:
            lower = min(img.height, int(rows[-1] + MARGIN_Y))
            img = img.crop((0, 0, img.width, lower))
    return img


def process_one(job):
    from PIL import Image
    from pdf2image import convert_from_bytes

    Image.MAX_IMAGE_PIXELS = None
    task, row, dpis, recover, out_root, font_path = job
    index = row["index"]
    try:
        header, haystack, tail = split_prompt(task, row["question"])

        expected = {}
        for dpi in dpis:
            out_dir = out_root / f"dpi_{dpi}" / "images" / task / str(index)
            expected[dpi] = out_dir
        if recover:
            done = {}
            for dpi, out_dir in expected.items():
                pages = sorted(out_dir.glob("page_*.png")) if out_dir.is_dir() else []
                done[dpi] = pages
            if all(done[dpi] for dpi in dpis) and len({len(v) for v in done.values()}) == 1:
                return _rows_from_existing(task, row, header, tail, dpis, done)

        pdf = build_pdf(haystack, font_path)
        results = {}
        for dpi in dpis:
            out_dir = expected[dpi]
            out_dir.mkdir(parents=True, exist_ok=True)
            pages = convert_from_bytes(pdf, dpi=dpi, poppler_path=POPPLER_PATH)
            paths, sizes = [], []
            for i, page in enumerate(pages, start=1):
                img = autocrop(page, is_last_page=(i == len(pages)))
                if min(img.size) <= 1:
                    raise RuntimeError(f"bad page {i} at dpi={dpi}")
                path = out_dir / f"page_{i:03d}.png"
                img.save(path, "PNG")
                paths.append(str(path))
                sizes.append(img.size)
                img.close()
            for page in pages:
                page.close()
            results[dpi] = _make_row(task, row, header, tail, dpi, paths, sizes)
        del pdf
        gc.collect()
        return results
    except Exception as exc:
        print(f"[ERROR] task={task} index={index}: {exc}", file=sys.stderr)
        return None


def smart_resize(w, h, factor=VISION_FACTOR, min_pixels=VISION_MIN_PIXELS, max_pixels=VISION_MAX_PIXELS):
    """Mirror of Qwen-VL's smart_resize, so token counts match what vLLM sees."""
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return w_bar, h_bar


def count_vision_tokens(sizes):
    total = 0
    for w, h in sizes:
        wb, hb = smart_resize(w, h)
        total += (wb * hb) // (VISION_FACTOR * VISION_FACTOR)
    return total


def _make_row(task, row, header, tail, dpi, paths, sizes):
    vision_tokens = count_vision_tokens(sizes)
    return {
        "index": row["index"],
        "task": task,
        "dpi": dpi,
        "question": header + "<image>" * len(paths) + tail,
        "expected_answer": row["expected_answer"],
        "text_length": row["length"],
        "images": paths,
        "num_pages": len(paths),
        "image_sizes": [list(s) for s in sizes],
        "vision_tokens": vision_tokens,
    }


def _rows_from_existing(task, row, header, tail, dpis, done):
    from PIL import Image

    results = {}
    for dpi in dpis:
        paths = [str(p) for p in done[dpi]]
        sizes = []
        for p in paths:
            with Image.open(p) as im:
                sizes.append(im.size)
        results[dpi] = _make_row(task, row, header, tail, dpi, paths, sizes)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-root", type=Path, default=SRC_ROOT,
                    help="source root containing <task>/test.jsonl (default: %(default)s)")
    ap.add_argument("--out-root", type=Path, default=DST_ROOT,
                    help="rendered dataset root (default: %(default)s)")
    ap.add_argument("--font-path", type=Path,
                    default=os.environ.get("FOCUSVTC_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                    help="TrueType font; defaults to FOCUSVTC_FONT_PATH or system DejaVu Sans")
    ap.add_argument("--dpis", type=int, nargs="+", default=[72, 144])
    ap.add_argument("--num-samples", type=int, default=100,
                    help="samples per task; 0 selects all rows (default: %(default)s)")
    ap.add_argument("--tasks", type=str, nargs="+", choices=list(SPEC), default=list(SPEC))
    ap.add_argument("--processes", type=int, default=48)
    ap.add_argument("--recover", action="store_true")
    args = ap.parse_args()

    if args.font_path is None:
        ap.error("provide --font-path or set FOCUSVTC_FONT_PATH to a readable TrueType font")
    args.font_path = args.font_path.expanduser().resolve()
    if not args.font_path.is_file() or not os.access(args.font_path, os.R_OK):
        ap.error(f"font is not a readable file: {args.font_path}")
    if args.processes < 1 or args.num_samples < 0 or any(dpi <= 0 for dpi in args.dpis):
        ap.error("--processes and --dpis must be positive; --num-samples must be nonnegative")
    args.dpis = list(dict.fromkeys(args.dpis))
    args.tasks = list(dict.fromkeys(args.tasks))
    args.src_root = args.src_root.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    for task in args.tasks:
        source = args.src_root / task / "test.jsonl"
        if not source.is_file():
            ap.error(f"source manifest does not exist: {source}")
        if any(source == args.out_root / f"dpi_{dpi}" / task / "test.jsonl" for dpi in args.dpis):
            ap.error("--out-root must not overwrite the source manifests")

    # Delay optional imports so --help and path validation do not need render dependencies.
    try:
        for module in ("numpy", "PIL.Image", "pdf2image", "reportlab.platypus", "tqdm"):
            importlib.import_module(module)
    except ImportError as exc:
        ap.error(f"missing rendering dependency: {exc}")
    from tqdm import tqdm

    for dpi in args.dpis:
        for task in args.tasks:
            (args.out_root / f"dpi_{dpi}" / task).mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        src = args.src_root / task / "test.jsonl"
        rows = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
                if args.num_samples and len(rows) >= args.num_samples:
                    break

        out_files = {
            dpi: open(args.out_root / f"dpi_{dpi}" / task / "test.jsonl", "w", encoding="utf-8")
            for dpi in args.dpis
        }
        jobs = [(task, r, args.dpis, args.recover, args.out_root, args.font_path) for r in rows]
        n_ok = 0
        try:
            with Pool(processes=args.processes) as pool:
                for res in tqdm(
                    pool.imap_unordered(process_one, jobs, chunksize=1),
                    total=len(jobs),
                    desc=task,
                ):
                    if not res:
                        continue
                    n_ok += 1
                    for dpi, item in res.items():
                        out_files[dpi].write(json.dumps(item, ensure_ascii=False) + "\n")
        finally:
            for f in out_files.values():
                f.close()

        for dpi in args.dpis:
            path = args.out_root / f"dpi_{dpi}" / task / "test.jsonl"
            with open(path, encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
            items.sort(key=lambda x: x["index"])
            with open(path, "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

        print(f"{task}: {n_ok}/{len(rows)} rendered", flush=True)


if __name__ == "__main__":
    main()
