#!/usr/bin/env python3
"""Render the `context` field of every LongBench sample to PNG pages.

Only `context` becomes an image. The surrounding instruction text and the
`input` question come from LongBench's official `dataset2prompt.json`, so the
rendered prompt is byte-identical to the text baseline except that `{context}`
is replaced by the `<image>` placeholders. That keeps the text run and the image
run comparable down to the wording.

Layout uses A4 pages, 10pt margins, a 9pt font, and 10pt leading.
The main font defaults to system DejaVu Sans; override it with --font-path
or FOCUSVTC_FONT_PATH. Optional --cjk-font-path and
--mono-font-path select fonts for CJK and code samples; otherwise the body
font is used and must cover the source characters. --out-suffix separates
font variants into different output directories.

Output layout:
    LongBench_VTC/dpi_{dpi}{suffix}/{dataset}/test.jsonl
    LongBench_VTC/dpi_{dpi}{suffix}/images/{dataset}/{index}/page_001.png
"""

import argparse
import gc
import io
import json
import math
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FOCUSVTC_DATA_ROOT", ROOT / "datasets")).expanduser()
FONTS = {}

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

SRC_ROOT = DATA_ROOT / "sources/longbench"
DST_ROOT = DATA_ROOT / "LongBench_VTC"
PROMPTS_PATH = ROOT / "eval/LongBench/config/dataset2prompt.json"
PROMPTS = {}
CODE_DATASETS = {"lcc", "repobench-p"}

# poppler ships inside the conda env rather than on the system PATH
_env_bin = Path(sys.executable).parent
POPPLER_PATH = str(_env_bin) if (_env_bin / "pdftoppm").exists() else None

_RE_MULTISPACE = re.compile(r" {2,}")
_RE_CJK = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")


def font_paths(font_path, cjk_font_path=None, mono_font_path=None):
    if not font_path:
        raise ValueError("set --font-path or FOCUSVTC_FONT_PATH to a readable .ttf file")
    fonts = {
        "body": font_path,
        "sourcehan": cjk_font_path or font_path,
        "mono": mono_font_path or font_path,
    }
    for key, value in fonts.items():
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"font does not exist: {path}")
        fonts[key] = str(path)
    return fonts


def init_worker(out_root, fonts, prompts):
    global DST_ROOT, FONTS, PROMPTS
    DST_ROOT, FONTS, PROMPTS = Path(out_root), fonts, prompts


def pick_font(dataset, text, body_font):
    if _RE_CJK.search(text):
        return "sourcehan"
    if dataset in CODE_DATASETS:
        return "mono"
    return body_font


def build_pdf(text, font_key):
    font_name = f"vtc_{font_key}"
    pdfmetrics.registerFont(TTFont(font_name, FONTS[font_key]))
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
        fontName=font_name,
        fontSize=FONT_SIZE,
        leading=LINE_HEIGHT,
        textColor=colors.HexColor(FONT_COLOR),
        backColor=colors.HexColor(PAGE_BG),
        alignment=TA_LEFT,
        # CJK has no spaces to break on, so let reportlab break mid-run
        wordWrap="CJK" if font_key == "sourcehan" else None,
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


def split_prompt(dataset, row):
    """Return (header, tail) around the `{context}` slot of the official prompt."""
    template = PROMPTS[dataset]
    if "{context}" not in template:
        raise ValueError(f"{dataset}: template has no {{context}} slot")
    head_tmpl, tail_tmpl = template.split("{context}", 1)
    head = head_tmpl.replace("{input}", row["input"])
    tail = tail_tmpl.replace("{input}", row["input"])
    return head, tail


def _make_row(dataset, row, index, header, tail, dpi, font_key, paths, sizes):
    return {
        "index": index,
        "_id": row["_id"],
        "dataset": dataset,
        "dpi": dpi,
        "font": font_key,
        "question": header + "<image>" * len(paths) + tail,
        "answers": row["answers"],
        "all_classes": row["all_classes"],
        "language": row["language"],
        "length": row["length"],
        "context_chars": len(row["context"]),
        "images": paths,
        "num_pages": len(paths),
        "image_sizes": [list(s) for s in sizes],
        "vision_tokens": count_vision_tokens(sizes),
    }


def _rows_from_existing(dataset, row, index, header, tail, dpis, font_key, done):
    results = {}
    for dpi in dpis:
        paths = [str(p) for p in done[dpi]]
        sizes = []
        for p in paths:
            with Image.open(p) as im:
                sizes.append(im.size)
        results[dpi] = _make_row(dataset, row, index, header, tail, dpi, font_key, paths, sizes)
    return results


def process_one(job):
    dataset, index, row, dpis, recover, body_font, suffix = job
    try:
        header, tail = split_prompt(dataset, row)
        font_key = pick_font(dataset, row["context"], body_font)

        expected = {
            dpi: DST_ROOT / f"dpi_{dpi}{suffix}" / "images" / dataset / str(index)
            for dpi in dpis
        }
        if recover:
            done = {
                dpi: (sorted(d.glob("page_*.png")) if d.is_dir() else [])
                for dpi, d in expected.items()
            }
            if all(done[dpi] for dpi in dpis) and len({len(v) for v in done.values()}) == 1:
                return _rows_from_existing(dataset, row, index, header, tail, dpis, font_key, done)

        pdf = build_pdf(row["context"], font_key)
        results = {}
        for dpi in dpis:
            out_dir = expected[dpi]
            out_dir.mkdir(parents=True, exist_ok=True)
            for stale in out_dir.glob("page_*.png"):
                stale.unlink()
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
            results[dpi] = _make_row(dataset, row, index, header, tail, dpi, font_key, paths, sizes)
        del pdf
        gc.collect()
        return results
    except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the sweep
        print(f"[ERROR] dataset={dataset} index={index}: {exc}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-root", type=Path, default=SRC_ROOT)
    ap.add_argument("--out-root", type=Path, default=DST_ROOT)
    ap.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    ap.add_argument("--font-path",
                    default=os.environ.get("FOCUSVTC_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    ap.add_argument("--cjk-font-path", default=os.environ.get("FOCUSVTC_CJK_FONT_PATH"))
    ap.add_argument("--mono-font-path", default=os.environ.get("FOCUSVTC_MONO_FONT_PATH"))
    ap.add_argument("--dpis", type=int, nargs="+", default=[72, 144])
    ap.add_argument("--num-samples", type=int, default=0, help="per dataset, 0 = all")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--processes", type=int, default=48)
    ap.add_argument("--recover", action="store_true")
    ap.add_argument(
        "--out-suffix",
        default="",
        help="append to the dpi_* directory name for separate font variants",
    )
    args = ap.parse_args()
    if min(args.dpis) <= 0 or args.processes < 1 or args.num_samples < 0:
        ap.error("DPIs and --processes must be positive; --num-samples must be nonnegative")
    try:
        fonts = font_paths(args.font_path, args.cjk_font_path, args.mono_font_path)
    except ValueError as exc:
        ap.error(str(exc))
    prompts = json.loads(args.prompts.expanduser().read_text(encoding="utf-8"))
    args.datasets = args.datasets or list(prompts)
    unknown = set(args.datasets) - prompts.keys()
    if unknown:
        ap.error(f"unknown datasets: {', '.join(sorted(unknown))}")
    src_root = args.src_root.expanduser().resolve()
    init_worker(args.out_root.expanduser().resolve(), fonts, prompts)

    for dpi in args.dpis:
        for dataset in args.datasets:
            (DST_ROOT / f"dpi_{dpi}{args.out_suffix}" / dataset).mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        rows = []
        with open(src_root / f"{dataset}.jsonl", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
                if args.num_samples and len(rows) >= args.num_samples:
                    break

        out_files = {
            dpi: open(
                DST_ROOT / f"dpi_{dpi}{args.out_suffix}" / dataset / "test.jsonl",
                "w",
                encoding="utf-8",
            )
            for dpi in args.dpis
        }
        jobs = [
            (dataset, i, r, args.dpis, args.recover, "body", args.out_suffix)
            for i, r in enumerate(rows)
        ]
        n_ok = 0
        try:
            with Pool(processes=args.processes, initializer=init_worker,
                      initargs=(DST_ROOT, fonts, prompts)) as pool:
                for res in tqdm(
                    pool.imap_unordered(process_one, jobs, chunksize=1),
                    total=len(jobs),
                    desc=dataset,
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
            path = DST_ROOT / f"dpi_{dpi}{args.out_suffix}" / dataset / "test.jsonl"
            items = [json.loads(l) for l in open(path, encoding="utf-8")]
            items.sort(key=lambda x: x["index"])
            with open(path, "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

        print(f"{dataset}: {n_ok}/{len(rows)} rendered", flush=True)


if __name__ == "__main__":
    main()
