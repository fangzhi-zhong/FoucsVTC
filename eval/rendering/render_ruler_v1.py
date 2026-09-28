#!/usr/bin/env python3
"""Render the RULER v1 8k split to images at several DPIs.

The PDF layout (page size, font size, margins, all in points) is identical for
every DPI, so page count and line breaks are byte-identical across the sweep and
the only variable is raster resolution. Each sample is typeset once and then
rasterised at every DPI.

Output layout:
    RULER_v1_VTC/
      manifest.json
      dpi_{N}/
        index.jsonl
        images/{task}/{id}/page_{i:03d}.png
"""

import argparse
import gc
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from pdf2image import convert_from_bytes
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate
from tqdm import tqdm

ALIGN_MAP = {"LEFT": TA_LEFT, "CENTER": TA_CENTER, "RIGHT": TA_RIGHT, "JUSTIFY": TA_JUSTIFY}

# Set once in the parent, inherited by forked workers.
CFG = {}


def parse_args():
    p = argparse.ArgumentParser(description="Render RULER v1 contexts to PNG at multiple DPIs")
    p.add_argument("--src", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--dpis", default="48,60,72,84,96,120,144")
    p.add_argument("--per-task", type=int, default=100, help="samples per task, 0 = all")
    p.add_argument("--processes", type=int, default=64)

    # Page layout in PDF points.
    p.add_argument("--page-size", default="595,842", help="width,height in pt")
    p.add_argument("--margin-x", type=float, default=10)
    p.add_argument("--margin-y", type=float, default=10)
    p.add_argument("--font-path",
                   default=os.environ.get("FOCUSVTC_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    p.add_argument("--font-size", type=float, default=9)
    p.add_argument("--line-height", type=float, default=10)
    p.add_argument("--page-bg-color", default="#FFFFFF")
    p.add_argument("--font-color", default="#000000")
    p.add_argument("--alignment", choices=list(ALIGN_MAP), default="LEFT")
    p.add_argument("--auto-crop-width", action="store_true", default=True)
    p.add_argument("--auto-crop-last-page", action="store_true", default=True)
    p.add_argument("--overwrite", action="store_true", help="ignore existing index.jsonl and redo everything")
    args = p.parse_args()
    if not args.font_path:
        p.error("set --font-path or FOCUSVTC_FONT_PATH to a readable .ttf file")
    return args


def build_pdf(text):
    """Typeset one context into a PDF. Resolution-independent."""
    font_name = str(Path(CFG["font_path"]).resolve())
    pdfmetrics.registerFont(TTFont(font_name, CFG["font_path"]))

    page_size = CFG["page_size"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=page_size,
        leftMargin=CFG["margin_x"],
        rightMargin=CFG["margin_x"],
        topMargin=CFG["margin_y"],
        bottomMargin=CFG["margin_y"],
    )

    bg = colors.HexColor(CFG["page_bg_color"])

    def draw_bg(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(bg)
        canvas.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1)
        canvas.restoreState()

    kwargs = dict(
        name="Custom",
        parent=getSampleStyleSheet()["Normal"],
        fontName=font_name,
        fontSize=CFG["font_size"],
        leading=CFG["line_height"],
        textColor=colors.HexColor(CFG["font_color"]),
        backColor=bg,
        alignment=ALIGN_MAP[CFG["alignment"]],
    )
    if re.search(r"[\u4E00-\u9FFF]", text):
        kwargs["wordWrap"] = "CJK"
    style = ParagraphStyle(**kwargs)

    text = text.replace("\xad", "").replace("\u200b", "")
    processed = re.sub(r" {2,}", lambda m: "&nbsp;" * len(m.group()), escape(text))
    processed = processed.replace("\n", "<br/>").replace("\t", "&nbsp;" * 4)

    doc.build([Paragraph(processed, style)], onFirstPage=draw_bg, onLaterPages=draw_bg)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


def crop_blank(img, dpi):
    """Trim trailing blank columns and rows, keeping a DPI-scaled margin so the
    crop is proportionally identical at every resolution."""
    gray = np.array(img.convert("L"))
    bg_gray = np.median(gray[:2, :2])
    mask = np.abs(gray - bg_gray) > 5

    right, lower = img.width, img.height
    if CFG["auto_crop_width"]:
        cols = np.where(mask.any(axis=0))[0]
        if cols.size:
            right = min(img.width, int(cols[-1] + 1 + CFG["margin_x"] * dpi / 72.0))
    if CFG["auto_crop_last_page"]:
        rows = np.where(mask.any(axis=1))[0]
        if rows.size:
            lower = min(img.height, int(rows[-1] + 1 + CFG["margin_y"] * dpi / 72.0))

    if (right, lower) == (img.width, img.height):
        return img
    return img.crop((0, 0, right, lower))


def process_one(item):
    """Render one sample at every DPI. Returns {dpi: record}."""
    _id = item["id"]
    task = item["task"]
    try:
        pdf_bytes = build_pdf(item["context"])
        out = {}
        for dpi in CFG["dpis"]:
            out_dir = Path(CFG["out_root"]) / f"dpi_{dpi}" / "images" / task / _id
            out_dir.mkdir(parents=True, exist_ok=True)

            pages = convert_from_bytes(pdf_bytes, dpi=dpi)
            image_paths, sizes = [], []
            for i, img in enumerate(pages, start=1):
                img = crop_blank(img, dpi)
                path = out_dir / f"page_{i:03d}.png"
                img.save(path, "PNG")
                image_paths.append(str(path))
                sizes.append([img.width, img.height])
                img.close()
            pages.clear()

            px = sum(w * h for w, h in sizes)
            out[dpi] = {
                "id": _id,
                "task": task,
                "dpi": dpi,
                "image": image_paths,
                "image_sizes": sizes,
                "num_pages": len(sizes),
                "question": item["question"],
                "answer_prefix": item["answer_prefix"],
                "answer": item["answer"],
                "max_new_tokens": item["max_new_tokens"],
                "context_chars": len(item["context"]),
                # 28x28 px per token for GLM-4.1V/Glyph, 32x32 for Qwen3-VL
                "vision_tokens_glyph": px // 784,
                "vision_tokens_qwen3vl": px // 1024,
            }
        del pdf_bytes
        gc.collect()
        return out
    except Exception as e:
        print(f"[ERROR] id={_id} task={task}: {e}", file=sys.stderr)
        return None


def load_samples(src, per_task):
    per_task_items = defaultdict(list)
    with open(src, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            task = d["task"]
            if per_task and len(per_task_items[task]) >= per_task:
                continue
            d["id"] = f"{task}_{len(per_task_items[task]):04d}"
            per_task_items[task].append(d)
    return [it for task in sorted(per_task_items) for it in per_task_items[task]]


def main():
    args = parse_args()
    w, h = map(float, args.page_size.split(","))
    dpis = [int(x) for x in args.dpis.split(",")]

    CFG.update(
        out_root=str(Path(args.out_root).expanduser().resolve()),
        dpis=dpis,
        page_size=(w, h),
        margin_x=args.margin_x,
        margin_y=args.margin_y,
        font_path=args.font_path,
        font_size=args.font_size,
        line_height=args.line_height,
        page_bg_color=args.page_bg_color,
        font_color=args.font_color,
        alignment=args.alignment,
        auto_crop_width=args.auto_crop_width,
        auto_crop_last_page=args.auto_crop_last_page,
    )
    if not os.path.exists(args.font_path):
        raise FileNotFoundError(f"font not found: {args.font_path}")

    samples = load_samples(args.src, args.per_task)
    print(f"loaded {len(samples)} samples from {args.src}")

    out_root = Path(args.out_root)
    index_paths = {d: out_root / f"dpi_{d}" / "index.jsonl" for d in dpis}
    for d, p in index_paths.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite and p.exists():
            p.unlink()

    # Resume: a sample is done only if it is present in every DPI index.
    done = None
    for d, p in index_paths.items():
        ids = set()
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        ids.add(json.loads(line)["id"])
                    except json.JSONDecodeError:
                        continue
        done = ids if done is None else (done & ids)
    done = done or set()
    todo = [s for s in samples if s["id"] not in done]
    print(f"already complete: {len(done)}, to render: {len(todo)} x {len(dpis)} dpis")

    if todo:
        handles = {d: open(p, "a", encoding="utf-8") for d, p in index_paths.items()}
        try:
            with Pool(processes=args.processes) as pool:
                it = pool.imap_unordered(process_one, todo, chunksize=1)
                for n, res in enumerate(tqdm(it, total=len(todo), desc="rendering"), start=1):
                    if not res:
                        continue
                    for d, rec in res.items():
                        handles[d].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if n % 50 == 0:
                        for fh in handles.values():
                            fh.flush()
        finally:
            for fh in handles.values():
                fh.close()

    manifest = {
        "source": args.src,
        "split": "8k",
        "benchmark": "RULER v1",
        "per_task": args.per_task,
        "num_samples": len(samples),
        "dpis": dpis,
        "layout": {
            "page_size_pt": [w, h],
            "margin_x_pt": args.margin_x,
            "margin_y_pt": args.margin_y,
            "font": Path(args.font_path).name,
            "font_size_pt": args.font_size,
            "line_height_pt": args.line_height,
            "alignment": args.alignment,
            "auto_crop_width": args.auto_crop_width,
            "auto_crop_last_page": args.auto_crop_last_page,
        },
        "note": "Layout is fixed in points, so page count and line breaks are identical across DPIs; only raster resolution varies.",
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nper-dpi summary:")
    print(f"{'dpi':>5} {'samples':>8} {'pages':>7} {'tok_glyph':>10} {'tok_qwen3vl':>12}")
    for d in dpis:
        recs = [json.loads(l) for l in open(index_paths[d], encoding="utf-8")]
        if not recs:
            continue
        n = len(recs)
        print(f"{d:>5} {n:>8} {sum(r['num_pages'] for r in recs)/n:>7.2f} "
              f"{sum(r['vision_tokens_glyph'] for r in recs)/n:>10.0f} "
              f"{sum(r['vision_tokens_qwen3vl'] for r in recs)/n:>12.0f}")


if __name__ == "__main__":
    main()
