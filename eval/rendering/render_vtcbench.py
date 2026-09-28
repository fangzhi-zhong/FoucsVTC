#!/usr/bin/env python3
"""Render VTCBench-Wild ``_context`` text with the standard VTC page layout.

The source JSON dumps are preserved.  The output mirrors their tier directories
and replaces only ``images`` with absolute PNG page paths.  dpi72/dpi144 use the
same PDF pagination and page numbering, so the tool gateway can map a low-res
page to its paired high-res page.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from . import render_longbench as r
except ImportError:
    import render_longbench as r

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FOCUSVTC_DATA_ROOT", ROOT / "datasets")).expanduser()
SOURCE = DATA_ROOT / "sources/vtcbench"
TIERS = ("data_json", "data_8k", "data_8k_16k", "data_16k_32k", "data_32k_64k")
SPLITS = ("Retrieval", "Reasoning", "Memory")


def init_renderer(fonts):
    r.FONTS = fonts


def render_sample(job):
    tier, split, index, sample, out_root, dpis, body_font = job
    text = str(sample.get("_context") or "")
    if not text:
        raise ValueError(f"{tier}/{split}#{index}: missing _context")
    font = r.pick_font(split, text, body_font)
    pdf = r.build_pdf(text, font)
    outputs = {}
    try:
        for dpi in dpis:
            pages = r.convert_from_bytes(pdf, dpi=dpi, poppler_path=r.POPPLER_PATH)
            paths, sizes = [], []
            out_dir = out_root / f"dpi_{dpi}" / tier / "images" / split / str(index)
            out_dir.mkdir(parents=True, exist_ok=True)
            for stale in out_dir.glob("page_*.png"):
                stale.unlink()
            try:
                for page_no, page in enumerate(pages, 1):
                    image = r.autocrop(page, page_no == len(pages))
                    try:
                        path = out_dir / f"page_{page_no:03d}.png"
                        image.save(path, "PNG")
                        paths.append(str(path))
                        sizes.append(list(image.size))
                    finally:
                        image.close()
            finally:
                for page in pages:
                    page.close()
            if not paths:
                raise ValueError(f"{tier}/{split}#{index}: no pages at dpi {dpi}")
            outputs[dpi] = paths, sizes
    finally:
        del pdf
    result = {}
    for dpi, (paths, sizes) in outputs.items():
        result[dpi] = {
            **sample,
            "images": paths,
            "num_pages": len(paths),
            "image_sizes": sizes,
            "render_dpi": dpi,
            "render_font": font,
            "render_layout": "A4_9pt",
            "vision_tokens_qwen32": r.count_vision_tokens(sizes),
        }
    if len(outputs[dpis[0]][0]) != len(outputs[dpis[-1]][0]):
        raise ValueError(f"{tier}/{split}#{index}: dpi page counts differ")
    return tier, split, index, result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--output", type=Path,
                    default=DATA_ROOT / "VTCBench-Wild_VTC")
    ap.add_argument("--font-path",
                    default=os.environ.get("FOCUSVTC_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    ap.add_argument("--cjk-font-path", default=os.environ.get("FOCUSVTC_CJK_FONT_PATH"))
    ap.add_argument("--dpis", type=int, nargs="+", default=[72, 144])
    ap.add_argument("--tiers", nargs="+", choices=TIERS, default=list(TIERS[1:]))
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    ap.add_argument("--processes", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if min(args.dpis) <= 0 or args.processes < 1 or args.limit < 0:
        ap.error("DPIs and --processes must be positive; --limit must be nonnegative")
    try:
        fonts = r.font_paths(args.font_path, args.cjk_font_path)
    except ValueError as exc:
        ap.error(str(exc))
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    if source == output:
        ap.error("output must be separate from source")
    jobs = []
    for tier in args.tiers:
        for split in args.splits:
            candidates = sorted((source / tier).glob(f"{split}-*.json"))
            if not candidates:
                raise FileNotFoundError(f"missing {tier}/{split} JSON")
            samples = read_json(candidates[0])
            if args.limit:
                samples = samples[:args.limit]
            for index, sample in enumerate(samples):
                jobs.append((tier, split, index, sample, output, args.dpis, "body"))
    if args.check_only:
        for tier, split, index, sample, *_ in jobs:
            for dpi in args.dpis:
                path = output / f"dpi_{dpi}" / tier / f"{split}-00000-of-00001.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
        print(f"checked {len(jobs)} samples")
        return
    grouped = {(tier, split): {} for tier in args.tiers for split in args.splits}
    with ProcessPoolExecutor(max_workers=args.processes, initializer=init_renderer,
                             initargs=(fonts,)) as pool:
        for done, (tier, split, index, result) in enumerate(pool.map(render_sample, jobs), 1):
            grouped[(tier, split)][index] = result
            if done % 100 == 0 or done == len(jobs):
                print(f"rendered {done}/{len(jobs)}", flush=True)
    for tier, split in grouped:
        for dpi in args.dpis:
            rows = [grouped[(tier, split)][i][dpi] for i in sorted(grouped[(tier, split)])]
            path = output / f"dpi_{dpi}" / tier / f"{split}-00000-of-00001.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
    manifest = {"source": str(source), "dpis": args.dpis, "tiers": args.tiers,
                "splits": args.splits, "layout": "A4_9pt", "fonts": fonts}
    (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
