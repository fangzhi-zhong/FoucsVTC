#!/usr/bin/env python3
"""Render a UTF-8 document into matched page images at several resolutions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pypdfium2 as pdfium

try:
    from . import render_ruler_v1 as renderer
except ImportError:
    import render_ruler_v1 as renderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--font-path", type=Path,
                        default=os.environ.get("FOCUSVTC_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    parser.add_argument("--dpis", default="72,144")
    parser.add_argument("--font-size", type=float, default=9.0)
    parser.add_argument("--line-height", type=float, default=10.0)
    parser.add_argument("--page-size", default="595,842", help="width,height in PDF points")
    parser.add_argument("--margin-x", type=float, default=10.0)
    parser.add_argument("--margin-y", type=float, default=10.0)
    args = parser.parse_args()
    if args.font_path is None or not args.font_path.expanduser().is_file():
        parser.error("set --font-path or FOCUSVTC_FONT_PATH to a readable .ttf file")
    try:
        dpis = sorted(set(int(value) for value in args.dpis.split(",")))
        page_size = tuple(float(value) for value in args.page_size.split(","))
    except ValueError:
        parser.error("--dpis must be comma-separated integers; --page-size must be width,height")
    if not dpis or any(value <= 0 for value in dpis):
        parser.error("DPIs must be positive")
    if len(page_size) != 2 or min(page_size) <= 0:
        parser.error("--page-size must contain two positive values")
    text = args.text_file.read_text(encoding="utf-8")
    if not text.strip():
        parser.error("the input document is empty")
    output_root = args.out_root.expanduser().resolve()
    for dpi in dpis:
        if any((output_root / f"dpi_{dpi}").glob("page_*.png")):
            parser.error(f"page images already exist in {output_root / f'dpi_{dpi}'}; choose a fresh output directory")
    renderer.CFG.update(
        font_path=str(args.font_path.expanduser().resolve()),
        font_size=args.font_size,
        line_height=args.line_height,
        page_size=page_size,
        margin_x=args.margin_x,
        margin_y=args.margin_y,
        page_bg_color="#FFFFFF",
        font_color="#000000",
        alignment="LEFT",
        auto_crop_width=True,
        auto_crop_last_page=True,
    )
    # Typeset once: page identity and line breaks stay aligned across DPIs.
    pdf = pdfium.PdfDocument(renderer.build_pdf(text))
    manifest = {"source": str(args.text_file.resolve()), "views": {}}
    try:
        for dpi in dpis:
            image_paths, image_sizes = [], []
            output_dir = output_root / f"dpi_{dpi}"
            output_dir.mkdir(parents=True, exist_ok=True)
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                bitmap = page.render(scale=dpi / 72.0)
                image = bitmap.to_pil().copy()
                bitmap.close()
                page.close()
                cropped = renderer.crop_blank(image, dpi)
                path = output_dir / f"page_{page_index + 1:03d}.png"
                cropped.save(path)
                image_paths.append(str(path))
                image_sizes.append(list(cropped.size))
                cropped.close()
                image.close()
            manifest["views"][str(dpi)] = {
                "image": image_paths,
                "image_sizes": image_sizes,
                "num_pages": len(image_paths),
            }
    finally:
        pdf.close()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(manifest['views'][str(dpis[0])]['image'])} pages at {dpis} DPI: {manifest_path}")


if __name__ == "__main__":
    main()
