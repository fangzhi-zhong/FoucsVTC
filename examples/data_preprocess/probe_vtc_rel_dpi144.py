#!/usr/bin/env python3
"""Probe VTC-REL layout/font fidelity at 72 DPI before rendering at 144 DPI.

For each selected sample the script:

1. rebuilds the resolution-independent PDF from ``text_path``;
2. rasterises it at 72 DPI and compares every retained page with the old PNG;
3. rasterises the same PDF at 144 DPI;
4. checks that ink bounds and cropped page geometry scale by two;
5. tries both Verdana and SourceHan for VTC_SFT so the old font is established
   from pixels rather than assumed from the current command-line default.

Nothing in train.jsonl or the original image directories is modified.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image


ROOT = Path("/vepfs-mlp2/c20250405/400042")
DATASET = ROOT / "data/VTC_REL/gemini-3.5-flash-30k"
RENDER_DIR = ROOT / "data/RULER_v1_VTC"
sys.path.insert(0, str(RENDER_DIR))
import render_ruler_v1 as renderer  # noqa: E402


VERDANA = ROOT / "data/VTC_SFT/word2png/config/Verdana.ttf"
SOURCE_HAN = ROOT / "data/VTC_SFT/word2png/config/SourceHanSansHWSC-VF.ttf"
DEJAVU = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
DEJAVU_MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")

DEFAULT_IDS = [
    "ChatQA-Training-Data_tatqa_10955",
    "ChatQA-Training-Data_ropes_4480",
    "long_sft_QA_train_66437",
    "NarrativeQA_131072_QA_train_17148",
    "train-00011-of-00026_3765",
    "HotpotQA_5ae01d8e554299025d62a403_long",
    "tr_page_long_sft_02450",
    "tr_box_long_sft_06202",
    "tr_needle_ruler_fwe_4096_1430",
    "ruler_niah_single_1_4096_0247",
    "long_hotpotqa_00003",
    "cnt_fwe_00152",
    "nid_single_00976",
    "code_page_csharp_00384",
    "code_complete_python_01229",
]


def source_kind(image: str) -> str:
    rel = image.split("/data/", 1)[1]
    if rel.startswith("VTC_SFT/"):
        return "VTC_SFT"
    if rel.startswith("RULER_v1_SFT/"):
        return "RULER_v1_SFT"
    if rel.startswith("TRANSCRIBE_SFT/"):
        return "TRANSCRIBE_SFT"
    if rel.startswith("VTC_GAP/code/"):
        return "VTC_GAP_CODE"
    if rel.startswith("VTC_GAP/"):
        return "VTC_GAP"
    raise ValueError(f"unknown image source: {image}")


def configure(font: Path) -> None:
    renderer.CFG.update(
        out_root="",
        dpis=[72, 144],
        page_size=(595.0, 842.0),
        margin_x=10.0,
        margin_y=10.0,
        font_path=str(font),
        font_size=9.0,
        line_height=10.0,
        page_bg_color="#FFFFFF",
        font_color="#000000",
        alignment="LEFT",
        auto_crop_width=True,
        auto_crop_last_page=True,
    )


def crop_vtc_sft(image: Image.Image, dpi: int, is_last: bool) -> Image.Image:
    """Mirror VTC_SFT at 72 DPI; scale its pixel margin at higher DPI."""
    gray = np.asarray(image.convert("L"))
    background = np.median(gray[:2, :2])
    mask = np.abs(gray.astype(np.int16) - int(background)) > 5
    scale = dpi / 72.0
    right = image.width
    lower = image.height
    columns = np.where(mask.any(axis=0))[0]
    if columns.size:
        right = min(image.width, int(columns[-1] + 1 + 10.0 * scale))
    if is_last:
        rows = np.where(mask.any(axis=1))[0]
        if rows.size:
            # The legacy VTC_SFT renderer used last_row + margin_y here
            # (unlike its width branch, which includes +1).
            lower = min(image.height, int(rows[-1] + 10.0 * scale))
    if (right, lower) == image.size:
        return image
    return image.crop((0, 0, right, lower))


def rasterise(pdf_bytes: bytes, dpi: int, kind: str) -> list[Image.Image]:
    kwargs = {"dpi": dpi}
    poppler = Path(sys.executable).parent
    if (poppler / "pdftoppm").exists():
        kwargs["poppler_path"] = str(poppler)
    pages = convert_from_bytes(pdf_bytes, **kwargs)
    out = []
    for index, image in enumerate(pages):
        if kind == "VTC_SFT":
            cropped = crop_vtc_sft(image, dpi, index == len(pages) - 1)
        else:
            cropped = renderer.crop_blank(image, dpi)
        if cropped is not image:
            image.close()
        out.append(cropped)
    return out


def image_diff(left: Image.Image, right: Image.Image) -> dict:
    result = {
        "old_size": list(left.size),
        "new_size": list(right.size),
        "same_size": left.size == right.size,
    }
    if left.size != right.size:
        result.update({"changed_pixels": None, "mae": None, "max_abs": None})
        return result
    a = np.asarray(left.convert("RGB"), dtype=np.int16)
    b = np.asarray(right.convert("RGB"), dtype=np.int16)
    delta = np.abs(a - b)
    result.update(
        changed_pixels=int(np.any(delta, axis=2).sum()),
        mae=float(delta.mean()),
        max_abs=int(delta.max()),
    )
    return result


def ink_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    gray = np.asarray(image.convert("L"), dtype=np.int16)
    background = int(np.median(gray[:2, :2]))
    ys, xs = np.where(np.abs(gray - background) > 5)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def geometry_2x(low: Image.Image, high: Image.Image) -> dict:
    low_box = ink_bbox(low)
    high_box = ink_bbox(high)
    result = {
        "dpi72_size": list(low.size),
        "dpi144_size": list(high.size),
        "size_delta_from_2x": [high.width - 2 * low.width, high.height - 2 * low.height],
        "dpi72_ink_bbox": list(low_box) if low_box else None,
        "dpi144_ink_bbox": list(high_box) if high_box else None,
    }
    if low_box and high_box:
        result["ink_bbox_delta_from_2x"] = [
            high_box[index] - 2 * low_box[index] for index in range(4)
        ]
    else:
        result["ink_bbox_delta_from_2x"] = None
    return result


def candidate_fonts(kind: str) -> list[Path]:
    if kind == "VTC_SFT":
        return [VERDANA, SOURCE_HAN]
    if kind == "RULER_v1_SFT":
        return [VERDANA]
    if kind == "VTC_GAP_CODE":
        return [DEJAVU_MONO]
    return [DEJAVU]


def compare_font(row: dict, kind: str, font: Path) -> tuple[dict, bytes, list[Image.Image]]:
    configure(font)
    text = Path(row["text_path"]).read_text(encoding="utf-8")
    pdf_bytes = renderer.build_pdf(text)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pdf_page_count = len(pdf.pages)
        word_counts = [len(page.extract_words(use_text_flow=True)) for page in pdf.pages]
    rendered = rasterise(pdf_bytes, 72, kind)
    originals = [Image.open(path).convert("RGB") for path in row["image"]]
    compared = min(len(originals), len(rendered))
    page_diffs = [image_diff(originals[index], rendered[index]) for index in range(compared)]
    for image in originals:
        image.close()
    exact_pages = sum(
        diff["same_size"] and diff["changed_pixels"] == 0 for diff in page_diffs
    )
    size_error = sum(
        abs(diff["old_size"][0] - diff["new_size"][0])
        + abs(diff["old_size"][1] - diff["new_size"][1])
        for diff in page_diffs
    )
    mae = sum(
        diff["mae"] if diff["mae"] is not None else 255.0
        for diff in page_diffs
    )
    metric = {
        "font": str(font),
        "pdf_pages": pdf_page_count,
        "old_pages": len(originals),
        "compared_pages": compared,
        "word_counts": word_counts,
        "exact_pages": exact_pages,
        "size_error": size_error,
        "mae_sum": mae,
        "page_diffs": page_diffs,
    }
    return metric, pdf_bytes, rendered


def metric_score(metric: dict) -> tuple:
    return (
        metric["compared_pages"] - metric["exact_pages"],
        metric["size_error"],
        metric["mae_sum"],
    )


def load_rows(ids: list[str]) -> list[dict]:
    wanted = set(ids)
    rows = {}
    with (DATASET / "train.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in wanted:
                rows[row["id"]] = row
                if len(rows) == len(wanted):
                    break
    missing = wanted - set(rows)
    if missing:
        raise ValueError(f"missing ids: {sorted(missing)}")
    return [rows[sid] for sid in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASET / "dpi144_probe",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {
        "python": sys.executable,
        "samples": [],
    }
    for row in load_rows(args.ids):
        sid = row["id"]
        kind = source_kind(row["image"][0])
        font_trials = []
        products = []
        for font in candidate_fonts(kind):
            metric, pdf_bytes, rendered72 = compare_font(row, kind, font)
            font_trials.append(metric)
            products.append((metric, pdf_bytes, rendered72))
        metric, pdf_bytes, rendered72 = min(products, key=lambda item: metric_score(item[0]))
        configure(Path(metric["font"]))
        rendered144 = rasterise(pdf_bytes, 144, kind)

        sample_dir = args.out / sid
        (sample_dir / "dpi72").mkdir(parents=True, exist_ok=True)
        (sample_dir / "dpi144").mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(rendered72[: len(row["image"])], 1):
            image.save(sample_dir / "dpi72" / f"page_{index:03d}.png")
        for index, image in enumerate(rendered144[: len(row["image"])], 1):
            image.save(sample_dir / "dpi144" / f"page_{index:03d}.png")
        geometry = [
            geometry_2x(rendered72[index], rendered144[index])
            for index in range(min(len(row["image"]), len(rendered72), len(rendered144)))
        ]
        sample_report = {
            "id": sid,
            "kind": kind,
            "text_path": row["text_path"],
            "old_images": row["image"],
            "selected_font": metric["font"],
            "font_trials": font_trials,
            "geometry": geometry,
            "probe_dir": str(sample_dir),
        }
        report["samples"].append(sample_report)
        print(
            sid,
            "kind", kind,
            "font", Path(metric["font"]).name,
            "exact72", f"{metric['exact_pages']}/{metric['compared_pages']}",
            "page_counts", f"old={metric['old_pages']} pdf={metric['pdf_pages']}",
            "max_size_delta144", max(
                max(abs(value) for value in page["size_delta_from_2x"])
                for page in geometry
            ),
            flush=True,
        )
        for _, _, images in products:
            for image in images:
                image.close()
        for image in rendered144:
            image.close()

    report_path = args.out / "probe_report.json"
    temp = report_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, report_path)
    print("report", report_path, flush=True)


if __name__ == "__main__":
    main()
