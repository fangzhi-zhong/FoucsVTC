#!/usr/bin/env python3
"""Render every VTC-REL ``text_path`` at 144 DPI with source-correct fonts.

The output is resumable and isolated from the original 72-DPI images.  A row
is appended to ``images_dpi144/index.jsonl`` only after all its pages have been
written atomically and their geometry has been checked against the old pages.
When (and only when) all train rows are complete, ``train.jsonl`` is atomically
updated with ``image_dpi144`` and ``image_sizes_dpi144``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import traceback
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image
from tqdm import tqdm

import probe_vtc_rel_dpi144 as probe


ROOT = Path("/vepfs-mlp2/c20250405/400042")
DATASET = ROOT / "data/VTC_REL/gemini-3.5-flash-30k"
TRAIN = DATASET / "train.jsonl"
OUT = DATASET / "images_dpi144"
INDEX = OUT / "index.jsonl"
FAILED = OUT / "failed.jsonl"
DPI = 144

FONT_BY_KIND = {
    "VTC_SFT": probe.VERDANA,
    "RULER_v1_SFT": probe.VERDANA,
    "TRANSCRIBE_SFT": probe.DEJAVU,
    "VTC_GAP": probe.DEJAVU,
    "VTC_GAP_CODE": probe.DEJAVU_MONO,
}
FONT_HASH_BY_KIND = {
    kind: hashlib.sha256(font.read_bytes()).hexdigest()
    for kind, font in FONT_BY_KIND.items()
}


def output_paths(row: dict) -> list[Path]:
    text_path = Path(row["text_path"])
    relative = text_path.relative_to(DATASET / "text").with_suffix("")
    sample_dir = OUT / relative
    return [sample_dir / f"page_{index:03d}.png" for index in range(1, len(row["image"]) + 1)]


def make_task(row: dict) -> dict:
    return {
        "id": row["id"],
        "text_path": row["text_path"],
        "old_images": row["image"],
        "new_images": [str(path) for path in output_paths(row)],
        "kind": probe.source_kind(row["image"][0]),
        "transcribe_single_page": (
            "/TRANSCRIBE_SFT/images/page/" in row["image"][0]
            or "/TRANSCRIBE_SFT/images/box/" in row["image"][0]
        ),
    }


def poppler_path() -> str | None:
    directory = Path(sys.executable).parent
    return str(directory) if (directory / "pdftoppm").exists() else None


def atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    # PNG compression changes only the encoded bytes, never decoded pixels.
    # Level 1 is substantially faster for this 567k-page offline render.
    image.save(temp, format="PNG", compress_level=1)
    os.replace(temp, path)


def render_one(task: dict) -> dict:
    sid = task["id"]
    images: list[Image.Image] = []
    try:
        kind = task["kind"]
        font = FONT_BY_KIND[kind]
        probe.configure(font)
        text = Path(task["text_path"]).read_text(encoding="utf-8")
        pdf_bytes = probe.renderer.build_pdf(text)
        kwargs = {}
        pp = poppler_path()
        if pp:
            kwargs["poppler_path"] = pp
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pdf_pages = len(pdf.pages)
        expected_pages = len(task["old_images"])
        if task["transcribe_single_page"]:
            if pdf_pages < expected_pages:
                raise ValueError(f"PDF pages {pdf_pages} < retained pages {expected_pages}")
        elif pdf_pages != expected_pages:
            raise ValueError(f"PDF pages {pdf_pages} != old pages {expected_pages}")

        images = convert_from_bytes(
            pdf_bytes,
            dpi=DPI,
            first_page=1,
            last_page=expected_pages,
            **kwargs,
        )
        if len(images) != expected_pages:
            raise ValueError(f"raster pages {len(images)} != expected {expected_pages}")

        sizes = []
        geometry_deltas = []
        for index, image in enumerate(images):
            if kind == "VTC_SFT":
                cropped = probe.crop_vtc_sft(image, DPI, index == expected_pages - 1)
            else:
                cropped = probe.renderer.crop_blank(image, DPI)
            if cropped is not image:
                image.close()
                images[index] = cropped
            with Image.open(task["old_images"][index]) as old_image:
                old_size = old_image.size
            delta = [cropped.width - 2 * old_size[0], cropped.height - 2 * old_size[1]]
            if max(abs(value) for value in delta) > 2:
                raise ValueError(
                    f"page {index + 1}: 144-DPI size {cropped.size} drifts from "
                    f"2x old {old_size} by {delta}"
                )
            atomic_save_png(cropped, Path(task["new_images"][index]))
            sizes.append([cropped.width, cropped.height])
            geometry_deltas.append(delta)

        return {
            "id": sid,
            "dpi": DPI,
            "kind": kind,
            "font": str(font),
            "font_sha256": FONT_HASH_BY_KIND[kind],
            "text_path": task["text_path"],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "image": task["new_images"],
            "image_sizes": sizes,
            "num_pages": expected_pages,
            "pdf_pages": pdf_pages,
            "geometry_delta_from_2x": geometry_deltas,
        }
    except Exception as exc:  # noqa: BLE001 - return one sample failure to parent
        return {
            "id": sid,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    finally:
        for image in images:
            image.close()


def load_index() -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not INDEX.is_file():
        return records
    with INDEX.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["id"]
            if sid in records:
                raise ValueError(f"duplicate index id at line {line_no}: {sid}")
            if all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in row["image"]):
                records[sid] = row
    return records


def load_train() -> list[dict]:
    with TRAIN.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate train ids")
    return rows


def canonicalise_index(train_rows: list[dict], records: dict[str, dict]) -> None:
    temp = INDEX.with_name(INDEX.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(records[row["id"]], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, INDEX)


def update_train(train_rows: list[dict], records: dict[str, dict]) -> Path:
    temp = TRAIN.with_name(TRAIN.name + ".tmp_dpi144")
    with temp.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            record = records[row["id"]]
            row["image_dpi144"] = record["image"]
            row["image_sizes_dpi144"] = record["image_sizes"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    backup = TRAIN.with_name(TRAIN.name + ".bak_pre_dpi144")
    if not backup.exists():
        shutil.copy2(TRAIN, backup)
    os.replace(temp, TRAIN)
    return backup


def validate_complete(train_rows: list[dict], records: dict[str, dict]) -> dict:
    if set(records) != {row["id"] for row in train_rows}:
        raise ValueError("index/train id coverage mismatch")
    total_pages = 0
    fonts = Counter()
    max_delta = 0
    for row in train_rows:
        record = records[row["id"]]
        if len(record["image"]) != len(row["image"]):
            raise ValueError(f"{row['id']}: page count mismatch")
        for path in record["image"]:
            if not Path(path).is_file() or Path(path).stat().st_size == 0:
                raise ValueError(f"{row['id']}: missing {path}")
        total_pages += len(record["image"])
        fonts[Path(record["font"]).name] += 1
        max_delta = max(
            max_delta,
            max(abs(value) for delta in record["geometry_delta_from_2x"] for value in delta),
        )
    return {
        "documents": len(train_rows),
        "pages": total_pages,
        "fonts": dict(fonts),
        "max_geometry_delta_from_2x": max_delta,
    }


def validate_updated_train(expected_rows: int) -> None:
    count = 0
    with TRAIN.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            high = row.get("image_dpi144")
            sizes = row.get("image_sizes_dpi144")
            if not isinstance(high, list) or len(high) != len(row["image"]):
                raise ValueError(f"line {line_no}: invalid image_dpi144")
            if not isinstance(sizes, list) or len(sizes) != len(high):
                raise ValueError(f"line {line_no}: invalid image_sizes_dpi144")
            count += 1
    if count != expected_rows:
        raise ValueError(f"updated train rows {count} != {expected_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, default=48)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument(
        "--no-update-train",
        action="store_true",
        help="never update train.jsonl, even if this invocation completes all rows",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    train_rows = load_train()
    completed = load_index()
    wanted = set(args.ids) if args.ids else None
    todo_rows = [
        row for row in train_rows
        if row["id"] not in completed and (wanted is None or row["id"] in wanted)
    ]
    if wanted:
        missing = wanted - {row["id"] for row in train_rows}
        if missing:
            raise ValueError(f"unknown requested ids: {sorted(missing)}")
    if args.limit:
        todo_rows = todo_rows[: args.limit]
    tasks = [make_task(row) for row in todo_rows]
    print(
        f"train={len(train_rows)} complete={len(completed)} todo={len(tasks)} "
        f"processes={args.processes}",
        flush=True,
    )

    failures = 0
    context = get_context("fork")
    with INDEX.open("a", encoding="utf-8") as index_handle, \
            FAILED.open("a", encoding="utf-8") as failed_handle, \
            context.Pool(processes=args.processes) as pool:
        for number, result in enumerate(
            tqdm(
                pool.imap_unordered(render_one, tasks, chunksize=1),
                total=len(tasks),
                desc="dpi144",
                disable=not sys.stderr.isatty(),
            ),
            1,
        ):
            if "error" in result:
                failed_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                failures += 1
            else:
                index_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed[result["id"]] = result
            if number % 50 == 0:
                index_handle.flush()
                failed_handle.flush()
        index_handle.flush()
        failed_handle.flush()
        os.fsync(index_handle.fileno())
        os.fsync(failed_handle.fileno())

    print(f"invocation complete success={len(tasks) - failures} failed={failures}", flush=True)
    if failures:
        raise SystemExit(1)

    if len(completed) == len(train_rows):
        summary = validate_complete(train_rows, completed)
        canonicalise_index(train_rows, completed)
        backup = None
        if not args.no_update_train:
            backup = update_train(train_rows, completed)
            validate_updated_train(len(train_rows))
        summary["index"] = str(INDEX)
        summary["train_backup"] = str(backup) if backup else None
        print("REPORT " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    else:
        print(f"partial complete={len(completed)}/{len(train_rows)}", flush=True)


if __name__ == "__main__":
    main()
