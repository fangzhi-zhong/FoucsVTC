#!/usr/bin/env python3
"""Build the 6.4K Qwen3.5-VL VTC GRPO shard for a 32K prompt budget.

The existing 50K builder intentionally stops at a 16K prompt estimate.  This
recipe scans the same raw SFT sources but keeps four length buckets up to 32K:
10% <=8K, 30% 8--16K, 35% 16--24K, and 25% 24--32K.  DPI is 50/30/20 for
72/96/144 DPI.  Source proportions remain 60/10/10/10/10 (Gemini,
LongBench, MRCR, RULER-v1, RULER-v2).

The output contains 6,400 training rows, 640 validation rows, and a manifest.
Images are referenced by the original paths; pixels are not copied.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import datasets


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_qwen35_vtc_grpo_50k as base  # noqa: E402


BUCKETS = ("le8k", "8k_16k", "16k_24k", "24k_32k")
# The <=8K bucket is only a small anchor because the base GRPO run already
# trained this range.  Sixty percent of this extension shard is above 16K.
BUCKET_RATIOS = {"le8k": 0.10, "8k_16k": 0.30, "16k_24k": 0.35, "24k_32k": 0.25}
DPI_RATIOS = {72: 0.50, 96: 0.30, 144: 0.20}
SOURCE_RATIOS = {
    "gemini-3.5-flash-30k": 0.60,
    "LongBench_SFT": 0.10,
    "MRCR_SFT": 0.10,
    "RULER_v1_SFT": 0.10,
    "RULER_v2_SFT": 0.10,
}
def _largest_remainder(total: int, ratios: dict[Any, float]) -> dict[Any, int]:
    """Convert ratios into deterministic integer quotas summing to ``total``."""

    raw = {key: total * value for key, value in ratios.items()}
    result = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda key: (raw[key] - result[key], str(key)), reverse=True)
    for key in order[:remainder]:
        result[key] += 1
    return result


def _bucket(length: int) -> str:
    if length <= 8_192:
        return "le8k"
    if length <= 16_384:
        return "8k_16k"
    if length <= 24_576:
        return "16k_24k"
    return "24k_32k"


def _scan_candidates(source_root: Path) -> tuple[dict[tuple[str, int, str], list[base.Ref]], dict[str, int]]:
    cells: dict[tuple[str, int, str], list[base.Ref]] = defaultdict(list)
    file_counts: dict[str, int] = {}
    seen: set[tuple[str, int, str]] = set()
    skipped = Counter()
    for spec in base.source_specs(source_root):
        for fixed_dpi, path in spec.files:
            if not path.is_file():
                raise FileNotFoundError(f"missing source file: {path}")
            count = 0
            with path.open(encoding="utf-8") as source_file:
                for line_no, line in enumerate(source_file):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    dpi = base._infer_dpi(row, fixed_dpi)
                    if dpi not in DPI_RATIOS:
                        skipped["invalid_dpi"] += 1
                        continue
                    high_view = row.get("images_144dpi") or row.get("image_dpi144")
                    if not isinstance(high_view, list) or not high_view:
                        skipped["missing_high_res"] += 1
                        continue
                    row_id = str(row.get("id", line_no))
                    identity = (spec.name, dpi, row_id)
                    if identity in seen:
                        skipped["duplicate"] += 1
                        continue
                    seen.add(identity)
                    prompt_length = base._estimate_prompt_length(row, dpi)
                    if prompt_length > 32_768:
                        skipped["over_32k"] += 1
                        continue
                    ref = base.Ref(
                        source=spec.name,
                        dpi=dpi,
                        subset=base._subset(row),
                        path=str(path),
                        line_no=line_no,
                        row_id=row_id,
                        prompt_length_estimate=prompt_length,
                        length_bucket=_bucket(prompt_length),
                    )
                    cells[(spec.name, dpi, ref.length_bucket)].append(ref)
                    count += 1
            file_counts[str(path)] = count
    print(f"candidate rows={sum(map(len, cells.values()))} skipped={dict(skipped)}")
    return cells, file_counts


def _cell_quotas(train_size: int, val_size: int) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], int]]:
    train_source = _largest_remainder(train_size, SOURCE_RATIOS)
    val_source = _largest_remainder(val_size, SOURCE_RATIOS)
    train: dict[tuple[str, int], int] = {}
    val: dict[tuple[str, int], int] = {}
    for source in SOURCE_RATIOS:
        train_dpi = _largest_remainder(train_source[source], DPI_RATIOS)
        val_dpi = _largest_remainder(val_source[source], DPI_RATIOS)
        for dpi in DPI_RATIOS:
            train[(source, dpi)] = train_dpi[dpi]
            val[(source, dpi)] = val_dpi[dpi]
    return train, val


def _allocate_bucket_counts(
    cells: dict[tuple[str, int, str], list[base.Ref]],
    cell_totals: dict[tuple[str, int], int],
    bucket_totals: dict[str, int],
) -> dict[tuple[str, int, str], int]:
    """Fill a cell x bucket table while respecting capacities and totals."""

    result = {(source, dpi, bucket): 0 for source, dpi in cell_totals for bucket in BUCKETS}
    remaining_cell = dict(cell_totals)
    remaining_bucket = dict(bucket_totals)
    capacities = {(source, dpi, bucket): len(refs) for (source, dpi, bucket), refs in cells.items()}
    while sum(remaining_cell.values()) > 0:
        feasible = [
            key
            for key in result
            if remaining_cell[key[:2]] > 0
            and remaining_bucket[key[2]] > 0
            and result[key] < capacities.get(key, 0)
        ]
        if not feasible:
            raise RuntimeError(
                "cannot satisfy 32K cell/bucket quotas; remaining cells="
                f"{remaining_cell}, remaining buckets={remaining_bucket}"
            )
        # Prioritize the most underfilled bucket, then the cell with the most
        # remaining rows. This keeps the requested global ratios exact whenever
        # the source pool has enough candidates.
        key = max(
            feasible,
            key=lambda item: (
                remaining_bucket[item[2]] / max(bucket_totals[item[2]], 1),
                remaining_cell[item[:2]] / max(cell_totals[item[:2]], 1),
                len(cells.get(item, ())),
                str(item),
            ),
        )
        result[key] += 1
        remaining_cell[key[:2]] -= 1
        remaining_bucket[key[2]] -= 1
    if any(remaining_bucket.values()):
        raise RuntimeError(f"bucket quotas not exhausted: {remaining_bucket}")
    return result


def _select(
    cells: dict[tuple[str, int, str], list[base.Ref]],
    train_cell: dict[tuple[str, int], int],
    val_cell: dict[tuple[str, int], int],
    seed: int,
) -> tuple[list[base.Ref], list[base.Ref], dict[str, Any]]:
    combined_cell = {key: train_cell[key] + val_cell[key] for key in train_cell}
    total_rows = sum(combined_cell.values())
    bucket_totals = _largest_remainder(total_rows, BUCKET_RATIOS)
    bucket_counts = _allocate_bucket_counts(cells, combined_cell, bucket_totals)
    train_refs: list[base.Ref] = []
    val_refs: list[base.Ref] = []
    selected_counts = Counter()
    for cell in sorted(combined_cell):
        source, dpi = cell
        for bucket in BUCKETS:
            key = (source, dpi, bucket)
            n = bucket_counts[key]
            refs = list(cells.get(key, ()))
            refs.sort(key=lambda ref: (ref.row_id, ref.path, ref.line_no))
            stable_key_seed = sum(ord(ch) for ch in ":".join(map(str, key)))
            random.Random(seed + stable_key_seed).shuffle(refs)
            chosen = refs[:n]
            if len(chosen) != n:
                raise RuntimeError(f"not enough candidates for {key}: need={n} have={len(refs)}")
            n_val = round(n * val_cell[cell] / combined_cell[cell]) if combined_cell[cell] else 0
            val_refs.extend(chosen[:n_val])
            train_refs.extend(chosen[n_val:])
            selected_counts[(source, dpi, bucket)] += n
    # Per-cell rounding can differ from the requested validation size by one
    # or two rows.  Correct that deterministically after the stratified split.
    requested_val = sum(val_cell.values())
    while len(val_refs) > requested_val:
        train_refs.append(val_refs.pop())
    while len(val_refs) < requested_val:
        val_refs.append(train_refs.pop())
    random.Random(seed + 991).shuffle(train_refs)
    random.Random(seed + 1991).shuffle(val_refs)
    manifest = {
        "source_ratios": SOURCE_RATIOS,
        "dpi_ratios": {str(key): value for key, value in DPI_RATIOS.items()},
        "length_ratios": BUCKET_RATIOS,
        "combined_length_quotas": bucket_totals,
        "cell_quotas_train": {f"{s}/dpi{d}": n for (s, d), n in train_cell.items()},
        "cell_quotas_validation": {f"{s}/dpi{d}": n for (s, d), n in val_cell.items()},
        "selected_source_dpi_length": {
            f"{s}/dpi{d}/{b}": n for (s, d, b), n in sorted(selected_counts.items())
        },
    }
    return train_refs, val_refs, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/vepfs-mlp2/c20250405/400042/data/VTC/SFT"))
    parser.add_argument("--output-root", type=Path, default=Path("/vepfs-mlp2/c20250405/400042/data/VTC/GRPO"))
    parser.add_argument("--train-size", type=int, default=6400)
    parser.add_argument("--val-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.train_size <= 0 or args.val_size <= 0:
        raise ValueError("train-size and val-size must be positive")

    cells, file_counts = _scan_candidates(args.source_root)
    train_cell, val_cell = _cell_quotas(args.train_size, args.val_size)
    train_refs, val_refs, selection_manifest = _select(cells, train_cell, val_cell, args.seed)
    if len(train_refs) != args.train_size or len(val_refs) != args.val_size:
        raise RuntimeError(f"selected train={len(train_refs)} val={len(val_refs)}")

    train_rows = base._load_selected(train_refs, verify_images=False)
    val_rows = base._load_selected(val_refs, verify_images=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_path = args.output_root / "train_6p4k_32k.parquet"
    val_path = args.output_root / "val_640_32k.parquet"
    manifest_path = args.output_root / "train_6p4k_32k.manifest.json"
    if not args.force and any(path.exists() for path in (train_path, val_path, manifest_path)):
        raise FileExistsError("output exists; pass --force to replace it")
    datasets.Dataset.from_list(train_rows).to_parquet(str(train_path))
    datasets.Dataset.from_list(val_rows).to_parquet(str(val_path))
    manifest = {
        "format": "verl-agent-parquet-v1",
        "recipe": "qwen35-vtc-grpo-6p4k-32k",
        "seed": args.seed,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "max_prompt_length_estimate": 32768,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "source_files_rows": file_counts,
        "train": base._counts(train_rows),
        "validation": base._counts(val_rows),
        "selection": selection_manifest,
        "notes": [
            "Prompt length is the conservative Qwen3.5 visual-token estimate used by the scanner.",
            "Parquet stores absolute source image paths and does not copy image pixels.",
            "Validation is sampled from the same source/DPI/length quotas with a separate row identity.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train": str(train_path), "validation": str(val_path), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
