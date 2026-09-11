#!/usr/bin/env python3
"""Create a reproducible, proportionally stratified GRPO subset.

The source GRPO shard is already materialized as a Parquet file.  This tool
selects rows without replacement while keeping the distribution of
``source``, ``dpi`` and ``length_bucket`` proportional to the source shard.
The selected rows are shuffled before writing, so the resulting file can be
used directly as a training or validation shard.  With ``--balance-subsets``,
quotas are allocated by source, then subset, then DPI and length bucket.
No image files are copied: the existing image paths in each row are retained.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


STRATIFY_FIELDS = ("source", "dpi", "length_bucket")
DEFAULT_INPUT = "/vepfs-mlp2/c20250405/400042/data/VTC/GRPO/train.parquet"
DEFAULT_OUTPUT = "/vepfs-mlp2/c20250405/400042/data/VTC/GRPO/train_10k_uniform.parquet"
DEFAULT_SEED = 20260906


def _key_sort_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    # Fields have a stable order and type, but repr also handles a malformed
    # or null metadata value without making sorting fail.
    return tuple(repr(value) for value in key)


def _allocate_proportional_quotas(
    group_sizes: dict[tuple[Any, ...], int], sample_size: int
) -> dict[tuple[Any, ...], int]:
    """Allocate an exact sample size using largest remainders."""

    total = sum(group_sizes.values())
    floors: dict[tuple[Any, ...], int] = {}
    remainders: list[tuple[int, tuple[Any, ...]]] = []
    for key in sorted(group_sizes, key=_key_sort_key):
        numerator = group_sizes[key] * sample_size
        floors[key] = numerator // total
        remainders.append((numerator % total, key))

    remaining = sample_size - sum(floors.values())
    # Stable tie-breaking makes the output independent of dictionary order.
    remainders.sort(key=lambda item: (-item[0], _key_sort_key(item[1])))
    for _, key in remainders[:remaining]:
        floors[key] += 1
    return floors


def _distribution(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    counts = Counter(tuple(row.get(field) for field in fields) for row in rows)
    result = []
    for key, count in sorted(counts.items(), key=lambda item: _key_sort_key(item[0])):
        result.append({**dict(zip(fields, key)), "count": count})
    return result


def _allocate_subset_balanced_quotas(
    group_sizes: dict[tuple[Any, ...], int], sample_size: int
) -> dict[tuple[Any, ...], int]:
    """Allocate source → subset → (DPI, length) quotas using existing rounding."""

    source_sizes: Counter = Counter()
    subset_sizes: Counter = Counter()
    for key, count in group_sizes.items():
        source_sizes[key[:1]] += count
        subset_sizes[key[:2]] += count
    quotas: dict[tuple[Any, ...], int] = {}
    for source, source_quota in _allocate_proportional_quotas(source_sizes, sample_size).items():
        source_subsets = {key: count for key, count in subset_sizes.items() if key[:1] == source}
        for subset, subset_quota in _allocate_proportional_quotas(source_subsets, source_quota).items():
            subset_groups = {key: count for key, count in group_sizes.items() if key[:2] == subset}
            quotas.update(_allocate_proportional_quotas(subset_groups, subset_quota))
    return quotas


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sample_dataset(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    sample_size: int,
    seed: int,
    force: bool,
    balance_subsets: bool = False,
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --force to replace it"
        )
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}; pass --force to replace it"
        )
    if sample_size <= 0:
        raise ValueError("sample size must be positive")

    table = pq.read_table(input_path)
    row_count = table.num_rows
    if sample_size > row_count:
        raise ValueError(f"sample size {sample_size} exceeds input rows {row_count}")
    if "extra_info" not in table.column_names:
        raise ValueError("input Parquet has no extra_info column for stratification")

    stratify_fields = ("source", "subset", "dpi", "length_bucket") if balance_subsets else STRATIFY_FIELDS
    extra_info = table["extra_info"].to_pylist()
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row_index, metadata in enumerate(extra_info):
        if metadata is None:
            metadata = {}
        key = tuple(metadata.get(field) for field in stratify_fields)
        groups[key].append(row_index)

    allocate_quotas = _allocate_subset_balanced_quotas if balance_subsets else _allocate_proportional_quotas
    quotas = allocate_quotas(
        {key: len(indices) for key, indices in groups.items()}, sample_size
    )
    rng = random.Random(seed)
    selected_indices: list[int] = []
    for key in sorted(groups, key=_key_sort_key):
        indices = groups[key]
        rng.shuffle(indices)
        selected_indices.extend(indices[: quotas[key]])
    rng.shuffle(selected_indices)

    if len(selected_indices) != sample_size or len(set(selected_indices)) != sample_size:
        raise RuntimeError("internal sampling error: selected indices are not unique")

    selected_table = table.take(pa.array(selected_indices, type=pa.int64()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    # Snappy is supported by the existing training readers and keeps startup
    # reads fast; table metadata (including the Hugging Face feature schema) is
    # retained by Arrow's take/write path.
    pq.write_table(selected_table, temporary_output, compression="snappy")
    temporary_output.replace(output_path)

    selected_extra_info = selected_table["extra_info"].to_pylist()
    input_rows = [row or {} for row in extra_info]
    output_rows = [row or {} for row in selected_extra_info]
    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "sampling_method": (
            "hierarchical_proportional_stratified_without_replacement"
            if balance_subsets else "proportional_stratified_without_replacement"
        ),
        "stratify_fields": list(stratify_fields),
        "balance_subsets": balance_subsets,
        "allocation_hierarchy": (
            [["source"], ["subset"], ["dpi", "length_bucket"]]
            if balance_subsets else [list(stratify_fields)]
        ),
        "seed": seed,
        "input_rows": row_count,
        "output_rows": sample_size,
        "input_schema": [str(field) for field in table.schema],
        "output_schema": [str(field) for field in selected_table.schema],
        "selected_row_indices": selected_indices,
        "quotas": [
            {**dict(zip(stratify_fields, key)), "input_count": len(groups[key]), "output_count": quotas[key]}
            for key in sorted(groups, key=_key_sort_key)
        ],
        "input_distribution": _distribution(input_rows, stratify_fields),
        "output_distribution": _distribution(output_rows, stratify_fields),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(manifest_path, manifest)

    print(f"Wrote {sample_size:,} rows to {output_path}")
    print(f"Wrote sampling manifest to {manifest_path}")
    print(f"Seed: {seed}; strata: {len(groups)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--balance-subsets", action="store_true",
        help="allocate quotas by source, then subset, then DPI and length; useful for small validation sets",
    )
    parser.add_argument("--force", action="store_true", help="replace existing output and manifest")
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = args.output.with_name(f"{args.output.stem}.manifest.json")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    sample_dataset(
        input_path=arguments.input,
        output_path=arguments.output,
        manifest_path=arguments.manifest,
        sample_size=arguments.sample_size,
        seed=arguments.seed,
        force=arguments.force,
        balance_subsets=arguments.balance_subsets,
    )
