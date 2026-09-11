#!/usr/bin/env python3
"""Build the 50K Qwen3.5-VL VTC GRPO dataset.

The source SFT rows are JSONL records whose conversations live in separate
JSON files.  This builder keeps the image files on disk and writes verl's
agent-ready Parquet schema.  Sampling is deterministic and stratified by
source, DPI, and the source's sub-dataset/task.

The default split is:

* train: 50,000 rows (Gemini 30K + 5K from each of four auxiliary sources)
* validation: 5,000 rows (the same source/DPI proportions as train)
* DPI ratio in every source: 3:1:1 (72:96:144), except Gemini's requested
  20K/8K/2K split.

Run from any directory, for example::

    python build_qwen35_vtc_grpo_50k.py --output-root \
      /vepfs-mlp2/c20250405/400042/data/VTC/GRPO

The script does not copy image pixels.  Parquet stores the existing absolute
image paths, so the training nodes must see the same shared filesystem.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import datasets


DEFAULT_ROOT = Path("/vepfs-mlp2/c20250405/400042/data/VTC/SFT")
DEFAULT_OUTPUT = Path("/vepfs-mlp2/c20250405/400042/data/VTC/GRPO")
ENV_NAME = "qwen3_vl_vtc_zoom"
DATA_SOURCE = "qwen35_vtc"

# The actual tool schema is injected by verl/RLHFDataset from tools_schema_path.
VTC_SYSTEM_PROMPT = "You are a helpful assistant."
VTC_TURN_PROMPT = (
    "\nThink first. During thinking, point out the relevant page or evidence location when needed. "
    "Then call zoom_region if needed, and answer.\n\n"
    "Format strictly as: <think>...</think> <tool_call>...</tool_call> "
    "(if tools are needed) <answer>...</answer>."
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    files: tuple[tuple[int | None, Path], ...]
    train_quota: dict[int, int]
    val_quota: dict[int, int]


@dataclass(frozen=True)
class Ref:
    source: str
    dpi: int
    subset: str
    path: str
    line_no: int
    row_id: str
    prompt_length_estimate: int = 0
    length_bucket: str = "unknown"


def source_specs(root: Path) -> tuple[SourceSpec, ...]:
    """Return the five requested sources and their exact quotas."""

    gemini = root / "gemini-3.5-flash-30k"
    longbench = root / "LongBench_SFT"
    mrcr = root / "MRCR_SFT"
    ruler1 = root / "RULER_v1_SFT"
    ruler2 = root / "RULER_v2_SFT"
    return (
        SourceSpec(
            "gemini-3.5-flash-30k",
            ((72, gemini / "train_72dpi.json"), (96, gemini / "train_96dpi.json"), (144, gemini / "train_144dpi.json")),
            {72: 20_000, 96: 8_000, 144: 2_000},
            {72: 2_000, 96: 800, 144: 200},
        ),
        SourceSpec(
            "LongBench_SFT",
            ((72, longbench / "train_72dpi.json"), (96, longbench / "train_96dpi.json"), (144, longbench / "train_144dpi.json")),
            {72: 3_000, 96: 1_000, 144: 1_000},
            {72: 300, 96: 100, 144: 100},
        ),
        SourceSpec(
            "MRCR_SFT",
            ((None, mrcr / "train.json"),),
            {72: 3_000, 96: 1_000, 144: 1_000},
            {72: 300, 96: 100, 144: 100},
        ),
        SourceSpec(
            "RULER_v1_SFT",
            ((72, ruler1 / "train.json"), (96, ruler1 / "train_96dpi.json"), (144, ruler1 / "train_144dpi.json")),
            {72: 3_000, 96: 1_000, 144: 1_000},
            {72: 300, 96: 100, 144: 100},
        ),
        SourceSpec(
            "RULER_v2_SFT",
            ((72, ruler2 / "train.json"), (96, ruler2 / "train_96dpi.json"), (144, ruler2 / "train_144dpi.json")),
            {72: 3_000, 96: 1_000, 144: 1_000},
            {72: 300, 96: 100, 144: 100},
        ),
    )


def _parse_literal(value: Any) -> Any:
    """Decode metadata that is occasionally stored as a Python repr string."""

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{(":
        return value
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value


def _infer_dpi(row: dict[str, Any], fixed_dpi: int | None) -> int | None:
    # Some LongBench *_96dpi rows retain metadata.dpi=72, so a known file view
    # takes precedence over metadata.
    if fixed_dpi in {72, 96, 144}:
        return fixed_dpi
    metadata = row.get("metadata") or {}
    dpi = metadata.get("dpi")
    if str(dpi).isdigit() and int(dpi) in {72, 96, 144}:
        return int(dpi)
    paths = row.get("image") or row.get("images") or []
    text = " ".join(str(p) for p in paths[:2])
    for candidate in (144, 96, 72):
        if f"{candidate}dpi" in text:
            return candidate
    return None


def _subset(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("task"):
        return str(metadata["task"])
    # Gemini has no task metadata.  The first two image path components retain
    # the original sub-training-set identity (e.g. multihop/hotpotqa_long).
    paths = row.get("image") or row.get("images") or []
    if paths:
        path = str(paths[0])
        match = re.search(r"/images(?:_[^/]+)?/(.+)$", path)
        if match:
            parts = match.group(1).split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])
            if parts:
                return parts[0]
    conversation = str(row.get("conversations", ""))
    return Path(conversation).parent.name or "unknown"


def _low_image_values(row: dict[str, Any], dpi: int) -> list[str]:
    """Get the selected-DPI pages without requiring a full row conversion."""

    for name in (f"images_{dpi}dpi", f"image_dpi{dpi}"):
        values = row.get(name)
        if isinstance(values, list) and values:
            return [str(path) for path in values]
    values = row.get("image") or row.get("images")
    if isinstance(values, list) and values:
        return [str(path) for path in values]
    return []


def _estimate_prompt_length(row: dict[str, Any], dpi: int) -> int:
    """Estimate text + visual tokens without decoding image pixels.

    The tokenizer's chat template counts ``<image>`` as one placeholder and
    therefore cannot enforce a multimodal context limit by itself.  The source
    shards use a small number of fixed page layouts, so this estimate uses a
    conservative per-page Qwen3.5 visual-token budget by DPI.  It intentionally
    overestimates the low-resolution layouts; selected rows carry the estimate
    in ``extra_info`` for later auditing.
    """

    pages = _low_image_values(row, dpi)
    if not pages:
        return 16_385
    # Typical rendered page budgets for this Qwen3.5 processor are roughly
    # 0.5K/1K/2K merged visual tokens at 72/96/144 DPI.  The 10% margin and
    # 1K text/tool allowance keep the filter conservative without an NFS stat
    # for every page of every candidate row.
    # 144-DPI MRCR has a sharp page-count boundary; 1.8K/page keeps enough
    # <=16K rows with the safety margin while still excluding its 8+ page
    # high-resolution records.
    visual_per_page = {72: 600, 96: 1_100, 144: 1_800}[dpi]
    return int(math.ceil(1_024 + 1.10 * visual_per_page * len(pages)))


def _scan_refs(specs: Iterable[SourceSpec]) -> tuple[dict[tuple[str, int], dict[str, list[Ref]]], dict[str, int]]:
    grouped: dict[tuple[str, int], dict[str, list[Ref]]] = defaultdict(lambda: defaultdict(list))
    file_counts: dict[str, int] = {}
    seen_ids: set[tuple[str, int, str]] = set()
    for spec in specs:
        for fixed_dpi, path in spec.files:
            if not path.is_file():
                raise FileNotFoundError(f"missing source file: {path}")
            count = 0
            with path.open(encoding="utf-8") as source_file:
                for line_no, line in enumerate(source_file):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    dpi = _infer_dpi(row, fixed_dpi)
                    if dpi not in {72, 96, 144}:
                        continue
                    # A zoom trajectory needs a 144-DPI target.  A small
                    # portion of the RULER-v1 legacy rows only has the low-DPI
                    # image; leave those rows out of the candidate pool rather
                    # than silently making zoom a no-op.
                    high_view = row.get("images_144dpi") or row.get("image_dpi144")
                    if not (isinstance(high_view, list) and high_view):
                        continue
                    row_id = str(row.get("id", line_no))
                    identity = (spec.name, dpi, row_id)
                    if identity in seen_ids:
                        # A few legacy RULER shards contain the same ID more
                        # than once.  Deduplicating before train/val sampling
                        # prevents an identical row leaking across splits.
                        continue
                    seen_ids.add(identity)
                    prompt_length = _estimate_prompt_length(row, dpi)
                    if prompt_length > 16_384:
                        continue
                    bucket = "le8k" if prompt_length <= 8_192 else "8k_16k"
                    ref = Ref(
                        spec.name,
                        dpi,
                        _subset(row),
                        str(path),
                        line_no,
                        row_id,
                        prompt_length,
                        bucket,
                    )
                    grouped[(spec.name, dpi)][ref.subset].append(ref)
                    count += 1
            file_counts[str(path)] = count
    return grouped, file_counts


def _allocate(total: int, capacities: dict[str, int]) -> dict[str, int]:
    """Allocate as evenly as possible while respecting per-group capacity."""

    if total < 0 or total > sum(capacities.values()):
        raise ValueError(f"cannot allocate {total} rows from capacities totaling {sum(capacities.values())}")
    allocation = {key: 0 for key in sorted(capacities)}
    # Round-robin gives every sub-train an equal chance before any group gets
    # a second extra row; capped groups are skipped once exhausted.
    while sum(allocation.values()) < total:
        eligible = [key for key in allocation if allocation[key] < capacities[key]]
        if not eligible:
            raise RuntimeError("allocation exhausted unexpectedly")
        key = min(eligible, key=lambda item: (allocation[item], item))
        allocation[key] += 1
    return allocation


def _choose_length_balanced(refs: list[Ref], total: int, seed: int) -> list[Ref]:
    """Choose ``total`` rows with an approximately 60/40 length mix."""

    buckets: dict[str, list[Ref]] = {"le8k": [], "8k_16k": []}
    for ref in refs:
        buckets.setdefault(ref.length_bucket, []).append(ref)
    for bucket in buckets:
        random.Random(seed + sum(ord(ch) for ch in bucket)).shuffle(buckets[bucket])
    target_low = round(total * 0.60)
    n_low = min(target_low, len(buckets["le8k"]))
    n_high = min(total - n_low, len(buckets["8k_16k"]))
    # Fill a deficit from the other bucket when a source/DPI has an
    # imbalanced natural length distribution.
    if n_low + n_high < total:
        n_low = min(total - n_high, len(buckets["le8k"]))
    if n_low + n_high < total:
        n_high = min(total - n_low, len(buckets["8k_16k"]))
    chosen = buckets["le8k"][:n_low] + buckets["8k_16k"][:n_high]
    if len(chosen) != total:
        raise RuntimeError(f"cannot select {total} rows with <=16K estimate; available={len(refs)}")
    random.Random(seed + 31337).shuffle(chosen)
    return chosen


def _rebalance_global_length(
    grouped: dict[tuple[str, int], dict[str, list[Ref]]],
    train_refs: list[Ref],
    val_refs: list[Ref],
    specs: Iterable[SourceSpec],
    high_ratio: float = 0.40,
) -> int:
    """Use spare candidates to bring each source/DPI near a 60/40 mix.

    Subset quotas are already fixed at this point.  Replacing a selected short
    row by an unused long row from the same subset keeps those quotas unchanged;
    if a subset has no long candidates, another subset supplies the replacement.
    """

    spec_by_name = {spec.name: spec for spec in specs}
    replacements = 0
    for key, by_subset in sorted(grouped.items()):
        source, dpi = key
        spec = spec_by_name[source]
        target = round((spec.train_quota[dpi] + spec.val_quota[dpi]) * high_ratio)
        selected = [ref for ref in train_refs + val_refs if ref.source == source and ref.dpi == dpi]
        current = sum(ref.length_bucket == "8k_16k" for ref in selected)
        deficit = max(0, target - current)
        if deficit == 0:
            continue
        selected_keys = {(ref.path, ref.line_no) for ref in selected}
        for subset in sorted(by_subset):
            if deficit == 0:
                break
            high_pool = [
                ref
                for ref in by_subset[subset]
                if ref.length_bucket == "8k_16k" and (ref.path, ref.line_no) not in selected_keys
            ]
            if not high_pool:
                continue
            # Find low rows in the same subset in both splits, preserving the
            # split assignment when replacing them.
            low_locations: list[tuple[list[Ref], int]] = []
            for collection in (train_refs, val_refs):
                low_locations.extend(
                    (collection, i)
                    for i, ref in enumerate(collection)
                    if ref.source == source and ref.dpi == dpi and ref.subset == subset and ref.length_bucket == "le8k"
                )
            n = min(deficit, len(high_pool), len(low_locations))
            for high, (collection, index) in zip(high_pool[:n], low_locations[:n]):
                old = collection[index]
                collection[index] = high
                selected_keys.discard((old.path, old.line_no))
                selected_keys.add((high.path, high.line_no))
            deficit -= n
            replacements += n
    return replacements


def _select_refs(
    grouped: dict[tuple[str, int], dict[str, list[Ref]]],
    specs: Iterable[SourceSpec],
    seed: int,
) -> tuple[list[Ref], list[Ref], dict[str, Any]]:
    train_refs: list[Ref] = []
    val_refs: list[Ref] = []
    manifest: dict[str, Any] = {"seed": seed, "strata": {}}
    spec_by_name = {spec.name: spec for spec in specs}
    for (source, dpi), by_subset in sorted(grouped.items()):
        spec = spec_by_name[source]
        train_total = spec.train_quota[dpi]
        val_total = spec.val_quota[dpi]
        capacities = {subset: len(refs) for subset, refs in by_subset.items()}
        combined = _allocate(train_total + val_total, capacities)
        val_counts = _allocate(val_total, combined)
        stratum_manifest = {
            "available": sum(capacities.values()),
            "subsets_available": dict(sorted(capacities.items())),
            "train_requested": train_total,
            "val_requested": val_total,
            "train_by_subset": {},
            "val_by_subset": {},
        }
        for subset in sorted(by_subset):
            refs = list(by_subset[subset])
            # Sorting before shuffling makes the sample independent of JSONL
            # concatenation order while remaining reproducible.
            refs.sort(key=lambda ref: (ref.row_id, ref.path, ref.line_no))
            group_seed = seed + sum(ord(ch) for ch in f"{source}:{dpi}:{subset}")
            selected = _choose_length_balanced(refs, combined[subset], group_seed)
            n_val = val_counts[subset]
            selected_low = [ref for ref in selected if ref.length_bucket == "le8k"]
            selected_high = [ref for ref in selected if ref.length_bucket == "8k_16k"]
            val_low_target = round(n_val * 0.60)
            val_low = min(val_low_target, len(selected_low))
            val_high = min(n_val - val_low, len(selected_high))
            if val_low + val_high < n_val:
                val_low = min(n_val - val_high, len(selected_low))
            if val_low + val_high < n_val:
                val_high = min(n_val - val_low, len(selected_high))
            random.Random(group_seed + 707).shuffle(selected_low)
            random.Random(group_seed + 909).shuffle(selected_high)
            val_refs.extend(selected_low[:val_low] + selected_high[:val_high])
            val_set = {(ref.path, ref.line_no) for ref in val_low and selected_low[:val_low] or []}
            val_set.update((ref.path, ref.line_no) for ref in selected_high[:val_high])
            train_refs.extend(ref for ref in selected if (ref.path, ref.line_no) not in val_set)
            stratum_manifest["train_by_subset"][subset] = combined[subset] - n_val
            stratum_manifest["val_by_subset"][subset] = n_val
            stratum_manifest.setdefault("length_by_subset", {})[subset] = {
                "train_le8k": sum(ref.length_bucket == "le8k" for ref in train_refs if ref.subset == subset and ref.source == source and ref.dpi == dpi),
                "train_8k_16k": sum(ref.length_bucket == "8k_16k" for ref in train_refs if ref.subset == subset and ref.source == source and ref.dpi == dpi),
                "val_le8k": val_low,
                "val_8k_16k": val_high,
            }
        manifest["strata"][f"{source}/dpi{dpi}"] = stratum_manifest
    replacements = _rebalance_global_length(grouped, train_refs, val_refs, specs)
    manifest["length_replacements_to_40_percent_high"] = replacements
    random.Random(seed + 991).shuffle(train_refs)
    random.Random(seed + 1991).shuffle(val_refs)
    return train_refs, val_refs, manifest


def _conversation_path(row: dict[str, Any], source_path: Path) -> Path:
    value = row.get("conversations")
    if not value:
        raise ValueError(f"row {row.get('id')} has no conversations path")
    path = Path(str(value))
    return path if path.is_absolute() else source_path.parent / path


def _load_conversation(row: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    with _conversation_path(row, source_path).open(encoding="utf-8") as conversation_file:
        conversation = json.load(conversation_file)
    if isinstance(conversation, dict):
        conversation = conversation.get("messages", conversation.get("conversations", conversation))
    if not isinstance(conversation, list):
        raise ValueError(f"row {row.get('id')} conversation is not a list")
    return conversation


_MAGIC_NUMBER_STEM = re.compile(
    r"\s+The special magic number(?:s)? for [^\n?]+ mentioned in the provided text "
    r"(?:is|are)\s*$",
    flags=re.IGNORECASE,
)


def _messages(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not conversation:
        raise ValueError("empty conversation")
    final_assistant = max(
        (i for i, item in enumerate(conversation) if item.get("from") in {"gpt", "assistant"} or item.get("role") == "assistant"),
        default=-1,
    )
    messages: list[dict[str, Any]] = []
    for i, item in enumerate(conversation):
        if i == final_assistant:
            continue
        source_role = item.get("from", item.get("role"))
        role = {"human": "user", "gpt": "assistant", "assistant": "assistant", "user": "user"}.get(source_role, source_role)
        value = item.get("value", item.get("content", ""))
        if role == "user" and isinstance(value, str):
            value = _MAGIC_NUMBER_STEM.sub("", value).rstrip()
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"unsupported conversation role: {item!r}")
        messages.append({"role": role, "content": value})
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("prompt must contain a user message")
    system_indices = [i for i, message in enumerate(messages) if message["role"] == "system"]
    if system_indices:
        first = system_indices[0]
        messages[first] = {**messages[first], "content": f"{VTC_SYSTEM_PROMPT}\n\n{messages[first]['content']}"}
    else:
        messages.insert(0, {"role": "system", "content": VTC_SYSTEM_PROMPT})
    last_user = max(i for i, message in enumerate(messages) if message["role"] == "user")
    messages[last_user] = {**messages[last_user], "content": f"{messages[last_user]['content']}{VTC_TURN_PROMPT}"}
    return messages


def _gold_from_conversation(conversation: list[dict[str, Any]]) -> list[str]:
    assistants = [item for item in conversation if item.get("from") in {"gpt", "assistant"} or item.get("role") == "assistant"]
    if not assistants:
        return []
    text = str(assistants[-1].get("value", assistants[-1].get("content", ""))).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if answer_match:
        text = answer_match.group(1).strip()
    text = re.sub(r"^<answer>\s*|\s*</answer>\s*$", "", text, flags=re.IGNORECASE).strip()
    return [text] if text else []


def _image_list(row: dict[str, Any], dpi: int, field: str = "low") -> list[str]:
    if field == "low":
        names = (f"images_{dpi}dpi", f"image_dpi{dpi}")
    else:
        names = ("images_144dpi", "image_dpi144")
    for name in names:
        values = row.get(name)
        if isinstance(values, list) and values:
            return [str(path) for path in values]
    # MRCR/RULER combined rows use image/images as the selected DPI view.
    values = row.get("image") or row.get("images")
    if isinstance(values, list) and values:
        if field == "low":
            return [str(path) for path in values]
    raise ValueError(f"row {row.get('id')} has no {field} image list for dpi {dpi}")


def _evidence_pairs(layout: Any) -> list[dict[str, Any]]:
    layout = _parse_literal(layout)
    pairs: list[dict[str, Any]] = []
    if isinstance(layout, dict):
        if "page" in layout or "page_id" in layout:
            page = layout.get("page", layout.get("page_id"))
            box = layout.get("bbox_2d", layout.get("bbox"))
            if isinstance(box, (list, tuple)) and len(box) == 4:
                pairs.append({"page": int(page), "bbox": [float(x) for x in box]})
        else:
            for page, boxes in layout.items():
                pairs.extend(_evidence_pairs({"page": page, "bbox": boxes}))
    elif isinstance(layout, (list, tuple)):
        for item in layout:
            item = _parse_literal(item)
            if isinstance(item, dict):
                pairs.extend(_evidence_pairs(item))
            elif isinstance(item, (list, tuple)) and len(item) == 5:
                pairs.append({"page": int(item[0]), "bbox": [float(x) for x in item[1:]]})
    return pairs


def _normalised_evidence(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = _evidence_pairs(metadata.get("evidence_locations", []))
    if evidence:
        return evidence
    by_dpi = metadata.get("evidence_bboxes_by_dpi", {})
    if isinstance(by_dpi, dict):
        for key in sorted(by_dpi, key=lambda value: int("".join(ch for ch in str(value) if ch.isdigit()) or 0), reverse=True):
            evidence = _evidence_pairs(by_dpi[key])
            if evidence:
                return evidence
    return []


def _normalise_gold(value: Any) -> list[str]:
    value = _parse_literal(value)
    if isinstance(value, dict):
        value = value.get("ground_truth", value.get("answer", []))
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if value is not None and str(value).strip() else []


def _materialise(
    ref: Ref,
    row: dict[str, Any],
    source_path: Path,
    index: int,
    verify_images: bool = False,
) -> dict[str, Any]:
    conversation = _load_conversation(row, source_path)
    metadata = dict(row.get("metadata") or {})
    gold = _normalise_gold(metadata.get("gold", row.get("gold")))
    if not gold:
        gold = _gold_from_conversation(conversation)
    if not gold:
        raise ValueError(f"row {row.get('id')} has no usable gold answer")
    low_images = _image_list(row, ref.dpi, "low")
    high_images = _image_list(row, ref.dpi, "high")
    if len(low_images) != len(high_images):
        raise ValueError(f"row {row.get('id')} low/high page count differs")
    # An exhaustive stat of every page is very slow on the shared filesystem
    # (some samples contain dozens of pages).  The source audit already checks
    # the image layout; optional verification checks one page from each view.
    if verify_images:
        for path in (low_images[0], high_images[0]):
            if not Path(path).is_file():
                raise FileNotFoundError(f"row {row.get('id')} references missing image: {path}")
    evidence = _normalised_evidence(metadata)
    extra_info = {
        "index": index,
        "id": str(row.get("id", ref.row_id)),
        "source": ref.source,
        "subset": ref.subset,
        "dpi": ref.dpi,
        "prompt_length_estimate": ref.prompt_length_estimate,
        "length_bucket": ref.length_bucket,
        "gold": gold,
        "num_pages": int(row.get("num_pages", len(low_images))),
        "evidence_locations": evidence,
        "task": str(metadata.get("task", ref.subset)),
        "ruler_version": str(metadata.get("ruler_version", "")),
        "max_tool_calls": 3,
    }
    return {
        "data_source": DATA_SOURCE,
        "env_name": ENV_NAME,
        "enable_tools": True,
        "prompt": _messages(conversation),
        "images": low_images,
        "high_res_images": high_images,
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": extra_info,
    }


def _load_selected(refs: list[Ref], verify_images: bool = False) -> list[dict[str, Any]]:
    by_file: dict[str, dict[int, Ref]] = defaultdict(dict)
    for ref in refs:
        by_file[ref.path][ref.line_no] = ref
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for path_string, needed in by_file.items():
        path = Path(path_string)
        with path.open(encoding="utf-8") as source_file:
            for line_no, line in enumerate(source_file):
                ref = needed.get(line_no)
                if ref is None or not line.strip():
                    continue
                row = json.loads(line)
                rows_by_key[(path_string, line_no)] = _materialise(
                    ref, row, path, len(rows_by_key), verify_images=verify_images
                )
    missing = [ref for ref in refs if (ref.path, ref.line_no) not in rows_by_key]
    if missing:
        raise RuntimeError(f"failed to materialise {len(missing)} selected rows; first={missing[0]}")
    return [rows_by_key[(ref.path, ref.line_no)] for ref in refs]


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = Counter(row["extra_info"]["source"] for row in rows)
    dpi = Counter(row["extra_info"]["dpi"] for row in rows)
    source_dpi = Counter((row["extra_info"]["source"], row["extra_info"]["dpi"]) for row in rows)
    subset = Counter((row["extra_info"]["source"], row["extra_info"]["subset"]) for row in rows)
    length_bucket = Counter(row["extra_info"]["length_bucket"] for row in rows)
    lengths = [int(row["extra_info"]["prompt_length_estimate"]) for row in rows]
    return {
        "total": len(rows),
        "source": dict(sorted(source.items())),
        "dpi": {str(k): v for k, v in sorted(dpi.items())},
        "source_dpi": {f"{source}/dpi{dpi}": n for (source, dpi), n in sorted(source_dpi.items())},
        "source_subset": {f"{source}/{subset}": n for (source, subset), n in sorted(subset.items())},
        "length_bucket": dict(sorted(length_bucket.items())),
        "prompt_length_estimate": {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--force", action="store_true", help="overwrite existing output Parquet/manifest")
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="stat one low/high page per selected row (slower on NFS)",
    )
    args = parser.parse_args()

    specs = source_specs(args.source_root)
    grouped, file_counts = _scan_refs(specs)
    train_refs, val_refs, sampling_manifest = _select_refs(grouped, specs, args.seed)
    if len(train_refs) != 50_000 or len(val_refs) != 5_000:
        raise RuntimeError(f"unexpected selected sizes: train={len(train_refs)} val={len(val_refs)}")
    train_rows = _load_selected(train_refs, verify_images=args.verify_images)
    val_rows = _load_selected(val_refs, verify_images=args.verify_images)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "train.parquet"
    val_path = output_root / "val.parquet"
    manifest_path = output_root / "manifest.json"
    if not args.force and any(path.exists() for path in (train_path, val_path, manifest_path)):
        raise FileExistsError("output already exists; pass --force to replace it")
    datasets.Dataset.from_list(train_rows).to_parquet(str(train_path))
    datasets.Dataset.from_list(val_rows).to_parquet(str(val_path))
    manifest = {
        "format": "verl-agent-parquet-v1",
        "seed": args.seed,
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "source_files_rows": file_counts,
        "train": _counts(train_rows),
        "validation": _counts(val_rows),
        "sampling": sampling_manifest,
        "notes": [
            "Validation is a deterministic 5K (10%) holdout sampled separately inside each source/DPI stratum.",
            "Subsets/tasks are allocated as evenly as possible; capacity limits are respected.",
            "Images are referenced by absolute path and are not copied into Parquet.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train": str(train_path), "validation": str(val_path), "manifest": str(manifest_path), "train_rows": len(train_rows), "val_rows": len(val_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
