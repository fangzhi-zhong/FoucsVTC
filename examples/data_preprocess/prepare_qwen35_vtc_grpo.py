#!/usr/bin/env python3
"""Convert the VTC/RULER JSONL files into verl GRPO Parquet shards.

The source annotations keep the conversation in a separate JSON file and
contain both low-resolution and 144-DPI page paths.  verl's RLHFDataset wants
OpenAI-style ``prompt`` messages plus explicit ``images`` and reward metadata,
so this script performs that small, deterministic conversion without loading
the image pixels.

Example (first 32 samples, useful for the first training iteration)::

    python prepare_qwen35_vtc_grpo.py \
      --input /path/to/RULER_v2_SFT/train.json \
      --output /path/to/vtc_grpo_train.parquet \
      --limit 32

Use ``--limit 0`` for the complete source file.  A separate validation shard
can be produced with ``--skip`` and ``--limit``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import datasets


ENV_NAME = "qwen3_vl_vtc_zoom"
DATA_SOURCE = "ruler_vtc"
VTC_SYSTEM_PROMPT = "You are a helpful assistant."
VTC_TURN_PROMPT = (
    "\nThink first. During thinking, point out the relevant page or evidence location when needed. "
    "Then call zoom_region if needed, and answer.\n\n"
    "Format strictly as: <think>...</think> <tool_call>...</tool_call> "
    "(if tools are needed) <answer>...</answer>."
)


_MAGIC_NUMBER_STEM = re.compile(
    r"\s+The special magic number(?:s)? for [^\n?]+ mentioned in the provided text "
    r"(?:is|are)\s*$",
    flags=re.IGNORECASE,
)


def _clean_user_text(value: Any) -> Any:
    """Remove the RULER answer-completion stem from the generation prompt.

    The stem is useful for supervised SFT but leaks a direct-answer format into
    this agent rollout.  Only the exact magic-number suffix is removed; the
    underlying question and page placeholders are left untouched.
    """

    if not isinstance(value, str):
        return value
    return _MAGIC_NUMBER_STEM.sub("", value).rstrip()


def _as_messages(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ``from/value`` records and omit the supervised final answer."""

    if not conversation:
        raise ValueError("empty conversation")
    messages = []
    # Keep context turns, but never leak the final reference answer into the
    # generation prompt.  RULER rows end in a single gpt answer; for a
    # multi-turn source, only the last gpt record is removed.
    final_gpt = max((i for i, item in enumerate(conversation) if item.get("from") in {"gpt", "assistant"}), default=-1)
    for i, item in enumerate(conversation):
        if i == final_gpt:
            continue
        role = {"human": "user", "gpt": "assistant", "assistant": "assistant", "user": "user"}.get(
            item.get("from"), item.get("role")
        )
        value = item.get("value", item.get("content", ""))
        if role == "user":
            value = _clean_user_text(value)
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"unsupported conversation role: {item!r}")
        messages.append({"role": role, "content": value})
    if not messages or not any(message["role"] == "user" for message in messages):
        raise ValueError("prompt must contain a user message")
    # The source SFT conversations usually contain only a user turn.  Make the
    # tool/retry policy explicit so the model has a stable behavior target;
    # the actual schema is still injected by RLHFDataset at chat-template time.
    system_indices = [i for i, message in enumerate(messages) if message["role"] == "system"]
    if system_indices:
        first = system_indices[0]
        messages[first] = {
            **messages[first],
            "content": f"{VTC_SYSTEM_PROMPT}\n\n{messages[first]['content']}",
        }
    else:
        messages.insert(0, {"role": "system", "content": VTC_SYSTEM_PROMPT})

    # DeepEyes puts the action/answer format in the current user turn.  Keep
    # the same separation here; the system message carries policy, while the
    # turn prompt is also reused after every tool observation.
    user_indices = [i for i, message in enumerate(messages) if message["role"] == "user"]
    last_user = user_indices[-1]
    messages[last_user] = {
        **messages[last_user],
        "content": f"{messages[last_user]['content']}{VTC_TURN_PROMPT}",
    }
    return messages


def _first_existing(row: dict[str, Any], names: tuple[str, ...]) -> list[str]:
    for name in names:
        value = row.get(name)
        if isinstance(value, list) and value:
            return [str(path) for path in value]
    raise KeyError(f"none of the image columns exists: {names}")


def convert_row(row: dict[str, Any], source_path: Path, index: int) -> dict[str, Any]:
    conversation_path = Path(row["conversations"])
    if not conversation_path.is_absolute():
        conversation_path = source_path.parent / conversation_path
    with conversation_path.open(encoding="utf-8") as conversation_file:
        conversation = json.load(conversation_file)

    metadata = dict(row.get("metadata") or {})
    gold = metadata.get("gold", row.get("gold", []))
    if isinstance(gold, str):
        gold = [gold]
    gold = [str(answer).strip() for answer in (gold or [])]
    if not gold:
        raise ValueError(f"row {row.get('id', index)} has no gold answer")

    low_images = _first_existing(row, ("images_72dpi", "image_dpi72", "image", "images"))
    high_images = _first_existing(row, ("images_144dpi", "image_dpi144", "images_96dpi", "image_dpi96", "image", "images"))
    if len(low_images) != len(high_images):
        raise ValueError(f"row {row.get('id', index)} low/high page count differs")

    # Keep only JSON-safe metadata needed by the tool/reward.  The full source
    # row is deliberately not copied into every Parquet cell.
    extra_info = {
        "index": index,
        "id": str(row.get("id", index)),
        "gold": gold,
        "num_pages": int(row.get("num_pages", len(low_images))),
        "evidence_locations": metadata.get("evidence_locations", []),
        "evidence_bboxes_by_dpi": metadata.get("evidence_bboxes_by_dpi", {}),
        "task": metadata.get("task", ""),
        "ruler_version": metadata.get("ruler_version", ""),
    }

    return {
        "data_source": DATA_SOURCE,
        "env_name": ENV_NAME,
        "enable_tools": True,
        "prompt": _as_messages(conversation),
        "images": low_images,
        "high_res_images": high_images,
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": extra_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=32, help="0 means all rows")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--task", action="append", default=[], help="optional metadata.task filter (repeatable)")
    parser.add_argument("--ruler-length", type=int, default=0, help="optional metadata.ruler_length filter")
    args = parser.parse_args()

    rows = []
    seen = 0
    selected = 0
    with args.input.open(encoding="utf-8") as source_file:
        for line in source_file:
            if not line.strip():
                continue
            row = json.loads(line)
            task = str((row.get("metadata") or {}).get("task", ""))
            if args.task and task not in args.task:
                continue
            if args.ruler_length and int((row.get("metadata") or {}).get("ruler_length", 0)) != args.ruler_length:
                continue
            if seen < args.skip:
                seen += 1
                continue
            rows.append(convert_row(row, args.input, selected))
            selected += 1
            if args.limit > 0 and selected >= args.limit:
                break

    if not rows:
        raise RuntimeError("no rows selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.Dataset.from_list(rows)
    dataset.to_parquet(str(args.output))
    print(f"wrote {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
