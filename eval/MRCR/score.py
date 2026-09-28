#!/usr/bin/env python3
"""Apply OpenAI MRCR's official prefix check and SequenceMatcher metric."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SUBSETS = ("2needle", "4needle", "8needle")
LENGTH_BINS = (
    "4K-8K",
    "8K-16K",
    "16K-32K",
    "32K-64K",
    "64K-128K",
    "128K-256K",
    "256K-512K",
    "512K-1M",
)


def grade(response: str, answer: str, prefix: str) -> float:
    """Metric from the openai/mrcr dataset card."""
    if not response.startswith(prefix):
        return 0.0
    response = response.removeprefix(prefix)
    answer = answer.removeprefix(prefix)
    return float(SequenceMatcher(None, response, answer).ratio())


def _prompt_token_count(record: dict[str, Any]) -> int | None:
    """Use the initial request length, not cumulative tool-turn usage."""
    value = record.get("initial_prompt_tokens")
    if value is None:
        value = record.get("prompt_tokens")
    return int(value) if value is not None else None


def _response_token_count(record: dict[str, Any]) -> int | None:
    """Use completion tokens summed across the whole agent trajectory."""
    usage = record.get("cumulative_usage")
    value = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if value is None:
        value = record.get("completion_tokens")
    return int(value) if value is not None else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    requested = len(records)
    skipped = sum(bool(record.get("skipped_reason")) for record in records)
    errors = sum(bool(record.get("error")) for record in records)
    generated = [
        record
        for record in records
        if not record.get("skipped_reason") and not record.get("error")
    ]
    scores = [
        grade(
            record.get("generation") or "",
            record["answer"],
            record["random_string_to_prepend"],
        )
        for record in generated
    ]
    prefix_hits = sum(
        (record.get("generation") or "").startswith(
            record["random_string_to_prepend"]
        )
        for record in generated
    )
    prompt_tokens = [
        value for record in generated
        if (value := _prompt_token_count(record)) is not None
    ]
    response_tokens = [
        value for record in generated
        if (value := _response_token_count(record)) is not None
    ]
    pages = [record.get("num_pages", 0) or 0 for record in records]
    return {
        "requested": requested,
        "generated": len(generated),
        "skipped": skipped,
        "errors": errors,
        "truncated": sum(
            record.get("finish_reason") == "length" for record in generated
        ),
        "coverage_pct": round(100 * len(generated) / requested, 2) if requested else 0,
        "score_all": round(100 * sum(scores) / requested, 2) if requested else 0,
        "score_generated": round(100 * sum(scores) / len(scores), 2) if scores else 0,
        "prefix_hit_pct": round(100 * prefix_hits / len(generated), 2)
        if generated
        else 0,
        "mean_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens))
        if prompt_tokens
        else None,
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else None,
        "mean_response_tokens": round(sum(response_tokens) / len(response_tokens))
        if response_tokens
        else None,
        "max_response_tokens": max(response_tokens) if response_tokens else None,
        "mean_pages": round(sum(pages) / len(pages), 1) if pages else 0,
        "max_pages": max(pages) if pages else 0,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    model = summary["model"]
    overall = summary["overall"]
    lines = [f"# MRCR VTC @ {model}", ""]
    if summary.get("input_mode") == "text":
        input_description = (
            "Requests use the original MRCR multi-turn text prompts. Over-context "
            "samples are counted in `score_all` as zero and excluded only from "
            "`score_generated`; prompts are never truncated."
        )
    else:
        input_description = (
            "Every image is preceded by an explicit `Page N` marker. Over-context "
            "samples are counted in `score_all` as zero and excluded only from "
            "`score_generated`; no pages are truncated."
        )
    lines += [
        f"System prompt: `{summary['system_prompt']}`",
        "",
        input_description,
        "",
        "## Overall",
        "",
        "| requested | generated | coverage | skipped | errors | hit max_tokens | score_all | score_generated | prefix hit | mean prompt tok | mean response tok |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {overall['requested']} | {overall['generated']} | {overall['coverage_pct']:.2f}% "
        f"| {overall['skipped']} | {overall['errors']} | {overall['truncated']} "
        f"| **{overall['score_all']:.2f}** | **{overall['score_generated']:.2f}** "
        f"| {overall['prefix_hit_pct']:.2f}% | {overall['mean_prompt_tokens'] or '-'} "
        f"| {overall['mean_response_tokens'] or '-'} |",
        "",
        "## By subset and official length bin",
        "",
        "| subset | length | requested | generated | coverage | skipped | errors | score_all | score_generated | prefix hit | mean prompt tok | mean response tok |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in SUBSETS:
        for length_bin in LENGTH_BINS:
            cell = summary["groups"].get(subset, {}).get(length_bin)
            if not cell:
                continue
            lines.append(
                f"| {subset} | {length_bin} | {cell['requested']} | "
                f"{cell['generated']} | {cell['coverage_pct']:.2f}% | "
                f"{cell['skipped']} | {cell['errors']} | {cell['score_all']:.2f} | "
                f"{cell['score_generated']:.2f} | {cell['prefix_hit_pct']:.2f}% | "
                f"{cell['mean_prompt_tokens'] or '-'} | "
                f"{cell['mean_response_tokens'] or '-'} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--label")
    args = parser.parse_args()

    all_records: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for subset in SUBSETS:
        result_path = args.results_dir / f"{subset}.jsonl"
        if not result_path.exists():
            continue
        for line in result_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            all_records.append(record)
            grouped[(record["subset"], record["length_bin"])].append(record)
    if not all_records:
        raise SystemExit(f"no MRCR outputs under {args.results_dir}")

    config_path = args.results_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = {
        "model": args.label or config["model_name"],
        "system_prompt": config.get("system_prompt"),
        "input_mode": config.get("input_mode", "vision"),
        "page_markers": config.get("page_markers", True),
        "overall": summarize(all_records),
        "groups": {subset: {} for subset in SUBSETS},
    }
    for (subset, length_bin), records in grouped.items():
        summary["groups"][subset][length_bin] = summarize(records)

    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.results_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "subset",
                "length_bin",
                "requested",
                "generated",
                "coverage_pct",
                "skipped",
                "errors",
                "truncated",
                "score_all",
                "score_generated",
                "prefix_hit_pct",
                "mean_prompt_tokens",
                "max_prompt_tokens",
                "mean_response_tokens",
                "max_response_tokens",
                "mean_pages",
                "max_pages",
            ]
        )
        for subset in SUBSETS:
            for length_bin in LENGTH_BINS:
                cell = summary["groups"][subset].get(length_bin)
                if cell:
                    writer.writerow(
                        [subset, length_bin]
                        + [
                            cell["requested"],
                            cell["generated"],
                            cell["coverage_pct"],
                            cell["skipped"],
                            cell["errors"],
                            cell["truncated"],
                            cell["score_all"],
                            cell["score_generated"],
                            cell["prefix_hit_pct"],
                            cell["mean_prompt_tokens"],
                            cell["max_prompt_tokens"],
                            cell["mean_response_tokens"],
                            cell["max_response_tokens"],
                            cell["mean_pages"],
                            cell["max_pages"],
                        ]
                    )
    write_markdown(summary, args.results_dir / "summary.md")

    overall = summary["overall"]
    print(
        f"{summary['model']}: generated={overall['generated']}/{overall['requested']} "
        f"score_all={overall['score_all']:.2f} "
        f"score_generated={overall['score_generated']:.2f}"
    )
    print(f"-> {args.results_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
