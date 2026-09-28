#!/usr/bin/env python3
# Modified for FocusVTC; metrics adapted from NVIDIA RULER / NeMo-Skills.
# SPDX-License-Identifier: Apache-2.0
"""Score the RULER v1 8k DPI ablation.

Metric definitions are ported from NeMo-Skills / upstream NVIDIA RULER so the
numbers stay comparable with the text-only RULER v1 leaderboard:

  nemo_skills/evaluation/evaluator/ruler.py  -> eval_ruler (string_match_all / _part)
  nemo_skills/dataset/ruler/prepare.py       -> MATCH_TYPE per task family
  nemo_skills/dataset/ruler/ruler_score.py   -> overall = unweighted mean of 13 tasks

Note this is plain case-insensitive substring matching. Unlike RULER v2 there is
no edit-distance relaxation -- do not copy `_hit` from ruler_v2_dpi/score.py.

Writes summary.json, summary.csv and summary.md into the results directory.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe",
    "qa_1", "qa_2",
]
# prepare.py: MATCH_TYPE = {niah: all, vt: all, cwe: all, fwe: all, qa: part},
# keyed on the first underscore-separated segment of the task name.
METRIC = {t: ("part" if t.startswith("qa") else "all") for t in TASKS}

_NON_PRINTABLE = re.compile(r"[\x00-\x1f]")


def score_all(pred: str, refs) -> float:
    """Recall: the fraction of gold answers that appear in the prediction."""
    pred = pred.lower()
    return sum(1.0 if str(r).lower() in pred else 0.0 for r in refs) / len(refs)


def score_part(pred: str, refs) -> float:
    """Binary: 1 if any gold answer appears in the prediction."""
    pred = pred.lower()
    return max(1.0 if str(r).lower() in pred else 0.0 for r in refs)


SCORERS = {"all": score_all, "part": score_part}


def score_file(path: Path, task: str) -> dict:
    scorer = SCORERS[METRIC[task]]
    scores, ptoks, ctoks, vtoks, pages = [], [], [], [], []
    errs = truncated = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("error"):
                errs += 1
                scores.append(0.0)
                continue
            gen = _NON_PRINTABLE.sub("\n", (rec.get("generation") or "")).strip()
            refs = rec["expected_answer"]
            if not isinstance(refs, list):
                refs = [refs]
            scores.append(scorer(gen, refs))
            # Tool-agent usage.prompt_tokens is cumulative over the whole
            # trajectory. Prefer the gateway's first-turn count so this stays
            # comparable with one-shot RULER runs; old artifacts fall back to
            # the legacy field until they are recomputed.
            prompt_tokens = rec.get("initial_prompt_tokens", rec.get("prompt_tokens"))
            if prompt_tokens:
                ptoks.append(prompt_tokens)
            if rec.get("completion_tokens"):
                ctoks.append(rec["completion_tokens"])
            if rec.get("vision_tokens_est"):
                vtoks.append(rec["vision_tokens_est"])
            if rec.get("num_pages"):
                pages.append(rec["num_pages"])
            if rec.get("finish_reason") == "length":
                truncated += 1
    mean = lambda xs: statistics.mean(xs) if xs else None  # noqa: E731
    return {
        "n": len(scores),
        "accuracy": statistics.mean(scores) if scores else 0.0,
        "prompt_tokens": mean(ptoks),
        "completion_tokens": mean(ctoks),
        "vision_tokens_est": mean(vtoks),
        "num_pages": mean(pages),
        "errors": errs,
        "truncated": truncated,
    }


def default_label(results_dir):
    """Model name for the report title, recovered from the results dir suffix."""
    name = Path(results_dir).name
    return "Qwen3-VL-8B-Instruct" if name == "results" else name.removeprefix("results_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--label", help="model name for the report title")
    args = ap.parse_args()
    label = args.label or default_label(args.results_dir)

    runs = []
    if (args.results_dir / "text").is_dir():
        runs.append(("text", args.results_dir / "text"))
    for d in sorted(args.results_dir.glob("dpi_*"), key=lambda p: int(p.name.split("_")[1])):
        runs.append((d.name, d))
    if not runs:
        raise SystemExit(f"no result splits under {args.results_dir}")

    summary = {}
    for name, d in runs:
        per_task = {}
        for task in TASKS:
            path = d / f"{task}.jsonl"
            if path.exists():
                per_task[task] = score_file(path, task)
        if not per_task:
            continue
        overall = statistics.mean(v["accuracy"] for v in per_task.values())
        summary[name] = {
            "overall": overall,
            "overall_valid": len(per_task) == len(TASKS),
            "mean_prompt_tokens": statistics.mean(
                [v["prompt_tokens"] for v in per_task.values() if v["prompt_tokens"]] or [0]
            ),
            "mean_completion_tokens": statistics.mean(
                [v["completion_tokens"] for v in per_task.values() if v["completion_tokens"]] or [0]
            ),
            "total_errors": sum(v["errors"] for v in per_task.values()),
            "total_truncated": sum(v["truncated"] for v in per_task.values()),
            "tasks": per_task,
        }

    with open(args.results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    header = ["run", "overall", "prompt_tokens", "completion_tokens", "errors", "truncated", *TASKS]
    lines = [",".join(header)]
    for name, s in summary.items():
        row = [
            name,
            f"{s['overall']:.4f}",
            f"{s['mean_prompt_tokens']:.0f}",
            f"{s['mean_completion_tokens']:.0f}",
            str(s["total_errors"]),
            str(s["total_truncated"]),
        ]
        row += [f"{s['tasks'][t]['accuracy']:.4f}" if t in s["tasks"] else "" for t in TASKS]
        lines.append(",".join(row))
    (args.results_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [f"# RULER v1 8k DPI ablation - {label}", ""]
    md.append("| run | overall | mean prompt tok | mean response tok | errors | truncated |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for name, s in summary.items():
        flag = "" if s["overall_valid"] else " (partial)"
        md.append(
            f"| {name} | {s['overall'] * 100:.1f}{flag} | {s['mean_prompt_tokens']:.0f} "
            f"| {s['mean_completion_tokens']:.0f} | {s['total_errors']} | {s['total_truncated']} |"
        )
    md += ["", "## per-task accuracy (%)", ""]
    md.append("| run | " + " | ".join(TASKS) + " |")
    md.append("| --- |" + " --- |" * len(TASKS))
    for name, s in summary.items():
        cells = [
            f"{s['tasks'][t]['accuracy'] * 100:.1f}" if t in s["tasks"] else "-" for t in TASKS
        ]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md += [
        "",
        "`mean prompt tok` is the first-turn rendered prompt length. Tool-loop "
        "records retain cumulative backend prompt usage in `prompt_tokens`.",
        "",
        "`mean response tok` is the cumulative generated-token count across the "
        "whole tool trajectory, so it includes tool-call actions and reasoning.",
        "",
        "Metrics follow upstream RULER v1: `all` (recall over every gold answer) for "
        "the niah/vt/cwe/fwe tasks, `part` (any gold answer is a substring) for qa_1 "
        "and qa_2. Matching is plain case-insensitive substring, with no edit-distance "
        "relaxation. Overall is the unweighted mean of the 13 tasks and is only "
        "comparable when all 13 are present.",
        "",
        "`text` is the original text-only RULER v1 8k prompt on the same sample ids -- "
        "the ceiling the DPI curve should be read against.",
        "",
    ]
    (args.results_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md[: 6 + len(summary)]))
    print(f"\nwrote {args.results_dir}/summary.{{json,csv,md}}")


if __name__ == "__main__":
    main()
