#!/usr/bin/env python3
# Modified for FocusVTC; metrics adapted from NVIDIA RULER / NeMo-Skills.
# SPDX-License-Identifier: Apache-2.0
"""Score the RULER v2 DPI ablation.

Metric definitions are ported from NeMo-Skills, with a local retrieval-task
override: exact references found in model reasoning also receive credit for
selected non-MCQ retrieval tasks.

  nemo_skills/evaluation/evaluator/ruler.py   -> eval_ruler2 (all / part / 2steps)
  nemo_skills/evaluation/evaluator/mcq.py     -> eval_mcq    (multichoice)
  nemo_skills/dataset/ruler2/prepare.py       -> which task uses which metric
  nemo_skills/dataset/ruler2/ruler2_score.py  -> overall = unweighted mean of 12 tasks

Writes summary.json, summary.csv and summary.md into the results directory.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

try:
    import editdistance

    def _edit(a, b):
        return editdistance.eval(a, b)
except ImportError:  # keep the scorer runnable in envs without the package

    def _edit(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


HERE = Path(__file__).resolve().parent
TASKS = [
    "mk_niah_basic", "mk_niah_easy", "mk_niah_medium", "mk_niah_hard",
    "mv_niah_basic", "mv_niah_easy", "mv_niah_medium", "mv_niah_hard",
    "qa_basic", "qa_easy", "qa_medium", "qa_hard",
]
# from prepare_task_for_ns()
METRIC = {t: "all" for t in TASKS}
METRIC.update({
    "mk_niah_medium": "multichoice",
    "mk_niah_hard": "multichoice",
    "mv_niah_medium": "2steps",
    "qa_basic": "part",
    "qa_easy": "part",
    "qa_medium": "part",
    "qa_hard": "part",
})
REASONING_EXACT_TASKS = {
    "mk_niah_basic", "mk_niah_easy",
    "mv_niah_basic", "mv_niah_easy", "mv_niah_medium", "mv_niah_hard",
    "qa_basic", "qa_easy",
}

_NON_PRINTABLE = re.compile(r"[\x00-\x1f]")
_DOC_BLOCK = re.compile(r"Document \d+:(?:.*\n)+?\n")
_THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_LINE = re.compile(r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])")
_LONE_LETTER = re.compile(r"\b[A-Z]\b(?!.*\b[A-Z]\b)", re.DOTALL)


def wer(hyp: str, ref: str) -> float:
    h_list, r_list = hyp.split(), ref.split()
    if not r_list:
        return float("inf")
    return _edit(h_list, r_list) / len(r_list)


def _hit(pred: str, ref: str) -> float:
    pred, ref = pred.lower(), ref.lower()
    return max(1.0 if ref in pred else 0.0, 1 - wer(pred, ref))


def score_all(pred, refs):
    return sum(_hit(pred, r) for r in refs) / len(refs)


def score_2steps(pred, refs):
    return score_all(pred.split("\n\n")[-1], refs)


def score_part(pred, refs):
    return max(_hit(_DOC_BLOCK.sub("", pred), r) for r in refs)


def get_reasoning(rec) -> str:
    """Read native reasoning, falling back to the think block in old results."""
    reasoning = rec.get("reasoning_content") or ""
    if reasoning:
        return str(reasoning)
    raw = rec.get("raw_generation") or ""
    match = _THINK_BLOCK.search(str(raw))
    return match.group(1) if match else ""


def score_with_reasoning(task, gen, reasoning, refs) -> float:
    """Apply local exact-match reasoning credit to selected retrieval tasks."""
    base_score = SCORERS[METRIC[task]](gen, refs)
    if task not in REASONING_EXACT_TASKS or not reasoning:
        return base_score

    reasoning_lower = reasoning.lower()
    if METRIC[task] == "all":
        # Preserve score_all's per-reference averaging for multi-value tasks.
        return sum(
            max(_hit(gen, ref), float(str(ref).lower() in reasoning_lower))
            for ref in refs
        ) / len(refs)

    reasoning_hit = any(str(ref).lower() in reasoning_lower for ref in refs)
    return max(base_score, float(reasoning_hit))


def extract_letter(text):
    boxed = _BOXED.findall(text)
    candidates = []
    if boxed:
        candidates.append(boxed[-1].strip())
    m = _ANSWER_LINE.findall(text)
    if m:
        candidates.append(m[-1].strip())
    for cand in candidates:
        if len(cand) == 1:
            return cand.upper()
        hit = _LONE_LETTER.findall(cand)
        if hit:
            return hit[-1].upper()
    hit = _LONE_LETTER.findall(text)
    return hit[-1].upper() if hit else None


def score_multichoice(pred, refs):
    expected = refs[0] if isinstance(refs, list) else refs
    return 1.0 if extract_letter(pred) == str(expected).strip().upper() else 0.0


SCORERS = {"all": score_all, "2steps": score_2steps, "part": score_part, "multichoice": score_multichoice}


def score_file(path, task):
    scores, ptoks, ctoks, errs, truncated = [], [], [], 0, 0
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
            reasoning = _NON_PRINTABLE.sub("\n", get_reasoning(rec)).strip()
            scores.append(score_with_reasoning(task, gen, reasoning, refs))
            prompt_tokens = rec.get("initial_prompt_tokens", rec.get("prompt_tokens"))
            if prompt_tokens:
                ptoks.append(prompt_tokens)
            if rec.get("completion_tokens"):
                ctoks.append(rec["completion_tokens"])
            if rec.get("finish_reason") == "length":
                truncated += 1
    return {
        "n": len(scores),
        "accuracy": statistics.mean(scores) if scores else 0.0,
        "prompt_tokens": statistics.mean(ptoks) if ptoks else None,
        "completion_tokens": statistics.mean(ctoks) if ctoks else None,
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
        complete = len(per_task) == len(TASKS)
        overall = statistics.mean(v["accuracy"] for v in per_task.values())
        summary[name] = {
            "overall": overall,
            "overall_valid": complete,
            "mean_prompt_tokens": statistics.mean(
                [v["prompt_tokens"] for v in per_task.values() if v["prompt_tokens"]] or [0]
            ),
            "total_errors": sum(v["errors"] for v in per_task.values()),
            "total_truncated": sum(v["truncated"] for v in per_task.values()),
            "tasks": per_task,
        }

    with open(args.results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    header = ["run", "overall", "prompt_tokens", "errors", "truncated", *TASKS]
    lines = [",".join(header)]
    for name, s in summary.items():
        row = [
            name,
            f"{s['overall']:.4f}",
            f"{s['mean_prompt_tokens']:.0f}",
            str(s["total_errors"]),
            str(s["total_truncated"]),
        ]
        row += [f"{s['tasks'][t]['accuracy']:.4f}" if t in s["tasks"] else "" for t in TASKS]
        lines.append(",".join(row))
    (args.results_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [f"# RULER v2 DPI ablation - {label}", ""]
    md.append("| run | overall | mean prompt tok | errors | truncated |")
    md.append("| --- | --- | --- | --- | --- |")
    for name, s in summary.items():
        flag = "" if s["overall_valid"] else " (partial)"
        md.append(
            f"| {name} | {s['overall'] * 100:.1f}{flag} | {s['mean_prompt_tokens']:.0f} "
            f"| {s['total_errors']} | {s['total_truncated']} |"
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
        "Metrics follow NeMo-Skills: `all` (every reference must appear), `part` "
        "(substring of any reference), `2steps` (only the last paragraph is matched), "
        "`multichoice` (A/B/C/D letter). Overall is the unweighted mean of the 12 tasks "
        "and is only comparable when all 12 are present.",
        "Local override: exact references in reasoning also receive credit for "
        "`mk_niah_basic/easy`, all four `mv_niah` tasks, and `qa_basic/easy`. "
        "MCQ tasks and `qa_medium/hard` remain final-answer-only.",
        "",
    ]
    (args.results_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md[:6 + len(summary)]))
    print(f"\nwrote {args.results_dir}/summary.{{json,csv,md}}")


if __name__ == "__main__":
    main()
