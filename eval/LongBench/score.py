#!/usr/bin/env python3
# Modified for FocusVTC; derived from THUDM/LongBench (see LICENSE).
# Copyright (c) 2023 THU-KEG & Zhipu AI
# SPDX-License-Identifier: MIT
"""Score the LongBench runs produced by run_eval.py.

Metrics are ported verbatim from THUDM/LongBench (`LongBench/metrics.py` and the
`scorer` in `LongBench/eval.py`), including the first-line truncation applied to
the few-shot datasets, so numbers are comparable with the public leaderboard.

Writes `summary.json`, `summary.csv` and `summary.md` next to the results.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import jieba
from fuzzywuzzy import fuzz
from rouge import Rouge

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "config"
DATASET2METRIC = json.loads((CONFIG_DIR / "dataset2metric.json").read_text(encoding="utf-8"))

# LongBench's own grouping, used for the category rollup on the leaderboard.
GROUPS = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Few-shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Synthetic": ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"],
    "Code": ["lcc", "repobench-p"],
}
DATASET_ORDER = [d for ds in GROUPS.values() for d in ds]

# Chinese subsets, skipped unless --include-zh. The VTC SFT data is English-only,
# so these measure the base model's Chinese rather than anything about rendering.
ZH_DATASETS = {"multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh"}

# upstream truncates these to the first line before scoring
FIRST_LINE_ONLY = {"trec", "triviaqa", "samsum", "lsht"}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def normalize_zh_answer(s):
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    all_punctuation = set(string.punctuation + cn_punctuation)
    return "".join(ch for ch in "".join(s.lower().split()) if ch not in all_punctuation)


def _f1(prediction_tokens, ground_truth_tokens):
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1(prediction, ground_truth, **_):
    return _f1(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def qa_f1_zh(prediction, ground_truth, **_):
    pred = [normalize_zh_answer(t) for t in jieba.cut(prediction, cut_all=False)]
    gold = [normalize_zh_answer(t) for t in jieba.cut(ground_truth, cut_all=False)]
    pred = [t for t in pred if t]
    gold = [t for t in gold if t]
    if not pred or not gold:
        return 0.0
    return _f1(pred, gold)


def rouge(prediction, ground_truth, **_):
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:  # noqa: BLE001 - upstream swallows empty/degenerate strings too
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh(prediction, ground_truth, **_):
    return rouge(
        " ".join(jieba.cut(prediction, cut_all=False)),
        " ".join(jieba.cut(ground_truth, cut_all=False)),
    )


def classification(prediction, ground_truth, all_classes=None, **_):
    em_match_list = [c for c in (all_classes or []) if c in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    return 1.0 / len(em_match_list) if ground_truth in em_match_list else 0.0


def _number_hit_rate(prediction, target):
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1 for n in numbers if str(n) == str(target)) / len(numbers)


def count(prediction, ground_truth, **_):
    return _number_hit_rate(prediction, ground_truth)


def retrieval(prediction, ground_truth, **_):
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    return _number_hit_rate(prediction, matches[0])


def retrieval_zh(prediction, ground_truth, **_):
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    return _number_hit_rate(prediction, matches[0])


def code_sim(prediction, ground_truth, **_):
    line = ""
    for candidate in prediction.lstrip("\n").split("\n"):
        if ("`" not in candidate) and ("#" not in candidate) and ("//" not in candidate):
            line = candidate
            break
    return fuzz.ratio(line, ground_truth) / 100


METRICS = {
    "qa_f1": qa_f1,
    "qa_f1_zh": qa_f1_zh,
    "rouge": rouge,
    "rouge_zh": rouge_zh,
    "classification": classification,
    "count": count,
    "retrieval": retrieval,
    "retrieval_zh": retrieval_zh,
    "code_sim": code_sim,
}


def score_dataset(dataset, records):
    metric = METRICS[DATASET2METRIC[dataset]]
    total = 0.0
    for rec in records:
        prediction = rec.get("generation") or ""
        if dataset in FIRST_LINE_ONLY:
            prediction = prediction.lstrip("\n").split("\n")[0]
        best = 0.0
        for gold in rec["answers"]:
            best = max(best, metric(prediction, gold, all_classes=rec.get("all_classes")))
        total += best
    return 100 * total / len(records) if records else 0.0


def summarize_run(run_dir, skip=frozenset()):
    per_dataset = {}
    for dataset in DATASET_ORDER:
        if dataset in skip:
            continue
        path = run_dir / f"{dataset}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(l) for l in open(path, encoding="utf-8")]
        if not records:
            continue
        # Tool-agent gateways can make several backend requests for one sample.
        # Prefer the separately reconstructed first-turn length when present;
        # keep the gateway's accumulated per-turn prompt usage as a diagnostic.
        prompt_tokens = [
            r.get("initial_prompt_tokens", r.get("prompt_tokens"))
            for r in records
            if r.get("initial_prompt_tokens", r.get("prompt_tokens"))
        ]
        cumulative_prompt_tokens = [
            r["prompt_tokens"]
            for r in records
            if r.get("initial_prompt_tokens") and r.get("prompt_tokens")
        ]
        pages = [r.get("num_pages", 0) or 0 for r in records]
        truncated = sum(1 for r in records if r.get("finish_reason") == "length")
        per_dataset[dataset] = {
            "score": round(score_dataset(dataset, records), 2),
            "n": len(records),
            "errors": sum(1 for r in records if r.get("error")),
            "truncated": truncated,
            "mean_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None,
            "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else None,
            "mean_cumulative_prompt_tokens": (
                round(sum(cumulative_prompt_tokens) / len(cumulative_prompt_tokens))
                if cumulative_prompt_tokens
                else None
            ),
            "max_cumulative_prompt_tokens": (
                max(cumulative_prompt_tokens) if cumulative_prompt_tokens else None
            ),
            "mean_pages": round(sum(pages) / len(pages), 1) if any(pages) else 0,
            "max_pages": max(pages) if pages else 0,
        }

    if not per_dataset:
        return None

    groups = {}
    for name, members in GROUPS.items():
        vals = [per_dataset[d]["score"] for d in members if d in per_dataset]
        if vals:
            groups[name] = round(sum(vals) / len(vals), 2)

    scores = [v["score"] for v in per_dataset.values()]
    toks = [v["mean_prompt_tokens"] for v in per_dataset.values() if v["mean_prompt_tokens"]]
    cumulative_toks = [
        v["mean_cumulative_prompt_tokens"]
        for v in per_dataset.values()
        if v["mean_cumulative_prompt_tokens"]
    ]
    return {
        "overall": round(sum(scores) / len(scores), 2),
        "groups": groups,
        "n_datasets": len(per_dataset),
        "mean_prompt_tokens": round(sum(toks) / len(toks)) if toks else None,
        "mean_cumulative_prompt_tokens": (
            round(sum(cumulative_toks) / len(cumulative_toks)) if cumulative_toks else None
        ),
        "datasets": per_dataset,
    }


def default_label(results_dir):
    """Model name for the report title, recovered from the results dir suffix."""
    name = Path(results_dir).name
    return "Qwen3-VL-8B-Instruct" if name == "results" else name.removeprefix("results_")


def write_markdown(summary, path, results_dir, skipped=frozenset(), label=None):
    runs = list(summary)
    label = label or default_label(results_dir)
    lines = [f"# LongBench @ {label}: text vs. rendered pages", ""]
    lines.append(f"Results dir: `{results_dir}`")
    if skipped:
        lines.append("")
        lines.append(f"Excluded: {', '.join(f'`{d}`' for d in sorted(skipped))}")
    lines.append("")

    lines += ["## Overall", "", "| run | overall | " + " | ".join(GROUPS) + " | mean prompt tok |", "|" + "---|" * (len(GROUPS) + 3)]
    for run in runs:
        s = summary[run]
        cells = [f"{s['groups'][g]:.2f}" if g in s["groups"] else "-" for g in GROUPS]
        lines.append(f"| {run} | **{s['overall']:.2f}** | " + " | ".join(cells) + f" | {s['mean_prompt_tokens']} |")
    lines.append("")
    if any(summary[run].get("mean_cumulative_prompt_tokens") for run in runs):
        lines.append(
            "`mean prompt tok` uses the first model request. For tool-agent runs, "
            "the separately reported cumulative value sums the full logical prompt "
            "reported by every backend turn."
        )
        lines.append("")

    lines += ["## Per dataset", "", "| dataset | " + " | ".join(runs) + " | " + " | ".join(f"tok({r})" for r in runs) + " |", "|" + "---|" * (1 + 2 * len(runs))]
    for dataset in DATASET_ORDER:
        if not any(dataset in summary[r]["datasets"] for r in runs):
            continue
        score_cells, tok_cells = [], []
        for run in runs:
            d = summary[run]["datasets"].get(dataset)
            score_cells.append(f"{d['score']:.2f}" if d else "-")
            tok_cells.append(str(d["mean_prompt_tokens"]) if d and d["mean_prompt_tokens"] else "-")
        lines.append(f"| {dataset} | " + " | ".join(score_cells) + " | " + " | ".join(tok_cells) + " |")
    lines.append("")

    image_runs = [r for r in runs if any(d["max_pages"] for d in summary[r]["datasets"].values())]
    if image_runs:
        lines += ["## Page cost", "", "| dataset | " + " | ".join(f"pages({r})" for r in image_runs) + " | " + " | ".join(f"max pages({r})" for r in image_runs) + " |", "|" + "---|" * (1 + 2 * len(image_runs))]
        for dataset in DATASET_ORDER:
            cells = [summary[r]["datasets"].get(dataset) for r in image_runs]
            if not any(cells):
                continue
            lines.append(
                f"| {dataset} | "
                + " | ".join(f"{c['mean_pages']}" if c else "-" for c in cells)
                + " | "
                + " | ".join(f"{c['max_pages']}" if c else "-" for c in cells)
                + " |"
            )
        lines.append("")

    agent_runs = [r for r in runs if summary[r].get("mean_cumulative_prompt_tokens")]
    if agent_runs:
        lines += [
            "## Tool-loop token accounting",
            "",
            "| run | mean initial prompt tok | mean cumulative prompt tok | cumulative / initial |",
            "|---|---:|---:|---:|",
        ]
        for run in agent_runs:
            initial = summary[run]["mean_prompt_tokens"]
            cumulative = summary[run]["mean_cumulative_prompt_tokens"]
            lines.append(f"| {run} | {initial} | {cumulative} | {cumulative / initial:.2f}× |")
        lines.append("")

    bad = [
        (run, dataset, d)
        for run in runs
        for dataset in DATASET_ORDER
        if (d := summary[run]["datasets"].get(dataset)) and (d["errors"] or d["truncated"])
    ]
    if bad:
        lines += ["## Errors and clipped generations", "", "| run | dataset | n | errors | hit max_tokens |", "|---|---|---|---|---|"]
        for run, dataset, d in bad:
            lines.append(f"| {run} | {dataset} | {d['n']} | {d['errors']} | {d['truncated']} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(summary, path):
    rows = [
        "run,dataset,score,n,errors,truncated,mean_prompt_tokens,max_prompt_tokens,"
        "mean_cumulative_prompt_tokens,max_cumulative_prompt_tokens,mean_pages,max_pages"
    ]
    for run, s in summary.items():
        for dataset in DATASET_ORDER:
            d = s["datasets"].get(dataset)
            if not d:
                continue
            rows.append(
                f"{run},{dataset},{d['score']},{d['n']},{d['errors']},{d['truncated']},"
                f"{d['mean_prompt_tokens']},{d['max_prompt_tokens']},"
                f"{d['mean_cumulative_prompt_tokens']},{d['max_cumulative_prompt_tokens']},"
                f"{d['mean_pages']},{d['max_pages']}"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument(
        "--include-zh",
        action="store_true",
        help=f"score the Chinese subsets too ({', '.join(sorted(ZH_DATASETS))})",
    )
    ap.add_argument("--label", help="model name for the report title")
    args = ap.parse_args()

    skip = frozenset() if args.include_zh else frozenset(ZH_DATASETS)

    run_dirs = sorted(
        (p for p in args.results_dir.iterdir() if p.is_dir()),
        key=lambda p: (p.name != "text", p.name),
    )
    summary = {}
    for run_dir in run_dirs:
        s = summarize_run(run_dir, skip)
        if s:
            summary[run_dir.name] = s

    if not summary:
        raise SystemExit(f"no scorable runs under {args.results_dir}")

    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(summary, args.results_dir / "summary.csv")
    write_markdown(summary, args.results_dir / "summary.md", args.results_dir, skip, args.label)

    for run, s in summary.items():
        print(f"{run:10s} overall={s['overall']:6.2f}  mean_prompt_tokens={s['mean_prompt_tokens']}")
    print(f"-> {args.results_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
