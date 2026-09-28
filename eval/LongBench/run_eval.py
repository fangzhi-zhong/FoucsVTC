#!/usr/bin/env python3
"""Run Qwen3-VL-8B over LongBench, either from rendered pages or from raw text.

Both modalities use LongBench's official `dataset2prompt.json`. In the image run
the `{context}` slot was replaced at render time by one `<image>` per page, so
the only difference between the two runs is how the context reaches the model.
`dataset2maxlen.json` sets the generation budget per dataset, as upstream does.

Generations land in `results/<run>/<dataset>.jsonl`; scoring is a separate step
(`score.py`) so a crashed run can be resumed without re-generating.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from think_prompt import apply_think, strip_think  # noqa: E402
from paths import data_path, resolve_images  # noqa: E402

VTC_ROOT = data_path("LongBench_VTC")
TEXT_ROOT = data_path("LongBench/data")
CONFIG_DIR = HERE / "config"

PROMPTS = json.loads((CONFIG_DIR / "dataset2prompt.json").read_text(encoding="utf-8"))
MAXLEN = json.loads((CONFIG_DIR / "dataset2maxlen.json").read_text(encoding="utf-8"))
DATASETS = list(PROMPTS)
SYSTEM_PROMPT = "You are a helpful assistant"
GROUNDING_INSTRUCTION = (
    "When thinking, ground specific visual evidence only when localization is needed. "
    "Every grounded region MUST use exactly this format: "
    "Page N <|box_start|>[x1, y1, x2, y2]<|box_end|>. "
    "Here N is the 1-based page number; x1, y1, x2, y2 are integer coordinates "
    "normalized to 0-1000 in left, top, right, bottom order. Put exactly one space "
    "between Page N and <|box_start|>; keep the four numbers inside square brackets "
    "and do not use parentheses, JSON, pixel coordinates, or any other bbox syntax. "
    "For counting, aggregation, or whole-document reasoning, reason concisely "
    "without enumerating or grounding every item."
)
_PAGE_BOX = re.compile(
    r"Page\s+(\d+)\s*<\|box_start\|>\s*"
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
    r"\s*<\|box_end\|>",
    re.IGNORECASE,
)


def extract_evidence_boxes(text):
    """Parse normalized, page-aware evidence boxes from model reasoning."""
    return [
        {"page": int(page), "bbox": [int(x1), int(y1), int(x2), int(y2)]}
        for page, x1, y1, x2, y2 in _PAGE_BOX.findall(text)
    ]


def build_messages(
    row,
    modality,
    think=False,
    disable_thinking=False,
    native_thinking=False,
    system_prompt=SYSTEM_PROMPT,
    page_markers=False,
    training_newline=False,
    ground_evidence=False,
    image_prompt_suffix=None,
):
    """Return (messages, extra_body) for one sample."""
    extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if disable_thinking
        else ({"chat_template_kwargs": {"enable_thinking": True}} if native_thinking else {})
    )
    question = row["question"]
    messages = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    if modality == "text":
        messages.append({"role": "user", "content": question})
        return apply_think(messages, extra_body) if think else (messages, extra_body)

    images = row["images"]
    segments = question.split("<image>")
    if len(segments) - 1 != len(images):
        raise ValueError(
            f"{row['dataset']}#{row['index']}: {len(segments) - 1} placeholders "
            f"but {len(images)} images"
        )

    parts = []
    for i, image in enumerate(images):
        seg = segments[i]
        if seg:
            parts.append({"type": "text", "text": seg})
        if page_markers:
            parts.append({"type": "text", "text": f"Page {i + 1}"})
        parts.append({"type": "image_url", "image_url": {"url": f"file://{image}"}})
        if training_newline:
            parts.append({"type": "text", "text": "\n"})
    if segments[-1]:
        parts.append({"type": "text", "text": segments[-1]})
    if ground_evidence:
        parts.append({"type": "text", "text": "\n" + GROUNDING_INSTRUCTION})
    if image_prompt_suffix:
        parts.append({"type": "text", "text": "\n" + image_prompt_suffix})
    messages.append({"role": "user", "content": parts})
    return apply_think(messages, extra_body) if think else (messages, extra_body)


def load_rows(modality, dpi, dataset, limit, vtc_root=VTC_ROOT):
    rows = []
    if modality == "text":
        with open(TEXT_ROOT / f"{dataset}.jsonl", encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                rows.append(
                    {
                        "index": i,
                        "dataset": dataset,
                        "question": PROMPTS[dataset].format(
                            context=r["context"], input=r["input"]
                        ),
                        "answers": r["answers"],
                        "all_classes": r["all_classes"],
                        "length": r["length"],
                        "context_chars": len(r["context"]),
                    }
                )
                if limit and len(rows) >= limit:
                    break
    else:
        with open(vtc_root / f"dpi_{dpi}" / dataset / "test.jsonl", encoding="utf-8") as f:
            for line in f:
                rows.append(resolve_images(json.loads(line), vtc_root / f"dpi_{dpi}" / dataset))
                if limit and len(rows) >= limit:
                    break
    return rows


async def generate_one(client, sem, args, row, modality, max_tokens):
    messages, extra_body = build_messages(
        row,
        modality,
        args.think,
        args.disable_thinking,
        args.native_thinking,
        system_prompt=args.system_prompt,
        page_markers=args.page_markers,
        training_newline=args.training_newline,
        ground_evidence=args.ground_evidence,
        image_prompt_suffix=args.image_prompt_suffix,
    )
    reasoning = args.think or args.strip_think or args.native_thinking
    if reasoning:
        # LongBench budgets are 32-512 tokens, too tight for a reasoning block.
        max_tokens += args.think_budget
    last_err = None
    for attempt in range(args.retries + 1):
        async with sem:
            t0 = time.time()
            try:
                resp = await client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                    max_tokens=max_tokens,
                    timeout=args.timeout,
                    extra_body=(
                        extra_body
                        | {
                            "top_k": args.top_k,
                            "min_p": args.min_p,
                            "repetition_penalty": args.repetition_penalty,
                        }
                        if extra_body or args.top_k is not None or args.min_p is not None
                        else None
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - surface any client/server error
                last_err = repr(exc)
                await asyncio.sleep(min(2**attempt, 30))
                continue
        if resp is None or not getattr(resp, "choices", None):
            last_err = "empty_response"
            continue
        usage = resp.usage
        response_payload = resp.model_dump()
        agent_trajectory = response_payload.get("vtc_model_outputs")
        tool_trace = response_payload.get("vtc_tool_trace")
        terminal_raw = response_payload.get("vtc_raw_generation")
        initial_prompt_tokens = response_payload.get("vtc_initial_prompt_tokens")
        cumulative_usage = response_payload.get("vtc_cumulative_usage")
        message = resp.choices[0].message
        content = message.content or ""
        native_reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        raw = content
        if native_reasoning:
            raw = f"<think>\n{native_reasoning}\n</think>\n\n{content}"
        record = {
            "index": row["index"],
            "dataset": row["dataset"],
            "generation": strip_think(raw) if reasoning else raw,
            **({"reasoning_content": native_reasoning} if native_reasoning else {}),
            "evidence_boxes": extract_evidence_boxes(raw),
            "answers": row["answers"],
            "all_classes": row["all_classes"],
            "length": row["length"],
            "vision_tokens_est": row.get("vision_tokens"),
            "num_pages": row.get("num_pages", 0),
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "finish_reason": resp.choices[0].finish_reason,
            "latency_s": round(time.time() - t0, 2),
        }
        if reasoning or terminal_raw is not None:
            record["raw_generation"] = terminal_raw if terminal_raw is not None else raw
        if agent_trajectory is not None:
            record["agent_trajectory"] = agent_trajectory
        if tool_trace is not None:
            record["tool_trace"] = tool_trace
        if response_payload.get("vtc_stop_reason") is not None:
            record["stop_reason"] = response_payload["vtc_stop_reason"]
        if response_payload.get("vtc_trajectory_tokens") is not None:
            record["trajectory_tokens"] = response_payload["vtc_trajectory_tokens"]
        if initial_prompt_tokens is not None:
            record["initial_prompt_tokens"] = initial_prompt_tokens
        if cumulative_usage is not None:
            record["cumulative_usage"] = cumulative_usage
        return record
    return {
        "index": row["index"],
        "dataset": row["dataset"],
        "generation": "",
        "answers": row["answers"],
        "all_classes": row["all_classes"],
        "length": row["length"],
        "error": last_err,
    }


async def run_split(client, args, modality, dpi, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    for dataset in args.datasets:
        out_path = out_dir / f"{dataset}.jsonl"
        done = set()
        if args.resume and out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if not rec.get("error"):
                        done.add(rec["index"])

        rows = [
            r for r in load_rows(modality, dpi, dataset, args.limit, args.vtc_root) if r["index"] not in done
        ]
        if not rows:
            print(f"[{out_dir.name}] {dataset}: nothing to do", flush=True)
            continue

        if args.dry_run:
            def user_parts(row):
                msgs = build_messages(
                    row,
                    modality,
                    args.think,
                    args.disable_thinking,
                    args.native_thinking,
                    system_prompt=args.system_prompt,
                    page_markers=args.page_markers,
                    training_newline=args.training_newline,
                    ground_evidence=args.ground_evidence,
                    image_prompt_suffix=args.image_prompt_suffix,
                )[0]
                return next(m for m in msgs if m["role"] == "user")["content"]

            parts = sum(len(user_parts(r)) for r in rows) if modality != "text" else 0
            pages = sum(r.get("num_pages", 0) for r in rows)
            print(
                f"[{out_dir.name}] {dataset}: {len(rows)} prompts built ok"
                f"{f', {parts} content parts, {pages} pages' if parts else ''}",
                flush=True,
            )
            continue

        max_tokens = MAXLEN[dataset]
        coros = [generate_one(client, sem, args, r, modality, max_tokens) for r in rows]
        with open(out_path, "a", encoding="utf-8") as f:
            with tqdm(total=len(coros), desc=f"{out_dir.name}/{dataset}") as bar:
                for coro in asyncio.as_completed(coros):
                    rec = await coro
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    bar.update(1)

        by_index = {}
        for line in open(out_path, encoding="utf-8"):
            rec = json.loads(line)
            by_index[rec["index"]] = rec
        records = [by_index[k] for k in sorted(by_index)]
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_err = sum(1 for r in records if r.get("error"))
        print(f"[{out_dir.name}] {dataset}: {len(records)} done, {n_err} errors", flush=True)


async def main_async(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0)
    for spec in args.runs:
        if spec == "text":
            await run_split(client, args, "text", None, args.results_dir / "text")
        else:
            await run_split(client, args, "image", spec, args.results_dir / f"dpi_{spec}")


def main():
    global TEXT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18350/v1")
    ap.add_argument("--model-name", default="FocusVTC")
    ap.add_argument(
        "--runs",
        nargs="+",
        default=["text", "72"],
        help="'text' for the text-only baseline, otherwise the tag of a render "
        "directory: a bare DPI like '72' for LongBench_VTC/dpi_72, or a DPI with "
        "a variant suffix like '72_dejavu' for a font ablation",
    )
    ap.add_argument("--text-root", type=Path, default=TEXT_ROOT)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--vtc-root", type=Path, default=VTC_ROOT,
                    help="render root containing dpi_*; use LongBench_VTC/fewshot_input for visual queries")
    ap.add_argument("--limit", type=int, default=0, help="samples per dataset, 0 = all")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--system-prompt",
        default=SYSTEM_PROMPT,
        help="system message prepended to every request; pass an empty string to disable",
    )
    ap.add_argument(
        "--page-markers",
        action="store_true",
        help="insert `Page N` immediately before every image, matching VTC SFT inputs",
    )
    ap.add_argument(
        "--training-newline",
        action="store_true",
        help="insert a literal newline text part after every image",
    )
    ap.add_argument(
        "--ground-evidence",
        action="store_true",
        help="append the VTC visual-evidence grounding instruction to image prompts",
    )
    ap.add_argument(
        "--image-prompt-suffix",
        default=None,
        help="optional text appended only to image prompts after the grounding instruction",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument(
        "--think", action="store_true",
        help="reshape prompts into the VTC SFT training format and prefill an open "
        "<think> turn; the reasoning block is stripped from `generation` and kept "
        "in `raw_generation`",
    )
    ap.add_argument(
        "--native-thinking", action="store_true",
        help="enable Qwen3.5 native thinking without prefilled <think> prompt",
    )
    ap.add_argument(
        "--strip-think", action="store_true",
        help="the prompt is left alone, but a `<think>...</think>` block the model "
        "emits on its own is moved out of `generation` into `raw_generation`. Use "
        "for models that always reason, e.g. Glyph on its GLM-4.1V backbone",
    )
    ap.add_argument(
        "--disable-thinking", action="store_true",
        help="pass chat_template_kwargs.enable_thinking=false (Qwen3.5 direct-answer mode)",
    )
    ap.add_argument(
        "--think-budget", type=int, default=768,
        help="extra max_tokens added on top of the LongBench budget under "
        "--think or --strip-think",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--config-name", default="run_config.json", help="give parallel workers distinct names")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build every prompt and validate placeholder/image alignment without calling the server",
    )
    args = ap.parse_args()
    TEXT_ROOT = args.text_root.resolve()
    args.text_root = TEXT_ROOT
    args.vtc_root = args.vtc_root.resolve()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.config_name).name != args.config_name:
        raise SystemExit("--config-name must be a filename, not a path")
    config_path = args.results_dir / args.config_name
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if Path(previous.get("vtc_root", str(VTC_ROOT))).resolve() != args.vtc_root:
            raise SystemExit("Refusing to resume results from a different --vtc-root")
        resume_keys = (
            "text_root", "model_name", "runs", "datasets", "system_prompt", "page_markers",
            "training_newline", "ground_evidence", "image_prompt_suffix",
            "think", "strip_think", "disable_thinking", "native_thinking", "think_budget",
            "temperature", "top_p", "top_k", "min_p", "presence_penalty",
            "repetition_penalty",
        )
        mismatches = {
            key: (previous.get(key), getattr(args, key))
            for key in resume_keys
            if previous.get(key) != (str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key))
        }
        if mismatches:
            raise SystemExit(
                "Refusing to mix results generated with different settings: "
                f"{mismatches}. Use a new --results-dir or remove that split first."
            )
    with open(config_path, "w", encoding="utf-8") as f:
        cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        cfg["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(cfg, f, indent=2)

    asyncio.run(main_async(args))
    print("ALL_GENERATION_DONE")


if __name__ == "__main__":
    main()
