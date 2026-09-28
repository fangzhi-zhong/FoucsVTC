#!/usr/bin/env python3
"""Run Qwen3-VL-8B over the RULER v2 DPI ablation via a vLLM OpenAI server.

For each DPI the rendered prompt is `<header><image>...<image><tail>`; the
placeholders are expanded into interleaved chat content parts pointing at the
PNG pages with file:// URLs (the server must run with
--allowed-local-media-path).

`--modality text` instead evaluates the original text-only RULER v2 prompts on
the same sample indices, which is the ceiling the DPI curve should be read
against.

Generations land in `results/<run>/<task>.jsonl`; scoring is a separate step
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
sys.path.insert(0, str(HERE.parent.parent))
from think_prompt import apply_think, strip_think  # noqa: E402
from paths import data_path, resolve_images  # noqa: E402
VTC_ROOT = data_path("RULER_v2_VTC")
TEXT_ROOT = data_path("RULER_v2")

TASKS = [
    "mk_niah_basic", "mk_niah_easy", "mk_niah_medium", "mk_niah_hard",
    "mv_niah_basic", "mv_niah_easy", "mv_niah_medium", "mv_niah_hard",
    "qa_basic", "qa_easy", "qa_medium", "qa_hard",
]
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
    """Parse the normalized page boxes used by the VTC SFT targets."""
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
    if native_thinking and disable_thinking:
        raise ValueError("native_thinking and disable_thinking are mutually exclusive")
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
            f"{row['task']}#{row['index']}: {len(segments) - 1} placeholders "
            f"but {len(images)} images"
        )

    parts = []
    for i, image in enumerate(images):
        seg = segments[i]
        if page_markers:
            # Match the SFT input shape exactly: `Page 1\n<image>`,
            # `Page 2\n<image>`, ... .  apply_think() subsequently strips
            # leading/trailing whitespace from each text part, as the training
            # data loader does.
            seg = f"{seg.rstrip()}\nPage {i + 1}" if seg.strip() else f"Page {i + 1}"
        if seg:
            parts.append({"type": "text", "text": seg})
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


def load_rows(modality, dpi, task, limit):
    if modality == "text":
        path = TEXT_ROOT / task / "test.jsonl"
    else:
        path = VTC_ROOT / f"dpi_{dpi}" / task / "test.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(resolve_images(json.loads(line), path.parent))
            if limit and len(rows) >= limit:
                break
    for r in rows:
        r.setdefault("task", task)
    return rows


async def generate_one(client, sem, args, row, modality):
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
    max_tokens = args.max_tokens
    if reasoning:
        max_tokens = max(max_tokens, args.think_budget)
    extra_body.update(
        {
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
        }
    )
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
                    extra_body=extra_body or None,
                )
            except Exception as exc:  # noqa: BLE001 - surface any client/server error
                last_err = repr(exc)
                await asyncio.sleep(min(2**attempt, 30))
                continue
        if resp is None or not getattr(resp, "choices", None):
            last_err = "empty_response"
            continue
        usage = resp.usage
        initial_prompt_tokens = getattr(resp, "vtc_initial_prompt_tokens", None)
        if initial_prompt_tokens is None:
            extras = getattr(resp, "model_extra", None) or {}
            initial_prompt_tokens = extras.get("vtc_initial_prompt_tokens")
        final_completion_tokens = getattr(resp, "vtc_final_completion_tokens", None)
        if final_completion_tokens is None:
            extras = getattr(resp, "model_extra", None) or {}
            final_completion_tokens = extras.get("vtc_final_completion_tokens")
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
        return {
            "index": row["index"],
            "task": row["task"],
            "generation": strip_think(raw) if reasoning else raw,
            **({"raw_generation": raw} if reasoning else {}),
            **({"reasoning_content": native_reasoning} if native_reasoning else {}),
            "evidence_boxes": extract_evidence_boxes(raw),
            "expected_answer": row["expected_answer"],
            "vision_tokens_est": row.get("vision_tokens"),
            "num_pages": row.get("num_pages", 0),
            "prompt_tokens": usage.prompt_tokens if usage else None,
            **({"initial_prompt_tokens": int(initial_prompt_tokens)}
               if initial_prompt_tokens is not None else {}),
            "completion_tokens": usage.completion_tokens if usage else None,
            **({"final_completion_tokens": int(final_completion_tokens)}
               if final_completion_tokens is not None else {}),
            "max_tokens": max_tokens,
            "finish_reason": resp.choices[0].finish_reason,
            "latency_s": round(time.time() - t0, 2),
        }
    return {
        "index": row["index"],
        "task": row["task"],
        "generation": "",
        "expected_answer": row["expected_answer"],
        "error": last_err,
    }


async def run_split(client, args, modality, dpi, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    for task in args.tasks:
        out_path = out_dir / f"{task}.jsonl"
        done = set()
        if args.resume and out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if not rec.get("error"):
                        done.add(rec["index"])

        rows = [r for r in load_rows(modality, dpi, task, args.limit) if r["index"] not in done]
        if not rows:
            print(f"[{out_dir.name}] {task}: nothing to do", flush=True)
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

            n_img = sum(len(user_parts(r)) for r in rows) if modality != "text" else 0
            print(
                f"[{out_dir.name}] {task}: {len(rows)} prompts built ok"
                f"{f', {n_img} content parts' if n_img else ''}",
                flush=True,
            )
            continue

        tasks_ = [generate_one(client, sem, args, r, modality) for r in rows]
        with open(out_path, "a", encoding="utf-8") as f:
            with tqdm(total=len(tasks_), desc=f"{out_dir.name}/{task}") as bar:
                for coro in asyncio.as_completed(tasks_):
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
        print(f"[{out_dir.name}] {task}: {len(records)} done, {n_err} errors", flush=True)


async def main_async(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0)
    for spec in args.runs:
        if spec == "text":
            await run_split(client, args, "text", None, args.results_dir / "text")
        else:
            dpi = int(spec)
            await run_split(client, args, "image", dpi, args.results_dir / f"dpi_{dpi}")


def main():
    global VTC_ROOT, TEXT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18350/v1")
    ap.add_argument("--model-name", default="FocusVTC")
    ap.add_argument(
        "--runs",
        nargs="+",
        default=["text", "48", "60", "72", "84", "96", "120", "144"],
        help="'text' for the text-only baseline, otherwise a DPI value",
    )
    ap.add_argument("--vtc-root", type=Path, default=VTC_ROOT)
    ap.add_argument("--text-root", type=Path, default=TEXT_ROOT)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--limit", type=int, default=100, help="samples per task")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument(
        "--system-prompt",
        default=SYSTEM_PROMPT,
        help="system message prepended to every request; pass an empty string to disable",
    )
    ap.add_argument(
        "--page-markers",
        action="store_true",
        help="insert `Page N` immediately before each image, matching VTC SFT inputs",
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
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument(
        "--think", action="store_true",
        help="reshape prompts into the VTC SFT training format and prefill an open "
        "<think> turn; the reasoning block is stripped from `generation` and kept "
        "in `raw_generation`",
    )
    ap.add_argument(
        "--strip-think", action="store_true",
        help="the prompt is left alone, but a `<think>...</think>` block the model "
        "emits on its own is moved out of `generation` into `raw_generation`. Use "
        "for models that always reason, e.g. Glyph on its GLM-4.1V backbone",
    )
    ap.add_argument(
        "--native-thinking", action="store_true",
        help="enable Qwen3.5 native thinking without a prefilled <think> prompt",
    )
    ap.add_argument(
        "--disable-thinking", action="store_true",
        help="pass chat_template_kwargs.enable_thinking=false (Qwen3.5 direct-answer mode)",
    )
    ap.add_argument(
        "--think-budget", type=int, default=8192,
        help="minimum max_tokens used by reasoning modes",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument(
        "--config-name",
        default="run_config.json",
        help="config filename inside --results-dir; use a unique name for concurrent workers",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build every prompt and validate placeholder/image alignment without calling the server",
    )
    args = ap.parse_args()
    VTC_ROOT = args.vtc_root.resolve()
    args.vtc_root = VTC_ROOT
    TEXT_ROOT = args.text_root.resolve()
    args.text_root = TEXT_ROOT

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.config_name).name != args.config_name:
        raise SystemExit("--config-name must be a filename, not a path")
    config_path = args.results_dir / args.config_name
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        resume_keys = (
            "vtc_root", "text_root", "model_name", "runs", "tasks", "system_prompt", "page_markers",
            "training_newline", "ground_evidence", "image_prompt_suffix",
            "think", "strip_think", "disable_thinking", "native_thinking",
            "think_budget", "max_tokens", "temperature", "top_p", "top_k",
            "min_p", "presence_penalty", "repetition_penalty",
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
