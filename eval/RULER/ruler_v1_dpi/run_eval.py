#!/usr/bin/env python3
"""Run Qwen3-VL-8B over the RULER v1 8k DPI ablation via a vLLM OpenAI server.

For each DPI the sample's rendered pages are sent as `file://` image parts
followed by the question text (the server must run with
--allowed-local-media-path). RULER's `answer_prefix` is replayed as an assistant
prefill so the model continues in the expected format, which is what the
upstream completion-style eval does.

`--runs text` instead evaluates the original text-only RULER v1 prompts on the
same sample ids, which is the ceiling the DPI curve should be read against.

Generations land in `results/<run>/<task>.jsonl`; scoring is a separate step
(`score.py`) so a crashed run can be resumed without re-generating.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from think_prompt import apply_think, strip_think  # noqa: E402
from paths import data_path, resolve_images  # noqa: E402

VTC_ROOT = data_path("RULER_v1_VTC")
TEXT_SRC = data_path("RULER_v1/ruler_8192.jsonl")

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe",
    "qa_1", "qa_2",
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
COUNT_PROMPT_CWE = (
    "This is a frequency-aggregation task, not a localization task. Ignore any "
    "worked example. Do not transcribe the list, enumerate occurrence positions, "
    "or use page numbers or bounding boxes. Output exactly 10 unique comma-separated "
    "words and nothing else."
)
COUNT_PROMPT_GENERIC = (
    "This is a counting or aggregation task, not a localization task. Do not output "
    "page numbers or bounding boxes. Answer only what the question asks, concisely, "
    "without explanations or occurrence positions."
)


def build_messages(
    row,
    modality,
    prefix_mode,
    page_markers,
    training_newline=False,
    think=False,
    enable_thinking=False,
    disable_thinking=False,
    system_prompt=SYSTEM_PROMPT,
    ground_evidence=False,
    user_prompt_suffix=None,
    count_prompt=False,
):
    """Return (messages, extra_body) for one sample.

    The image runs put the rendered context in place of the text context; the
    task instruction line lives inside page 1, exactly as RULER wrote it.

    Only one assistant turn can be left open, so `--think` takes it and the
    caller is expected to have forced `prefix_mode` to "user".
    """
    question = row["question"]
    prefix = row.get("answer_prefix") or ""

    if modality == "text":
        user_content = row["context"] + "\n" + question
    else:
        parts = []
        for i, path in enumerate(row["image"], start=1):
            if page_markers:
                parts.append({"type": "text", "text": f"Page {i}"})
            parts.append({"type": "image_url", "image_url": {"url": f"file://{path}"}})
            if training_newline:
                # Match the VTC-SFT prompt seen during
                # checkpoint: every image placeholder is followed by a literal
                # newline before the next page marker/question.
                parts.append({"type": "text", "text": "\n"})
        parts.append({"type": "text", "text": "\n" + question})
        user_content = parts

    if prefix_mode == "user" and prefix:
        tail = "\n" + prefix
        if isinstance(user_content, str):
            user_content += tail
        else:
            user_content[-1]["text"] += tail

    # Keep this instruction literally last in image prompts, including when
    # RULER's completion-style answer prefix has been moved into the user turn.
    if modality != "text" and ground_evidence:
        tail = "\n" + GROUNDING_INSTRUCTION
        user_content[-1]["text"] += tail

    if user_prompt_suffix:
        tail = "\n" + user_prompt_suffix
        if isinstance(user_content, str):
            user_content += tail
        else:
            user_content[-1]["text"] += tail

    if count_prompt:
        suffix = COUNT_PROMPT_CWE if row.get("task") == "cwe" else COUNT_PROMPT_GENERIC
        tail = "\n" + suffix
        if isinstance(user_content, str):
            user_content += tail
        else:
            user_content[-1]["text"] += tail

    messages = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    messages.append({"role": "user", "content": user_content})
    if enable_thinking and disable_thinking:
        raise ValueError("enable_thinking and disable_thinking are mutually exclusive")
    extra_body = {}
    if enable_thinking or disable_thinking:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": bool(enable_thinking),
        }
    if think:
        return apply_think(messages, extra_body)
    if prefix_mode == "prefill" and prefix:
        messages.append({"role": "assistant", "content": prefix})
        # vLLM: keep the assistant turn open instead of starting a new one.
        extra_body.update(
            {"continue_final_message": True, "add_generation_prompt": False}
        )
    return messages, extra_body


def load_rows(modality, dpi, limit, vtc_root=VTC_ROOT):
    """Return {task: [row, ...]}, ids matching render_ruler_v1.py's numbering."""
    by_task = defaultdict(list)
    if modality == "text":
        with open(TEXT_SRC, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                task = d["task"]
                if limit and len(by_task[task]) >= limit:
                    continue
                d["id"] = f"{task}_{len(by_task[task]):04d}"
                by_task[task].append(d)
    else:
        path = vtc_root / f"dpi_{dpi}" / "index.jsonl"
        rows = [resolve_images(json.loads(l), path.parent) for l in open(path, encoding="utf-8")]
        rows.sort(key=lambda r: r["id"])
        for r in rows:
            task = r["task"]
            if limit and len(by_task[task]) >= limit:
                continue
            by_task[task].append(r)
    return by_task


async def generate_one(client, sem, args, row, modality):
    messages, extra_body = build_messages(
        row, modality, args.answer_prefix_mode, args.page_markers,
        args.training_newline, args.think,
        args.enable_thinking, args.disable_thinking, system_prompt=args.system_prompt,
        ground_evidence=args.ground_evidence,
        user_prompt_suffix=args.user_prompt_suffix,
        count_prompt=args.count_prompt,
    )
    reasoning = args.think or args.strip_think or args.enable_thinking
    max_tokens = args.max_tokens or int(row.get("max_new_tokens") or 128)
    if reasoning:
        # RULER's 32-128 token budgets leave no room for a reasoning block.
        max_tokens = max(max_tokens, args.think_budget)
    # top_k, min_p, and repetition_penalty are vLLM extensions rather than
    # fields in the OpenAI Chat Completions schema. Keep them in extra_body;
    # top_p and presence_penalty use their standard OpenAI request fields.
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
        usage = resp.usage
        # The tool gateway returns the first-turn length separately because
        # usage.prompt_tokens is cumulative across all backend calls in a
        # multi-turn zoom trajectory.
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
        # With vLLM `--reasoning-parser qwen3`, native thinking is returned
        # separately from the final answer. Keep both in the artifact.
        native_reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        raw = content
        if native_reasoning:
            raw = f"<think>\n{native_reasoning}\n</think>\n\n{content}"
        return {
            "id": row["id"],
            "task": row["task"],
            "generation": strip_think(raw) if reasoning else raw,
            **({"raw_generation": raw} if reasoning else {}),
            **({"reasoning_content": native_reasoning} if native_reasoning else {}),
            "expected_answer": row["answer"],
            "answer_prefix": row.get("answer_prefix"),
            "vision_tokens_est": row.get("vision_tokens_qwen3vl"),
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
        "id": row["id"],
        "task": row["task"],
        "generation": "",
        "expected_answer": row["answer"],
        "error": last_err,
    }


async def run_split(client, args, modality, dpi, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_task = load_rows(modality, dpi, args.limit, args.vtc_root)
    sem = asyncio.Semaphore(args.concurrency)

    for task in args.tasks:
        out_path = out_dir / f"{task}.jsonl"
        done = set()
        if args.resume and out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if not rec.get("error"):
                        done.add(rec["id"])

        rows = [r for r in by_task.get(task, []) if r["id"] not in done]
        if not rows:
            print(f"[{out_dir.name}] {task}: nothing to do", flush=True)
            continue

        if args.dry_run:
            missing = 0
            n_parts = 0
            for r in rows:
                msgs, _ = build_messages(
                    r, modality, args.answer_prefix_mode, args.page_markers,
                    args.training_newline, args.think,
                    args.enable_thinking, args.disable_thinking, system_prompt=args.system_prompt,
                    ground_evidence=args.ground_evidence,
                    user_prompt_suffix=args.user_prompt_suffix,
                    count_prompt=args.count_prompt,
                )
                content = next(m for m in msgs if m["role"] == "user")["content"]
                n_parts += 1 if isinstance(content, str) else len(content)
                for p in r.get("image", []):
                    if not Path(p).exists():
                        missing += 1
            flag = f", {missing} MISSING IMAGES" if missing else ""
            print(
                f"[{out_dir.name}] {task}: {len(rows)} prompts built ok, "
                f"{n_parts} content parts{flag}",
                flush=True,
            )
            if missing:
                raise SystemExit(f"{missing} image files referenced but not on disk")
            continue

        tasks_ = [generate_one(client, sem, args, r, modality) for r in rows]
        with open(out_path, "a", encoding="utf-8") as f:
            with tqdm(total=len(tasks_), desc=f"{out_dir.name}/{task}") as bar:
                for coro in asyncio.as_completed(tasks_):
                    rec = await coro
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    bar.update(1)

        by_id = {}
        for line in open(out_path, encoding="utf-8"):
            rec = json.loads(line)
            by_id[rec["id"]] = rec
        records = [by_id[k] for k in sorted(by_id)]
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
    global TEXT_SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18350/v1")
    ap.add_argument("--model-name", default="FocusVTC")
    ap.add_argument(
        "--runs",
        nargs="+",
        default=["text", "48", "60", "72", "84", "96", "120", "144"],
        help="'text' for the text-only baseline, otherwise a DPI value",
    )
    ap.add_argument("--text-source", type=Path, default=TEXT_SRC)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--limit", type=int, default=100, help="samples per task")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument(
        "--vtc-root", type=Path, default=VTC_ROOT,
        help="rendered RULER image dataset root (must contain dpi_N/index.jsonl)",
    )
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument(
        "--system-prompt",
        default=SYSTEM_PROMPT,
        help="system message prepended to every request; pass an empty string to disable",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=None,
        help="override the per-task RULER budget (128/120/50/32/30); leave unset "
        "to stay comparable with the upstream numbers",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument(
        "--top-k", type=int, default=-1,
        help="vLLM sampling parameter; -1 disables top-k filtering",
    )
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument(
        "--answer-prefix-mode", choices=("prefill", "user", "none"), default="prefill",
        help="prefill: replay answer_prefix as an open assistant turn (faithful to "
        "upstream); user: append it to the user turn; none: drop it",
    )
    ap.add_argument(
        "--page-markers", action="store_true",
        help="insert a 'Page N' text part before each image (off by default, "
        "matching the VTC_SFT prompt format)",
    )
    ap.add_argument(
        "--training-newline", action="store_true",
        help="insert a literal newline text part after every image; use to "
        "reproduce the legacy VTC-SFT page/image prompt layout",
    )
    ap.add_argument(
        "--ground-evidence", action="store_true",
        help="append the page-aware grounding instruction to image prompts; it is "
        "not added to the text-only run",
    )
    ap.add_argument(
        "--user-prompt-suffix",
        default=None,
        help="append a task-specific instruction to the end of every user prompt",
    )
    ap.add_argument(
        "--count-prompt", action="store_true",
        help="append the established non-localization counting prompt (CWE exact-10 variant; generic for FWE/VT)",
    )
    ap.add_argument(
        "--think", action="store_true",
        help="reshape prompts into the VTC SFT training format and prefill an open "
        "<think> turn; forces --answer-prefix-mode user since only one assistant "
        "turn can stay open",
    )
    ap.add_argument(
        "--strip-think", action="store_true",
        help="the prompt is left alone, but a `<think>...</think>` block the model "
        "emits on its own is moved out of `generation` into `raw_generation`. Use "
        "for models that always reason, e.g. Glyph on its GLM-4.1V backbone",
    )
    thinking_group = ap.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking", action="store_true",
        help="pass chat_template_kwargs.enable_thinking=true (Qwen3.5 native thinking; no manual <think> prefill)",
    )
    thinking_group.add_argument(
        "--disable-thinking", action="store_true",
        help="pass chat_template_kwargs.enable_thinking=false (Qwen3.5 direct-answer mode)",
    )
    ap.add_argument(
        "--think-budget", type=int, default=2560,
        help="floor on max_tokens under --think, --strip-think, or --enable-thinking. The enumeration tasks (vt/cwe/fwe) "
        "walk every match with its own page coordinates, so 1024 truncates them "
        "mid-reasoning and no answer is ever emitted",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build every prompt and check every image path without calling the server",
    )
    ap.add_argument(
        "--config-name",
        default="run_config.json",
        help="config filename under --results-dir; use one name per parallel worker",
    )
    args = ap.parse_args()
    TEXT_SRC = args.text_source.resolve()
    args.text_source = TEXT_SRC
    args.vtc_root = args.vtc_root.resolve()

    if args.enable_thinking and args.think:
        raise SystemExit("--enable-thinking is native thinking; do not combine it with --think prefill")
    if args.think and args.answer_prefix_mode == "prefill":
        args.answer_prefix_mode = "user"
        print("[config] --think takes the assistant turn: answer_prefix_mode -> user", flush=True)
    if args.enable_thinking and args.answer_prefix_mode == "prefill":
        args.answer_prefix_mode = "none"
        print("[config] native thinking uses the chat-template generation prompt: answer_prefix_mode -> none", flush=True)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.results_dir / args.config_name
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != (str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key))
               for key in ("system_prompt", "model_name", "vtc_root", "text_source")):
            raise SystemExit(
                "Refusing to resume results generated with a different or unrecorded "
                "system prompt, model, or data source. Use a new --results-dir or pass --no-resume for a "
                "full regeneration."
            )
    with open(config_path, "w", encoding="utf-8") as f:
        cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        cfg["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(cfg, f, indent=2)

    asyncio.run(main_async(args))
    print("ALL_GENERATION_DONE")


if __name__ == "__main__":
    sys.exit(main())
