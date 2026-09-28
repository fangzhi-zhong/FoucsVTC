#!/usr/bin/env python3
"""Evaluate page-rendered OpenAI MRCR through a vLLM OpenAI endpoint.

Every request has a system turn and an explicit ``Page N`` text part before
each image.  Records outside a checkpoint's native context/image ceiling are
written as structured skips instead of silently truncating the conversation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from think_prompt import apply_think, strip_think  # noqa: E402
from paths import data_path, resolve_images  # noqa: E402

DATA_ROOT = data_path("MRCR_VTC/dpi_72")
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
# Default experiments stop at the official 128K-256K text-length bin.
# Larger bins remain available through an explicit --length-bins override.
DEFAULT_LENGTH_BINS = LENGTH_BINS[:6]
GATEWAY_TRACE_FIELDS = {
    "vtc_model_outputs": "agent_trajectory",
    "vtc_tool_trace": "tool_trace",
    "vtc_raw_generation": "raw_generation",
    "vtc_initial_prompt_tokens": "initial_prompt_tokens",
    "vtc_cumulative_usage": "cumulative_usage",
    "vtc_final_completion_tokens": "final_completion_tokens",
    "vtc_trajectory_tokens": "trajectory_tokens",
    "vtc_stop_reason": "stop_reason",
    "vtc_limits": "agent_limits",
}
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

# (vision merge factor in pixels, minimum pixels, maximum pixels)
VISION_PROFILES = {
    "qwen": (32, 65_536, 16_777_216),
    "glm": (28, 12_544, 9_633_792),
}


def smart_resize(
    width: int,
    height: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Match the Qwen/GLM image processors' resize arithmetic."""
    resized_h = max(factor, round(height / factor) * factor)
    resized_w = max(factor, round(width / factor) * factor)
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_h = max(factor, math.floor(height / beta / factor) * factor)
        resized_w = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = math.ceil(height * beta / factor) * factor
        resized_w = math.ceil(width * beta / factor) * factor
    return resized_w, resized_h


def estimate_vision_tokens(row: dict[str, Any], profile: str) -> int:
    if profile == "qwen" and row.get("vision_tokens") is not None:
        return int(row["vision_tokens"])
    factor, min_pixels, max_pixels = VISION_PROFILES[profile]
    total = 0
    for width, height in row["image_sizes"]:
        width, height = smart_resize(
            width, height, factor, min_pixels, max_pixels
        )
        total += width * height // (factor * factor)
    return total


def estimate_prompt_tokens(
    row: dict[str, Any],
    profile: str,
    extra_text: str = "",
    training_newline: bool = False,
) -> tuple[int, int]:
    """Conservative estimate used only for pre-request context filtering."""
    vision_tokens = estimate_vision_tokens(row, profile)
    page_overhead = (9 if training_newline else 8) * int(row["num_pages"])
    text = row["question"].replace("<image>", "") + extra_text
    text_overhead = len(text) // 3 + 256
    return vision_tokens + page_overhead + text_overhead, vision_tokens


def build_messages(
    row: dict[str, Any],
    system_prompt: str,
    disable_thinking: bool,
    think: bool = False,
    native_thinking: bool = False,
    training_newline: bool = False,
    ground_evidence: bool = False,
    user_prompt_suffix: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = row["images"]
    segments = row["question"].split("<image>")
    if len(segments) - 1 != len(images):
        raise ValueError(
            f"{row['subset']}#{row['index']}: {len(segments) - 1} image "
            f"placeholders but {len(images)} paths"
        )

    content: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if segment:
            content.append({"type": "text", "text": segment})
        if index < len(images):
            # Keep the marker as its own text part, matching the VTC SFT format.
            content.append({"type": "text", "text": f"Page {index + 1}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{images[index]}"},
                }
            )
            if training_newline:
                content.append({"type": "text", "text": "\n"})

    if ground_evidence:
        content.append({"type": "text", "text": "\n" + GROUNDING_INSTRUCTION})
    if user_prompt_suffix:
        content.append({"type": "text", "text": "\n" + user_prompt_suffix})

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if disable_thinking
        else ({"chat_template_kwargs": {"enable_thinking": True}} if native_thinking else {})
    )
    return apply_think(messages, extra_body) if think else (messages, extra_body)


def common_record(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "index",
        "_id",
        "dataset",
        "subset",
        "length_bin",
        "answers",
        "answer",
        "random_string_to_prepend",
        "n_needles",
        "desired_msg_index",
        "total_messages",
        "num_pages",
        "vision_tokens",
        "source_file",
        "source_row",
    )
    return {key: row.get(key) for key in keys}


def load_rows(args: argparse.Namespace, subset: str) -> list[dict[str, Any]]:
    path = args.data_root / subset / "test.jsonl"
    selected = set(args.length_bins)
    rows = [
        resolve_images(json.loads(line), path.parent)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]
    rows = [row for row in rows if row["length_bin"] in selected]
    order = {name: index for index, name in enumerate(LENGTH_BINS)}
    rows.sort(key=lambda row: (order[row["length_bin"]], row["index"]))
    rows = rows[args.shard_index :: args.shard_count]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def skip_reason(
    row: dict[str, Any], args: argparse.Namespace
) -> tuple[str | None, int, int]:
    extra_text = ""
    if args.ground_evidence:
        extra_text += "\n" + GROUNDING_INSTRUCTION
    if args.user_prompt_suffix:
        extra_text += "\n" + args.user_prompt_suffix
    prompt_tokens, vision_tokens = estimate_prompt_tokens(
        row, args.vision_profile, extra_text, args.training_newline
    )
    if args.max_images and row["num_pages"] > args.max_images:
        return (
            f"num_pages={row['num_pages']} exceeds server max_images={args.max_images}",
            prompt_tokens,
            vision_tokens,
        )
    if args.context_limit and prompt_tokens + args.max_tokens > args.context_limit:
        return (
            f"estimated prompt {prompt_tokens} + max_tokens {args.max_tokens} "
            f"exceeds native context {args.context_limit}",
            prompt_tokens,
            vision_tokens,
        )
    return None, prompt_tokens, vision_tokens


async def generate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    row: dict[str, Any],
    prompt_tokens_est: int,
    vision_tokens_est: int,
) -> dict[str, Any]:
    try:
        messages, extra_body = build_messages(
            row,
            args.system_prompt,
            args.disable_thinking,
            args.think,
            args.native_thinking,
            training_newline=args.training_newline,
            ground_evidence=args.ground_evidence,
            user_prompt_suffix=args.user_prompt_suffix,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            **common_record(row),
            "generation": "",
            "error": f"prompt_build_error: {exc!r}",
            "prompt_tokens_est": prompt_tokens_est,
            "vision_tokens_model_est": vision_tokens_est,
        }

    last_error = None
    for attempt in range(args.retries + 1):
        async with semaphore:
            started = time.time()
            try:
                response = await client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                    max_tokens=args.max_tokens,
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
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
            else:
                if response is None or not getattr(response, "choices", None):
                    last_error = "empty_response"
                    continue
                message = response.choices[0].message
                content = message.content or ""
                native_reasoning = (
                    getattr(message, "reasoning_content", None)
                    or getattr(message, "reasoning", None)
                    or ""
                )
                raw = content
                if native_reasoning:
                    raw = f"<think>\n{native_reasoning}\n</think>\n\n{content}"
                reasoning = args.think or args.strip_think or args.native_thinking
                generation = strip_think(raw) if reasoning else raw.strip()
                usage = response.usage
                record = {
                    **common_record(row),
                    "input_messages": messages,
                    "generation": generation,
                    "raw_generation": raw,
                    **(
                        {"reasoning_content": native_reasoning}
                        if native_reasoning
                        else {}
                    ),
                    "prompt_tokens_est": prompt_tokens_est,
                    "vision_tokens_model_est": vision_tokens_est,
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "finish_reason": response.choices[0].finish_reason,
                    "latency_s": round(time.time() - started, 2),
                }
                # The MRCR client normally sees an OpenAI ChatCompletion,
                # while the VTC gateway adds the complete multi-turn agent
                # trace as extra response fields.  Preserve those fields so
                # a scored row can be audited without rerunning inference.
                extras = getattr(response, "model_extra", None) or {}
                if not extras and hasattr(response, "model_dump"):
                    extras = response.model_dump()
                for key, output_key in GATEWAY_TRACE_FIELDS.items():
                    value = getattr(response, key, None)
                    if value is None:
                        value = extras.get(key)
                    if value is not None:
                        record[output_key] = value
                return record
        if attempt < args.retries:
            await asyncio.sleep(min(2**attempt, 30))

    return {
        **common_record(row),
        "generation": "",
        "error": last_error,
        "prompt_tokens_est": prompt_tokens_est,
        "vision_tokens_model_est": vision_tokens_est,
    }


def read_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            records[int(record["index"])] = record
    return records


def compact(path: Path, records: dict[int, dict[str, Any]]) -> None:
    order = {name: index for index, name in enumerate(LENGTH_BINS)}
    sorted_records = sorted(
        records.values(),
        key=lambda record: (order[record["length_bin"]], record["index"]),
    )
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_subset(
    client: AsyncOpenAI, args: argparse.Namespace, subset: str
) -> None:
    out_path = args.results_dir / f"{subset}.jsonl"
    existing = read_existing(out_path) if args.resume else {}
    rows = load_rows(args, subset)
    selected_indices = {row["index"] for row in rows}

    # Successful and deterministic over-context records are done. Runtime
    # errors are retried when a run is resumed.
    done = {
        index
        for index, record in existing.items()
        if index in selected_indices and not record.get("error")
    }
    pending = [row for row in rows if row["index"] not in done]
    skipped: list[dict[str, Any]] = []
    runnable: list[tuple[dict[str, Any], int, int]] = []
    for row in pending:
        reason, prompt_tokens_est, vision_tokens_est = skip_reason(row, args)
        if reason:
            skipped.append(
                {
                    **common_record(row),
                    "generation": "",
                    "skipped_reason": reason,
                    "prompt_tokens_est": prompt_tokens_est,
                    "vision_tokens_model_est": vision_tokens_est,
                }
            )
        else:
            runnable.append((row, prompt_tokens_est, vision_tokens_est))

    if args.dry_run:
        for row, _, _ in runnable:
            build_messages(
                row,
                args.system_prompt,
                args.disable_thinking,
                args.think,
                args.native_thinking,
                training_newline=args.training_newline,
                ground_evidence=args.ground_evidence,
                user_prompt_suffix=args.user_prompt_suffix,
            )
            for image in row["images"]:
                if not Path(image).is_file():
                    raise FileNotFoundError(image)
        print(
            f"[{subset}] selected={len(rows)} runnable={len(runnable)} "
            f"skipped={len(skipped)} prompts/images valid",
            flush=True,
        )
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for record in skipped:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing[record["index"]] = record
        handle.flush()

        semaphore = asyncio.Semaphore(args.concurrency)
        jobs = [
            generate_one(client, semaphore, args, row, prompt_est, vision_est)
            for row, prompt_est, vision_est in runnable
        ]
        with tqdm(total=len(jobs), desc=subset) as bar:
            for job in asyncio.as_completed(jobs):
                record = await job
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                existing[record["index"]] = record
                bar.update(1)

    compact(out_path, existing)
    selected = [existing[index] for index in selected_indices if index in existing]
    errors = sum(bool(record.get("error")) for record in selected)
    n_skipped = sum(bool(record.get("skipped_reason")) for record in selected)
    print(
        f"[{subset}] selected={len(rows)} generated={len(rows) - errors - n_skipped} "
        f"skipped={n_skipped} errors={errors}",
        flush=True,
    )


def validate_outputs(args: argparse.Namespace) -> None:
    problems = []
    count = 0
    for subset in args.subsets:
        records = read_existing(args.results_dir / f"{subset}.jsonl")
        wanted = {row["index"] for row in load_rows(args, subset)}
        for index in wanted:
            count += 1
            record = records.get(index)
            if record is None:
                problems.append(f"{subset}#{index}: missing output")
            elif record.get("error"):
                problems.append(f"{subset}#{index}: {record['error']}")
            elif record.get("skipped_reason"):
                problems.append(f"{subset}#{index}: {record['skipped_reason']}")
            elif not record.get("generation"):
                problems.append(f"{subset}#{index}: empty generation")
    if problems:
        raise SystemExit("generation validation failed:\n" + "\n".join(problems))
    print(f"ALL_SELECTED_OUTPUTS_VALID ({count} generations)", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0)
    for subset in args.subsets:
        await run_subset(client, args, subset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18350/v1")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument(
        "--length-bins", nargs="+", choices=LENGTH_BINS,
        default=list(DEFAULT_LENGTH_BINS),
        help="length bins to run (default: through 128K-256K)",
    )
    parser.add_argument("--limit", type=int, default=0, help="samples per subset")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--context-limit", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--vision-profile", choices=VISION_PROFILES, required=True)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--strip-think", action="store_true")
    parser.add_argument(
        "--native-thinking", action="store_true",
        help="enable Qwen3.5 native thinking without prefilled <think> prompt",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="use the VTC SFT prompt shape, prefill an open <think> turn, and "
        "strip the reasoning block before scoring",
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--training-newline",
        action="store_true",
        help="append a literal newline after every image, matching VTC training",
    )
    parser.add_argument(
        "--ground-evidence",
        action="store_true",
        help="append the RULER v1 page-aware grounding instruction",
    )
    parser.add_argument(
        "--user-prompt-suffix",
        help="append an additional instruction after the grounding prompt",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="require a nonempty generation for every selected record",
    )
    args = parser.parse_args()

    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < count")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.results_dir / "run_config.json"
    config_keys = (
        "model_name",
        "data_root",
        "subsets",
        "length_bins",
        "limit",
        "shard_count",
        "shard_index",
        "max_tokens",
        "context_limit",
        "max_images",
        "vision_profile",
        "system_prompt",
        "think",
        "strip_think",
        "disable_thinking",
        "native_thinking",
        "training_newline",
        "ground_evidence",
        "user_prompt_suffix",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
    )
    current = {
        key: str(getattr(args, key))
        if isinstance(getattr(args, key), Path)
        else getattr(args, key)
        for key in vars(args)
    }
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        changed = [key for key in config_keys if previous.get(key) != current.get(key)]
        if changed:
            raise SystemExit(
                "refusing to mix incompatible resumed results; changed: "
                + ", ".join(changed)
            )
    current["output_schema_version"] = 2
    current["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config_path.write_text(
        json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    asyncio.run(main_async(args))
    if args.fail_on_error and not args.dry_run:
        validate_outputs(args)
    print("ALL_GENERATION_DONE", flush=True)


if __name__ == "__main__":
    main()
