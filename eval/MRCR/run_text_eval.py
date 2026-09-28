#!/usr/bin/env python3
"""Evaluate the original text-only OpenAI MRCR prompts through vLLM."""

from __future__ import annotations

import argparse
import os
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

SOURCE_ROOT = Path(
    os.environ.get("FOCUSVTC_MRCR_SOURCE", str(Path(__file__).resolve().parents[2] / "datasets/mrcr"))
)
TOKENIZER_PATH = Path(os.environ.get("FOCUSVTC_MODEL", "models/FocusVTC"))
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
DEFAULT_LENGTH_BINS = LENGTH_BINS[:6]
SYSTEM_PROMPT = "You are a helpful assistant"


def build_bin_map(source_root: Path, subset: str) -> dict[tuple[str, int], str]:
    """Recover the official bins from the shuffled 100-row Parquet blocks."""
    blocks = []
    for path in sorted((source_root / subset).glob("*.parquet")):
        lengths = pq.read_table(path, columns=["n_chars"])["n_chars"].to_pylist()
        if len(lengths) % 100:
            raise ValueError(f"{path}: row count is not divisible by 100")
        for start in range(0, len(lengths), 100):
            blocks.append((statistics.median(lengths[start : start + 100]), path.name, start))
    if len(blocks) != len(LENGTH_BINS):
        raise ValueError(f"{subset}: expected 8 length blocks, found {len(blocks)}")
    blocks.sort()
    return {
        (filename, start): length_bin
        for length_bin, (_, filename, start) in zip(LENGTH_BINS, blocks, strict=True)
    }


def iter_rows(args: argparse.Namespace, subset: str) -> Iterator[dict[str, Any]]:
    bin_map = build_bin_map(args.source_root, subset)
    selected_bins = set(args.length_bins)
    index = 0
    selected = 0
    for path in sorted((args.source_root / subset).glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for source_row, batch in enumerate(parquet.iter_batches(batch_size=1)):
            length_bin = bin_map[(path.name, (source_row // 100) * 100)]
            take = (
                length_bin in selected_bins
                and index % args.shard_count == args.shard_index
                and (not args.limit or selected < args.limit)
            )
            if take:
                source = batch.to_pylist()[0]
                yield {
                    "index": index,
                    "_id": f"{subset}-{index:04d}",
                    "dataset": "mrcr",
                    "subset": subset,
                    "length_bin": length_bin,
                    "answers": [source["answer"]],
                    "answer": source["answer"],
                    "random_string_to_prepend": source["random_string_to_prepend"],
                    "n_needles": source["n_needles"],
                    "desired_msg_index": source["desired_msg_index"],
                    "total_messages": source["total_messages"],
                    "n_chars": source["n_chars"],
                    "date_added": source["date_added"],
                    "source_file": f"{subset}/{path.name}",
                    "source_row": source_row,
                    "prompt": source["prompt"],
                    "num_pages": 0,
                    "vision_tokens": 0,
                }
                selected += 1
            index += 1


def build_messages(row: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    messages = json.loads(row["prompt"])
    if len(messages) != row["total_messages"]:
        raise ValueError(
            f"{row['_id']}: total_messages={row['total_messages']}, parsed={len(messages)}"
        )
    if system_prompt and (not messages or messages[0].get("role") != "system"):
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def count_prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(token_ids)


def output_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "prompt"}


async def generate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    row: dict[str, Any],
    messages: list[dict[str, str]],
    prompt_tokens: int,
) -> dict[str, Any]:
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
                    extra_body={
                        "top_k": args.top_k,
                        "min_p": args.min_p,
                        "repetition_penalty": args.repetition_penalty,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
            else:
                if response is None or not getattr(response, "choices", None):
                    last_error = "empty_response"
                    continue
                usage = response.usage
                return {
                    **output_fields(row),
                    "generation": (response.choices[0].message.content or "").strip(),
                    "prompt_tokens_est": prompt_tokens,
                    "vision_tokens_model_est": 0,
                    "prompt_tokens": usage.prompt_tokens if usage else prompt_tokens,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "finish_reason": response.choices[0].finish_reason,
                    "latency_s": round(time.time() - started, 2),
                }
        if attempt < args.retries:
            await asyncio.sleep(min(2**attempt, 30))
    return {
        **output_fields(row),
        "generation": "",
        "error": last_error,
        "prompt_tokens_est": prompt_tokens,
        "vision_tokens_model_est": 0,
    }


def read_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            records[int(record["index"])] = record
    return records


def compact(path: Path, records: dict[int, dict[str, Any]]) -> None:
    order = {name: index for index, name in enumerate(LENGTH_BINS)}
    rows = sorted(records.values(), key=lambda row: (order[row["length_bin"]], row["index"]))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def drain_completed(
    pending: set[asyncio.Task],
    handle: Any,
    existing: dict[int, dict[str, Any]],
    bar: tqdm,
    wait_all: bool = False,
) -> set[asyncio.Task]:
    if not pending:
        return pending
    done, still_pending = await asyncio.wait(
        pending,
        return_when=asyncio.ALL_COMPLETED if wait_all else asyncio.FIRST_COMPLETED,
    )
    for task in done:
        record = task.result()
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        existing[record["index"]] = record
        bar.update(1)
    return set(still_pending)


async def run_subset(
    client: AsyncOpenAI,
    tokenizer: Any,
    args: argparse.Namespace,
    subset: str,
) -> None:
    out_path = args.results_dir / f"{subset}.jsonl"
    existing = read_existing(out_path) if args.resume else {}
    done = {index for index, record in existing.items() if not record.get("error")}
    total = (100 * len(args.length_bins) + args.shard_count - 1) // args.shard_count
    if args.limit:
        total = min(total, args.limit)
    initial = min(len(done), total)
    semaphore = asyncio.Semaphore(args.concurrency)
    pending: set[asyncio.Task] = set()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle, tqdm(
        total=total, initial=initial, desc=subset
    ) as bar:
        for row in iter_rows(args, subset):
            if row["index"] in done:
                continue
            try:
                messages = build_messages(row, args.system_prompt)
                prompt_tokens = await asyncio.to_thread(
                    count_prompt_tokens, tokenizer, messages
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    **output_fields(row),
                    "generation": "",
                    "error": f"prompt_build_error: {exc!r}",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                existing[row["index"]] = record
                bar.update(1)
                continue

            if prompt_tokens + args.max_tokens > args.context_limit:
                record = {
                    **output_fields(row),
                    "generation": "",
                    "skipped_reason": (
                        f"prompt {prompt_tokens} + max_tokens {args.max_tokens} "
                        f"exceeds native context {args.context_limit}"
                    ),
                    "prompt_tokens_est": prompt_tokens,
                    "vision_tokens_model_est": 0,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                existing[row["index"]] = record
                bar.update(1)
                continue

            pending.add(
                asyncio.create_task(
                    generate_one(client, semaphore, args, row, messages, prompt_tokens)
                )
            )
            if len(pending) >= 2 * args.concurrency:
                pending = await drain_completed(pending, handle, existing, bar)

        while pending:
            pending = await drain_completed(
                pending, handle, existing, bar, wait_all=True
            )

    compact(out_path, existing)
    selected = list(existing.values())
    errors = sum(bool(record.get("error")) for record in selected)
    skipped = sum(bool(record.get("skipped_reason")) for record in selected)
    print(
        f"[{subset}] selected={len(selected)} generated={len(selected)-errors-skipped} "
        f"skipped={skipped} errors={errors}",
        flush=True,
    )


async def dry_run(tokenizer: Any, args: argparse.Namespace) -> None:
    for subset in args.subsets:
        counts = []
        for row in iter_rows(args, subset):
            messages = build_messages(row, args.system_prompt)
            counts.append(
                await asyncio.to_thread(count_prompt_tokens, tokenizer, messages)
            )
        print(f"[{subset}] dry-run prompt_tokens={counts}", flush=True)


async def main_async(tokenizer: Any, args: argparse.Namespace) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    for subset in args.subsets:
        await run_subset(client, tokenizer, args, subset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18115/v1")
    parser.add_argument("--model-name", default="Qwen3.5-9B-text-only")
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument(
        "--length-bins", nargs="+", choices=LENGTH_BINS,
        default=list(DEFAULT_LENGTH_BINS),
        help="length bins to run (default: through 128K-256K)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--context-limit", type=int, default=262144)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard count/index")
    if not args.source_root.is_dir():
        parser.error(f"source dataset not found: {args.source_root}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.dry_run:
        asyncio.run(dry_run(tokenizer, args))
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.results_dir / "run_config.json"
    keys = (
        "model_name", "tokenizer_path", "source_root", "subsets", "length_bins",
        "limit", "shard_count", "shard_index", "max_tokens", "context_limit",
        "system_prompt", "temperature", "top_p", "top_k", "min_p",
        "presence_penalty", "repetition_penalty",
    )
    config = {
        key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
        for key in vars(args)
    }
    config["input_mode"] = "text"
    config["page_markers"] = False
    config["disable_thinking"] = True
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        changed = [key for key in keys if previous.get(key) != config.get(key)]
        if changed:
            raise SystemExit("refusing to mix incompatible results; changed: " + ", ".join(changed))
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    asyncio.run(main_async(tokenizer, args))
    print("ALL_GENERATION_DONE", flush=True)


if __name__ == "__main__":
    main()
