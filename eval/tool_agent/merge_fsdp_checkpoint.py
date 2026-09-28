#!/usr/bin/env python3
"""Merge this checkout's rank-local FSDP DTensor files into HF safetensors.

The GRPO checkpoint stores every model tensor as a dim-0 DTensor, one local
shard per rank.  This script streams parameters into bounded safetensor files
and copies tokenizer/config assets from the SFT starting checkpoint.  Floating
point tensors are saved as bfloat16 to match inference precision.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


def discover(actor: Path) -> list[Path]:
    found: dict[int, Path] = {}
    world_size = None
    for path in actor.glob("model_world_size_*_rank_*.pt"):
        match = SHARD_RE.match(path.name)
        if not match:
            continue
        size, rank = map(int, match.groups())
        world_size = size if world_size is None else world_size
        if size != world_size:
            raise ValueError("mixed world sizes in actor checkpoint")
        found[rank] = path
    if world_size is None or sorted(found) != list(range(world_size)):
        raise FileNotFoundError(f"incomplete FSDP model shards under {actor}")
    return [found[rank] for rank in range(world_size)]


def copy_hf_assets(base_model: Path, target: Path) -> None:
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    for source in base_model.iterdir():
        if not source.is_file() or source.suffix in weight_suffixes:
            continue
        if source.name.endswith(".index.json"):
            continue
        shutil.copy2(source, target / source.name)


def local_tensor(value) -> torch.Tensor:
    if not hasattr(value, "to_local") or not hasattr(value, "placements"):
        raise TypeError(f"expected DTensor, got {type(value)!r}")
    placements = value.placements
    if len(placements) != 1 or not placements[0].is_shard() or placements[0].dim != 0:
        raise ValueError(f"only one-dimensional dim-0 FSDP shards are supported: {placements}")
    return value.to_local()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--max-shard-size-gb", type=float, default=4.0)
    args = parser.parse_args()

    shard_paths = discover(args.actor)
    if not (args.base_model / "config.json").is_file():
        raise FileNotFoundError(f"base model config not found: {args.base_model / 'config.json'}")
    if args.target.exists() and any(args.target.iterdir()):
        raise FileExistsError(f"target must not exist or must be empty: {args.target}")

    args.target.mkdir(parents=True, exist_ok=True)
    states = [torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in shard_paths]
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise ValueError("rank model state dicts have different key order")

    limit = int(args.max_shard_size_gb * 1024**3)
    buffered: dict[str, torch.Tensor] = {}
    buffered_bytes = 0
    parts: list[Path] = []
    weight_map: dict[str, str] = {}
    total_size = 0

    def flush() -> None:
        nonlocal buffered, buffered_bytes
        if not buffered:
            return
        part = args.target / f".model-part-{len(parts) + 1:05d}.safetensors"
        save_file(buffered, part, metadata={"format": "pt"})
        parts.append(part)
        buffered = {}
        buffered_bytes = 0

    for index, key in enumerate(keys, start=1):
        first = states[0][key]
        shape = tuple(first.shape)
        pieces = []
        for state in states:
            local = local_tensor(state[key])
            pieces.append(local.to(torch.bfloat16) if local.is_floating_point() else local)
        tensor = torch.cat(pieces, dim=0)[: shape[0]].contiguous()
        if tuple(tensor.shape) != shape:
            raise ValueError(f"merged shape mismatch for {key}: {tuple(tensor.shape)} != {shape}")
        size = tensor.numel() * tensor.element_size()
        if buffered and buffered_bytes + size > limit:
            flush()
        buffered[key] = tensor
        buffered_bytes += size
        total_size += size
        if index % 50 == 0 or index == len(keys):
            print(f"merged {index}/{len(keys)} tensors", flush=True)
    flush()

    count = len(parts)
    for index, part in enumerate(parts, start=1):
        final_name = f"model-{index:05d}-of-{count:05d}.safetensors"
        final = args.target / final_name
        part.rename(final)
        with safe_open(final, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = final_name

    copy_hf_assets(args.base_model, args.target)
    config_path = args.target / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config["torch_dtype"] = "bfloat16"
    config_path.write_text(json.dumps(model_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (args.target / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"HF checkpoint ready: {args.target} ({count} shards, {total_size / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()

