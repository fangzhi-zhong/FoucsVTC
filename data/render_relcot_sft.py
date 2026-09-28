#!/usr/bin/env python3
"""Render released REL-CoT text into multiple DPI views and SFT manifests."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
from contextlib import ExitStack
from multiprocessing import Pool
from pathlib import Path


DPIS = (48, 60, 72, 84, 96, 120, 144)
PUBLISHED_PREFIX = "/vepfs-mlp2/c20250405/400042/data/VTC/SFT/gemini-3.5-flash-30k"
CONFIG = {}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_asset(value: str, config: dict) -> Path:
    path = Path(value).expanduser()
    root = Path(config["dataset_root"])
    if not path.is_absolute():
        path = root / path
    else:
        # Apply the published prefix before testing existence: a local download
        # must win even when the original producer's directory is still mounted.
        try:
            path = root / path.relative_to(config["source_prefix"])
        except ValueError:
            pass
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"asset not found: {path}; check --dataset-root / --source-prefix")
    return path


def load_inputs(row: dict, config: dict) -> tuple:
    from PIL import Image

    image_values = row.get("image_dpi72") or row.get("image")
    if not isinstance(image_values, list) or not image_values:
        raise ValueError("expected an ordered list of 72-DPI images")
    images = [resolve_asset(value, config) for value in image_values]
    sizes = []
    for path in images:
        with Image.open(path) as image:
            sizes.append(list(image.size))
    text_path = resolve_asset(row["text_path"], config)
    text = text_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("source text is empty")
    conversation = row["conversations"]
    assets = [*images, text_path]
    if isinstance(conversation, str):
        path = resolve_asset(conversation, config)
        assets.append(path)
        conversation = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("conversations must contain a nonempty list of turns")
    placeholders = 0
    has_target = False
    for turn in conversation:
        role = turn.get("from", turn.get("role"))
        value = turn.get("value", turn.get("content"))
        if not isinstance(value, str):
            raise ValueError("conversation turns must contain text")
        if role in ("human", "user"):
            placeholders += value.count("<image>")
        has_target |= role in ("gpt", "assistant") and bool(value.strip())
    if placeholders != len(images) or not has_target:
        raise ValueError("conversation must have one <image> per page and an assistant target")
    signatures = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in assets}
    return images, sizes, text_path, text, conversation, signatures


def locations_for_view(locations: list, sizes: list) -> list:
    result = []
    for location in locations:
        item = copy.deepcopy(location)
        page = item["page"]
        box = item["bbox"]
        if not isinstance(page, int) or not 1 <= page <= len(sizes):
            raise ValueError(f"invalid evidence page: {page}")
        if (not isinstance(box, list) or len(box) != 4
                or not all(isinstance(value, (float, int)) and math.isfinite(value) for value in box)
                or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)):
            raise ValueError(f"invalid normalized evidence box: {box}")
        width, height = sizes[page - 1]
        item["bbox_abs"] = [math.floor(box[0] * width / 1000), math.floor(box[1] * height / 1000),
                            math.ceil(box[2] * width / 1000), math.ceil(box[3] * height / 1000)]
        # Token/patch indices belong to a specific processor and image size.
        # SFT reads the unchanged normalized boxes in the conversation instead.
        for key in ("evidence_token_id", "evidence_token_ids", "token_ids", "patch_indices",
                    "image_size", "grid_size", "token_grid", "resized_size"):
            item.pop(key, None)
        result.append(item)
    return result


def make_records(row: dict, views: dict, conversation_path: Path, text_path: Path,
                 selected_font: str | None) -> list:
    records = []
    for dpi in CONFIG["dpis"]:
        view = views[dpi]
        record = copy.deepcopy(row)
        # Old 96/144 fields are present in the release even though their image
        # files are not shipped. Only describe views that actually exist here.
        for key in list(record):
            if (re.match(r"^(?:image|images)(?:_sizes)?_dpi\d+$", key)
                    or re.match(r"^images_\d+dpi$", key)
                    or key.startswith("vision_tokens")
                    or key in ("high_res_images", "low_res_images", "images_high_res",
                               "high_res_dpi", "low_res_dpi")):
                record.pop(key)
        metadata = record.setdefault("metadata", {})
        for key in list(metadata):
            if key.startswith("vision_tokens") or key in ("evidence_token_id", "evidence_token_ids"):
                metadata.pop(key)
        for field in ("all_locations", "evidence_locations"):
            if field in metadata:
                metadata[field] = locations_for_view(metadata[field], view["image_sizes"])
        metadata.update(base_id=row["id"], dpi=dpi, source_dataset="zhongfangzhi/REL-CoT")
        if selected_font:
            metadata["render_font"] = selected_font
        record.update(id=f"{row['id']}__{dpi}dpi", image=view["image"],
                      image_sizes=view["image_sizes"], num_pages=view["num_pages"], dpi=dpi,
                      conversations=str(conversation_path), text_path=str(text_path),
                      low_res_images=view["image"], low_res_dpi=dpi)
        for rendered_dpi, rendered_view in views.items():
            record[f"image_dpi{rendered_dpi}"] = rendered_view["image"]
        if 144 in views:
            record.update(high_res_images=views[144]["image"], high_res_dpi=144,
                          image_sizes_dpi144=views[144]["image_sizes"])
        records.append(record)
    return records


def init_worker(config: dict) -> None:
    global CONFIG
    CONFIG = config


def process_sample(task: tuple) -> dict:
    index, row, resume = task
    sample_id = row.get("id")
    try:
        images, sizes, text_path, text, conversation, signatures = load_inputs(row, CONFIG)
        root = Path(CONFIG["out_root"])
        key = f"sample_{index:08d}"
        state_path = root / "state" / f"{key}.json"
        if resume and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["source_row"] == row and state["assets"] == signatures:
                required = {p for record in state["records"] for p in record["image"]}
                required.update(record["conversations"] for record in state["records"])
                if all(Path(path).is_file() for path in required):
                    return {"id": sample_id, "records": state["records"], "reused": True}

        output_dirs = {dpi: str(root / "images" / f"dpi_{dpi}" / key)
                       for dpi in CONFIG["dpis"]}
        if __package__:
            from .relcot_rendering import render_sample
            from .relcot_annotations import collect_locations, remap_metadata, rewrite_conversation
        else:
            from relcot_rendering import render_sample
            from relcot_annotations import collect_locations, remap_metadata, rewrite_conversation

        rendered = render_sample(
            text, [str(path) for path in images], output_dirs, CONFIG["font_path"],
            source_font_paths=CONFIG["source_font_paths"], poppler_path=CONFIG["poppler_path"],
            locations=collect_locations(row, conversation),
        )
        views = rendered["views"]
        selected_font = rendered["font_path"]
        num_pages = next(iter(views.values()))["num_pages"]
        conversation = rewrite_conversation(conversation, rendered["location_map"],
                                            rendered["page_map"], num_pages)
        placeholders = sum(
            turn.get("value", turn.get("content", "")).count("<image>")
            for turn in conversation
            if turn.get("from", turn.get("role")) in ("human", "user")
        )
        if placeholders != num_pages:
            raise ValueError("rewritten conversation must have one <image> per rendered page; "
                             "multiple image-bearing human turns are unsupported")
        conversation_path = root / "conversations" / f"{key}.json"
        atomic_json(conversation_path, conversation)
        mapped_row = copy.deepcopy(row)
        mapped_row["metadata"] = remap_metadata(row.get("metadata", {}), rendered["location_map"])
        records = make_records(mapped_row, views, conversation_path, text_path, selected_font)
        atomic_json(state_path, {"source_row": row, "assets": signatures, "records": records})
        return {"id": sample_id, "records": records, "reused": False}
    except Exception as exc:
        return {"id": sample_id, "error": f"{type(exc).__name__}: {exc}"}


def iter_tasks(path: Path, limit: int, resume: bool):
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError(f"missing or duplicate source id at {path}:{line_number}")
            seen.add(sample_id)
            yield len(seen), row, resume
            if limit and len(seen) >= limit:
                return
    if not seen:
        raise ValueError(f"empty input manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="unpacked REL-CoT directory")
    parser.add_argument("--input", type=Path, help="JSONL manifest; default: <dataset-root>/train_72dpi.json")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-prefix", default=PUBLISHED_PREFIX,
                        help="original absolute asset prefix to map to --dataset-root")
    parser.add_argument("--font-path", type=Path,
                        default=os.environ.get("FOCUSVTC_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                        help="DejaVu Sans font used for every output DPI")
    parser.add_argument("--source-font-path", type=Path, action="append", default=[],
                        help="additional original font for locating evidence in older 72-DPI pages; never used for output")
    parser.add_argument("--dpis", type=int, nargs="+", choices=DPIS, default=list(DPIS))
    parser.add_argument("--poppler-path", type=Path, help="directory containing pdftoppm and pdfinfo")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="source samples to process; 0 means all")
    parser.add_argument("--resume", action="store_true", help="reuse completed samples with unchanged inputs")
    args = parser.parse_args()
    if args.processes < 1 or args.limit < 0:
        parser.error("--processes must be positive; --limit must be nonnegative")
    dataset_root = args.dataset_root.expanduser().resolve()
    default_input = dataset_root / "train_72dpi.json"
    if not default_input.is_file() and (dataset_root / "train.jsonl").is_file():
        default_input = dataset_root / "train.jsonl"
    input_path = (args.input or default_input).expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"input manifest does not exist: {input_path}")
    if out_root == dataset_root or out_root in dataset_root.parents:
        parser.error("--out-root must be separate from the source dataset")
    code_root = Path(__file__).resolve().parent
    if out_root == code_root or code_root in out_root.parents:
        parser.error("data/ contains tools only; put rendered assets under datasets/ or an external directory")
    font_path = args.font_path.expanduser().resolve()
    source_fonts = [path.expanduser().resolve() for path in args.source_font_path]
    fonts = [font_path, *source_fonts]
    for font in fonts:
        if not font.is_file():
            parser.error(f"font does not exist: {font}")
    config = {
        "dataset_root": str(dataset_root), "input": str(input_path), "out_root": str(out_root),
        "source_prefix": str(Path(args.source_prefix).expanduser()), "dpis": sorted(set(args.dpis)),
        "font_path": str(font_path), "source_font_paths": [str(path) for path in source_fonts],
        "font_files": [[str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in fonts],
        "poppler_path": str(args.poppler_path.expanduser().resolve()) if args.poppler_path else None,
    }
    config_path = out_root / "render_config.json"
    if out_root.exists() and any(out_root.iterdir()):
        if not args.resume or not config_path.is_file():
            parser.error("output directory is not empty; use a fresh directory or --resume")
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            parser.error("render options changed; use a fresh output directory")
    out_root.mkdir(parents=True, exist_ok=True)
    atomic_json(config_path, config)
    init_worker(config)
    targets = {dpi: out_root / f"train_{dpi}dpi.jsonl" for dpi in config["dpis"]}
    targets["all"] = out_root / "train.jsonl"
    completed = reused = failed = 0
    with ExitStack() as stack:
        handles = {key: stack.enter_context(path.with_suffix(path.suffix + ".tmp").open("w", encoding="utf-8"))
                   for key, path in targets.items()}
        failures = stack.enter_context((out_root / "failed.jsonl").open("w", encoding="utf-8"))
        tasks = iter_tasks(input_path, args.limit, args.resume)
        if args.processes == 1:
            results = map(process_sample, tasks)
        else:
            pool = stack.enter_context(Pool(args.processes, initializer=init_worker, initargs=(config,)))
            results = pool.imap(process_sample, tasks, chunksize=1)
        for result in results:
            if "error" in result:
                failed += 1
                failures.write(json.dumps(result, ensure_ascii=False) + "\n")
                failures.flush()
                print(f"FAILED {result['id']}: {result['error']}", flush=True)
                continue
            for record in result["records"]:
                line = json.dumps(record, ensure_ascii=False) + "\n"
                handles[record["dpi"]].write(line)
                handles["all"].write(line)
            completed += 1
            reused += int(result["reused"])
            if completed % 100 == 0:
                print(f"Completed {completed} sources ({reused} reused), {failed} failed", flush=True)
    if failed:
        raise SystemExit(f"{failed} samples failed; inspect {out_root / 'failed.jsonl'} and rerun with --resume")
    for path in targets.values():
        path.with_suffix(path.suffix + ".tmp").replace(path)
    print(f"Wrote {completed * len(config['dpis'])} SFT rows from {completed} sources ({reused} reused): {out_root / 'train.jsonl'}")


if __name__ == "__main__":
    main()
