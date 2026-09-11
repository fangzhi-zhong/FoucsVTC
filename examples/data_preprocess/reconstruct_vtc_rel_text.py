#!/usr/bin/env python3
"""Recover the exact text inputs rendered by the VTC-REL training images.

The script covers every row in
``data/VTC_REL/gemini-3.5-flash-30k/train.jsonl``.  It writes one UTF-8 text
file per sample below ``text/`` and atomically adds an absolute ``text_path``
to every manifest row.

Recovery intentionally follows the original builders instead of OCR:

* TRANSCRIBE_SFT: replay the seeded ChatQA2 chunk plan; resolve RULER needles.
* VTC_SFT: resolve the field consumed by each source-specific word2png script.
* VTC_GAP: replay count/needle/long/code generation and validate stored golds.
* RULER_v1_SFT: resolve the upstream RULER JSONL row encoded in the sample id.

Large top-level JSON arrays are streamed.  The TRANSCRIBE plan stores only
source indices and document lengths on its first pass, so it does not retain
several gigabytes of ChatQA2 text in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


DEFAULT_ROOT = Path("/vepfs-mlp2/c20250405/400042")
DEFAULT_DATASET = DEFAULT_ROOT / "data/VTC_REL/gemini-3.5-flash-30k"

TRANSCRIBE_SEED = 20260804
GAP_SEED = 20260805
TRANSCRIBE_PAGE_CHARS = (5200, 6800)
TRANSCRIBE_BOX_WORDS = (4, 28)

ARTICLE_RE = re.compile(
    r"Article:\s*\n(.*?)(?:\n\n(?:Question|User):|\n(?:Question|User):|\Z)",
    re.S | re.I,
)
BOM_RE = re.compile(r"^(?:<\|begin_of_text\|>)?\ufeff?")


@dataclass(frozen=True)
class SampleInfo:
    sid: str
    origin: str
    group: tuple[str, ...]
    text_path: Path
    conversation: Path
    image0: str
    metadata: dict


def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Iterator[dict]:
    """Stream a top-level JSON array without the empty-buffer early-exit bug."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buf = ""
        eof = False
        started = False
        expect_separator = False

        while True:
            if not eof and len(buf) < chunk_size:
                chunk = handle.read(chunk_size)
                if chunk:
                    buf += chunk
                else:
                    eof = True

            pos = 0
            size = len(buf)
            if not started:
                while pos < size and buf[pos].isspace():
                    pos += 1
                if pos >= size:
                    if eof:
                        raise ValueError(f"{path}: empty JSON input")
                    buf = ""
                    continue
                if buf[pos] != "[":
                    raise ValueError(f"{path}: top-level value is not an array")
                pos += 1
                started = True

            while True:
                while pos < size and buf[pos].isspace():
                    pos += 1
                if pos >= size:
                    break

                if expect_separator:
                    if buf[pos] == ",":
                        pos += 1
                        expect_separator = False
                        continue
                    if buf[pos] == "]":
                        pos += 1
                        if buf[pos:].strip():
                            if eof:
                                raise ValueError(f"{path}: trailing data after array")
                            break
                        return
                    raise ValueError(f"{path}: expected ',' or ']' near buffer offset {pos}")

                if buf[pos] == "]":
                    pos += 1
                    if buf[pos:].strip():
                        if eof:
                            raise ValueError(f"{path}: trailing data after array")
                        break
                    return

                try:
                    value, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: array member is not an object")
                yield value
                pos = end
                expect_separator = True

            buf = buf[pos:]
            if eof:
                if buf.strip():
                    raise ValueError(f"{path}: incomplete JSON array")
                return


def visual_text(text: str) -> str:
    """Mirror the renderer-only invisible-character cleanup."""
    return str(text).replace("\xad", "").replace("\u200b", "")


def clean_transcribe_doc(text: str) -> str:
    text = BOM_RE.sub("", text or "")
    return visual_text(text).strip()


def extract_long_sft_article(question: str) -> str:
    match = ARTICLE_RE.search(question or "")
    if match:
        return clean_transcribe_doc(match.group(1))
    marker = "Article:\n"
    if marker in (question or ""):
        return clean_transcribe_doc(question.split(marker, 1)[1])
    return clean_transcribe_doc(question or "")


def split_ruler_v2_question(question: str) -> str:
    parts = re.split(r"\n\n+", question or "", maxsplit=1)
    if len(parts) == 2 and len(parts[1]) > 500:
        return parts[1].strip()
    return (question or "").strip()


def planned_chunk(rng: random.Random, length: int) -> tuple[int, int | None]:
    target = rng.randint(*TRANSCRIBE_PAGE_CHARS)
    if length <= target:
        return target, None
    return target, rng.randint(0, length - target)


def apply_planned_chunk(doc: str, target: int, start: int | None) -> str:
    if start is None or len(doc) <= target:
        return doc
    while start > 0 and doc[start - 1] not in " \n\t":
        start -= 1
    chunk = doc[start : start + target]
    for separator in (". ", ".\n", " ", "\n"):
        end = chunk.rfind(separator)
        if end > target // 2:
            return chunk[: end + (1 if separator.startswith(".") else 0)].strip()
    return chunk.strip()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conversation_target(path: Path) -> str:
    conversation = json.loads(path.read_text(encoding="utf-8"))
    turns = [turn for turn in conversation if turn.get("from") in ("gpt", "assistant")]
    if not turns:
        raise ValueError(f"{path}: no GPT turn")
    value = str(turns[-1].get("value", ""))
    if "</think>" in value:
        value = value.split("</think>", 1)[1]
        if value.startswith("\n"):
            value = value[1:]
    return value


def clean_code(content: str) -> str:
    content = str(content).replace("\r\n", "\n").replace("\r", "\n")
    content = content.replace("\ufeff", "")
    lines = []
    for line in content.split("\n"):
        if len(line) <= 240:
            lines.append(line.rstrip())
    return "\n".join(lines).strip()


def pick_complete_cut(rng: random.Random, content: str) -> tuple[str, str] | None:
    if len(content) < 12000:
        return None
    lines = content.split("\n")
    if len(lines) < 40:
        return None
    lo = max(20, int(len(lines) * 0.70))
    hi = min(len(lines) - 2, int(len(lines) * 0.92))
    if hi <= lo:
        return None
    skip = ("#", "//", "/*", "*", "import ", "using ", "package ", "from ")
    candidates = [
        index
        for index in range(lo, hi + 1)
        if 4 <= len(lines[index].strip()) <= 200
        and not lines[index].strip().startswith(skip)
    ]
    if not candidates:
        candidates = [index for index in range(lo, hi + 1) if lines[index].strip()]
    if not candidates:
        return None
    cut = rng.choice(candidates)
    prefix = "\n".join(lines[:cut])
    target = lines[cut]
    if len(prefix) > 55000:
        prefix = prefix[-55000:]
        newline = prefix.find("\n")
        if newline > 0:
            prefix = prefix[newline + 1 :]
    if len(prefix) < 8000:
        return None
    return prefix, target


def pad_long_document(
    rng: random.Random,
    evidence: str,
    distractors: list[str],
    target_chars: int,
    depth: float,
) -> tuple[str, dict]:
    evidence = evidence.strip()
    if len(evidence) > target_chars:
        raise ValueError("evidence exceeds target length")
    needed = target_chars - len(evidence)
    before_budget = int(needed * depth)
    after_budget = needed - before_budget

    def fill(budget: int) -> str:
        parts: list[str] = []
        used = 0
        order = list(range(len(distractors)))
        rng.shuffle(order)
        for index in order:
            text = distractors[index]
            if text == evidence:
                continue
            if used >= budget:
                break
            take = min(len(text), budget - used)
            chunk = text[:take]
            cut = chunk.rfind("\n\n")
            if cut > take // 2:
                chunk = chunk[:cut]
            if chunk.strip():
                parts.append(chunk.strip())
                used += len(chunk)
        return "\n\n".join(parts)

    before = fill(before_budget)
    after = fill(after_budget)
    body = "\n\n".join(part for part in (before, evidence, after) if part)
    start = len(before) + 2 if before else 0
    end = start + len(evidence)
    if body[start:end] != evidence:
        raise AssertionError("long evidence offsets drifted")
    return body, {"evidence_start_char": start, "evidence_end_char": end}


class Reconstructor:
    def __init__(self, root: Path, dataset: Path, update_manifest: bool = True):
        self.root = root
        self.dataset = dataset
        self.train_path = dataset / "train.jsonl"
        self.text_root = dataset / "text"
        self.update_manifest_requested = update_manifest
        self.samples: dict[str, SampleInfo] = {}
        self.by_origin: dict[str, set[str]] = defaultdict(set)
        self.written: set[str] = set()
        self.bytes_written = 0
        self.resolver_counts: Counter[str] = Counter()
        self.paragraphs: list[str] = []
        self.haystacks: list[str] = []

    def scan_manifest(self) -> None:
        seen: set[str] = set()
        with self.train_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                sid = row.get("id")
                if not isinstance(sid, str) or not sid:
                    raise ValueError(f"manifest line {line_no}: invalid id")
                if sid in seen:
                    raise ValueError(f"manifest line {line_no}: duplicate id {sid}")
                seen.add(sid)
                images = row.get("image") or []
                if not images:
                    raise ValueError(f"{sid}: no images")
                origin, group = self.classify_image(str(images[0]))
                text_path = self.text_root.joinpath(origin, *group, f"{sid}.txt")
                metadata = row.get("metadata") or {}
                if origin == "VTC_GAP":
                    keep = {
                        key: metadata.get(key)
                        for key in (
                            "kind", "gold", "counts", "n_items", "depth",
                            "repo", "path", "lang", "chars", "src_id",
                            "corpus", "variant", "target_chars", "seed",
                            "body_sha256", "doc_chars", "evidence_start_char",
                            "evidence_end_char",
                        )
                    }
                else:
                    keep = {}
                info = SampleInfo(
                    sid=sid,
                    origin=origin,
                    group=group,
                    text_path=text_path,
                    conversation=Path(row["conversations"]),
                    image0=str(images[0]),
                    metadata=keep,
                )
                self.samples[sid] = info
                self.by_origin[origin].add(sid)
        print("manifest", len(self.samples), dict(sorted((k, len(v)) for k, v in self.by_origin.items())), flush=True)

    @staticmethod
    def classify_image(image: str) -> tuple[str, tuple[str, ...]]:
        if "/data/" not in image:
            raise ValueError(f"image path does not contain /data/: {image}")
        parts = image.split("/data/", 1)[1].split("/")
        origin = parts[0]
        if origin == "TRANSCRIBE_SFT" and parts[1] == "images":
            return origin, (parts[2],)
        if origin == "RULER_v1_SFT" and parts[1] == "images":
            return origin, (parts[2],)
        if origin == "VTC_SFT" and parts[1] == "images":
            return origin, (parts[2], parts[3])
        if origin == "VTC_GAP" and parts[2] == "images":
            return origin, (parts[1], parts[3])
        raise ValueError(f"unrecognised image layout: {image}")

    def write(self, sid: str, text: str, resolver: str) -> None:
        if sid not in self.samples:
            return
        if sid in self.written:
            raise ValueError(f"{sid}: text resolved more than once")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{sid}: recovered text is empty")
        path = self.samples[sid].text_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
        self.written.add(sid)
        self.bytes_written += len(text.encode("utf-8"))
        self.resolver_counts[resolver] += 1

    def resolve_ruler(self) -> None:
        main_needed = self.by_origin.get("RULER_v1_SFT", set()) - self.written
        trans_needed = {
            sid
            for sid in self.by_origin.get("TRANSCRIBE_SFT", set())
            if sid.startswith("tr_needle_ruler_")
        }
        src = self.root / "data/huggingface/ruler_jsonl"
        for length in (4096, 8192, 16384):
            path = src / f"ruler_{length}.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    row = json.loads(line)
                    base = f"ruler_{row['task']}_{length}_{index:04d}"
                    if base in main_needed:
                        self.write(base, str(row["context"]), "RULER_v1")
                    trans = f"tr_needle_{base}"
                    if trans in trans_needed:
                        self.write(trans, str(row["context"]), "TRANSCRIBE_RULER_v1")

        v2_needed = {
            sid
            for sid in self.by_origin.get("TRANSCRIBE_SFT", set())
            if sid.startswith("tr_needle_ruler2_")
        }
        v2_root = self.root / "data/RULER_v2/ruler2/Qwen3-VL-8B-Instruct-8192"
        for task_dir in sorted(v2_root.iterdir()):
            path = task_dir / "test.jsonl"
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    sid = f"tr_needle_ruler2_{task_dir.name}_{row['index']:04d}"
                    if sid in v2_needed:
                        self.write(sid, split_ruler_v2_question(row["question"]), "TRANSCRIBE_RULER_v2")
        print("ruler resolved", len(main_needed), len(trans_needed), len(v2_needed), flush=True)

    def _scan_chatqa2_first_pass(
        self,
        path: Path,
        corpus: str,
        pool_raw_limit: int,
        direct_needed: set[str],
    ) -> list[tuple[int, int]]:
        refs: list[tuple[int, int]] = []
        remaining = set(direct_needed)
        paragraph_seen: set[str] = set()
        collect_paragraphs = corpus == "long_sft" and not self.paragraphs
        collect_haystacks = corpus == "long_sft" and not self.haystacks

        for raw_index, row in enumerate(iter_json_array(path)):
            row_id = row.get("id")
            raw = row.get("question", "") if corpus == "long_sft" else row.get("sub-paragraphs", "")
            if row_id in remaining:
                self.write(row_id, visual_text(raw), f"VTC_SFT_ChatQA2_{corpus}")
                remaining.remove(row_id)

            doc = extract_long_sft_article(raw) if corpus == "long_sft" else clean_transcribe_doc(raw)
            # The original TRANSCRIBE loader capped the number of raw array
            # objects inspected (57k/18k), not the number of eligible docs.
            if raw_index < pool_raw_limit and len(doc) >= 800:
                refs.append((raw_index, len(doc)))

            if collect_haystacks and len(doc) >= 5000:
                self.haystacks.append(doc)
                if len(self.haystacks) >= 1500:
                    collect_haystacks = False

            if collect_paragraphs:
                for paragraph in doc.split("\n"):
                    paragraph = paragraph.strip()
                    if 600 <= len(paragraph) <= 2000 and paragraph not in paragraph_seen:
                        paragraph_seen.add(paragraph)
                        self.paragraphs.append(paragraph)
                if len(self.paragraphs) >= 4000:
                    collect_paragraphs = False

            if (
                not remaining
                and raw_index + 1 >= pool_raw_limit
                and not collect_paragraphs
                and not collect_haystacks
            ):
                break

        if remaining:
            raise ValueError(f"{path}: missing {len(remaining)} requested ids, e.g. {sorted(remaining)[:3]}")
        minimum = 17000 if corpus == "long_sft" else 8000
        if len(refs) < minimum:
            raise ValueError(f"{path}: eligible pool {len(refs)} < {minimum}")
        print(f"TRANSCRIBE {corpus} pool {len(refs)} from first {pool_raw_limit} rows", flush=True)
        return refs

    def resolve_chatqa2_and_transcribe(self) -> None:
        chat_root = self.root / "data/modelscope/datasets/nv-community/ChatQA2-Long-SFT-data"
        long_path = chat_root / "long_sft/long_sft_QA_train.json"
        narr_path = chat_root / "NarrativeQA_131072/NarrativeQA_131072_QA_train.json"
        long_direct = {
            sid for sid, info in self.samples.items()
            if info.origin == "VTC_SFT" and info.group == ("ChatQA2-Long-SFT-data", "long_sft")
        }
        narr_direct = {
            sid for sid, info in self.samples.items()
            if info.origin == "VTC_SFT" and info.group == ("ChatQA2-Long-SFT-data", "NarrativeQA_131072")
        }
        print("ChatQA2 first pass: long_sft", flush=True)
        long_refs = self._scan_chatqa2_first_pass(long_path, "long_sft", 57000, long_direct)
        print("ChatQA2 first pass: NarrativeQA", flush=True)
        narr_refs = self._scan_chatqa2_first_pass(narr_path, "narrativeqa", 18000, narr_direct)
        print("aux pools", {"paragraphs": len(self.paragraphs), "haystacks": len(self.haystacks)}, flush=True)

        rng = random.Random(TRANSCRIBE_SEED)
        rng.shuffle(long_refs)
        rng.shuffle(narr_refs)
        cursors = {"long_sft": 0, "narrativeqa": 0}
        refs_by_corpus = {"long_sft": long_refs, "narrativeqa": narr_refs}
        planned_by_source: dict[str, dict[int, tuple[str, int, int | None]]] = {
            "long_sft": {}, "narrativeqa": {}
        }
        plans = [
            ("page", (("long_sft", 10000), ("narrativeqa", 5000))),
            ("box", (("long_sft", 7000), ("narrativeqa", 3000))),
        ]
        trans_needed = self.by_origin["TRANSCRIBE_SFT"]
        for kind, sources in plans:
            for corpus, count in sources:
                start_cursor = cursors[corpus]
                for local_index in range(count):
                    raw_index, length = refs_by_corpus[corpus][start_cursor + local_index]
                    target, start = planned_chunk(rng, length)
                    sid = f"tr_{kind}_{corpus}_{local_index:05d}"
                    rng.randrange(4)  # renderer prompt variant
                    if kind == "box":
                        rng.randint(*TRANSCRIBE_BOX_WORDS)  # historical unused draw
                    if sid in trans_needed:
                        if raw_index in planned_by_source[corpus]:
                            raise AssertionError(f"duplicate TRANSCRIBE source index {corpus}:{raw_index}")
                        planned_by_source[corpus][raw_index] = (sid, target, start)
                cursors[corpus] += count

        for corpus, path in (("long_sft", long_path), ("narrativeqa", narr_path)):
            pending = planned_by_source[corpus]
            remaining_contexts = {sid for sid, _, _ in pending.values()}
            print(f"ChatQA2 second pass: {corpus}, contexts={len(pending)}", flush=True)
            for raw_index, row in enumerate(iter_json_array(path)):
                plan = pending.get(raw_index)
                if plan is None:
                    continue
                sid, target, start = plan
                raw = row.get("question", "") if corpus == "long_sft" else row.get("sub-paragraphs", "")
                doc = extract_long_sft_article(raw) if corpus == "long_sft" else clean_transcribe_doc(raw)
                self.write(sid, apply_planned_chunk(doc, target, start), "TRANSCRIBE_ChatQA2")
                remaining_contexts.remove(sid)
                if not remaining_contexts:
                    break

    def resolve_chatqa_training(self) -> None:
        root = self.root / "data/modelscope/datasets/nv-community/ChatQA-Training-Data"
        groups: dict[str, set[str]] = defaultdict(set)
        for sid, info in self.samples.items():
            if info.origin == "VTC_SFT" and info.group[0] == "ChatQA-Training-Data":
                groups[info.group[1]].add(sid)
        for subset, needed in sorted(groups.items()):
            remaining = set(needed)
            files = sorted((root / subset).glob("*.json"))
            for path in files:
                for row in iter_json_array(path):
                    sid = row.get("id")
                    if sid in remaining:
                        self.write(sid, visual_text(row.get("document", "")), f"VTC_SFT_ChatQA_{subset}")
                        remaining.remove(sid)
            if remaining:
                raise ValueError(f"ChatQA/{subset}: missing {len(remaining)} ids, e.g. {sorted(remaining)[:3]}")
            print("ChatQA", subset, len(needed), flush=True)

    def resolve_trivia(self) -> None:
        needed = {
            sid for sid, info in self.samples.items()
            if info.origin == "VTC_SFT" and info.group == ("trivia_qa", "rc_json")
        }
        by_file: dict[str, set[str]] = defaultdict(set)
        for sid in needed:
            stem, _ = sid.rsplit("_", 1)
            by_file[stem].add(sid)
        root = self.root / "data/modelscope/datasets/evalscope/trivia_qa/rc_json"
        for stem, ids in sorted(by_file.items()):
            remaining = set(ids)
            path = root / f"{stem}.json"
            for row in iter_json_array(path):
                sid = row.get("id")
                if sid not in remaining:
                    continue
                search = (row.get("search_results") or {}).get("search_context")
                if isinstance(search, list):
                    text = "\n\n\n".join(str(value) for value in search if value is not None)
                else:
                    text = str(search or "")
                self.write(sid, visual_text(text), "VTC_SFT_TriviaQA")
                remaining.remove(sid)
                if not remaining:
                    break
            if remaining:
                raise ValueError(f"Trivia {stem}: missing {len(remaining)} ids")
            print("Trivia", stem, len(ids), flush=True)

    def resolve_multihop_and_gap_long(self) -> None:
        multihop_needed = {
            sid for sid, info in self.samples.items()
            if info.origin == "VTC_SFT" and info.group[0] == "multihop"
        }
        long_samples = [
            info for info in self.samples.values()
            if info.origin == "VTC_GAP" and info.group[0] == "long"
        ]
        evidence_keys = {(info.metadata["variant"], info.metadata["src_id"]) for info in long_samples}
        evidence_docs: dict[tuple[str, str], str] = {}
        distractors: dict[str, list[str]] = defaultdict(list)
        source_root = self.root / "data/multihop_src"

        variants = (
            ("hotpotqa", "hotpotqa"), ("hotpotqa", "hotpotqa_long"),
            ("2wiki", "2wiki"), ("2wiki", "2wiki_long"),
            ("musique", "musique"), ("musique", "musique_long"),
            ("finqa", "finqa"),
        )
        remaining = set(multihop_needed)
        for corpus, variant in variants:
            path = source_root / variant / "train.json"
            for row in iter_json_array(path):
                sid = row.get("id")
                document_raw = str(row.get("document") or "")
                if sid in remaining:
                    self.write(sid, visual_text(document_raw), f"VTC_SFT_multihop_{variant}")
                    remaining.remove(sid)
                document = document_raw.strip()
                if corpus != "finqa" and len(document) >= 800:
                    distractors[corpus].append(document)
                    key = (variant, sid)
                    if key in evidence_keys:
                        evidence_docs[key] = document
            print("multihop source", variant, "distractors", len(distractors.get(corpus, [])), flush=True)
        if remaining:
            raise ValueError(f"multihop: missing {len(remaining)} ids, e.g. {sorted(remaining)[:3]}")
        missing_evidence = evidence_keys - set(evidence_docs)
        if missing_evidence:
            raise ValueError(f"GAP long: missing {len(missing_evidence)} evidence docs")

        for index, info in enumerate(long_samples, 1):
            meta = info.metadata
            evidence = evidence_docs[(meta["variant"], meta["src_id"])]
            body, audit = pad_long_document(
                random.Random(int(meta["seed"])), evidence,
                distractors[meta["corpus"]], int(meta["target_chars"]), float(meta["depth"]),
            )
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if digest != meta["body_sha256"]:
                raise ValueError(f"{info.sid}: body SHA mismatch {digest} != {meta['body_sha256']}")
            if len(body) != int(meta["doc_chars"]):
                raise ValueError(f"{info.sid}: doc length mismatch")
            if audit["evidence_start_char"] != int(meta["evidence_start_char"]):
                raise ValueError(f"{info.sid}: evidence start mismatch")
            if audit["evidence_end_char"] != int(meta["evidence_end_char"]):
                raise ValueError(f"{info.sid}: evidence end mismatch")
            self.write(info.sid, body, "VTC_GAP_long")
            if index % 500 == 0:
                print("GAP long", index, "/", len(long_samples), flush=True)

    def resolve_gap_count_and_needle(self) -> None:
        gap_root = self.root / "data/VTC_GAP"
        gen_count = load_module("vtc_gap_gen_count", gap_root / "gen_count.py")
        gen_needle = load_module("vtc_gap_gen_needle", gap_root / "gen_needle.py")

        rng = random.Random(GAP_SEED)
        count_specs: list[dict] = []
        count_plan = (("cwe", 1500), ("fwe", 1500), ("pcnt", 1000))
        sizes = {
            "cwe": gen_count.CWE_ITEMS,
            "fwe": gen_count.FWE_TOKENS,
            "pcnt": gen_count.PCNT_PARAS,
        }
        for kind, count in count_plan:
            for index in range(count):
                count_specs.append({
                    "id": f"cnt_{kind}_{index:05d}",
                    "kind": kind,
                    "size": sizes[kind][index % len(sizes[kind])],
                    "seed": rng.randrange(1 << 30),
                })
        for spec in count_specs:
            info = self.samples[spec["id"]]
            sample_rng = random.Random(spec["seed"])
            if spec["kind"] == "cwe":
                sample = gen_count.gen_cwe(sample_rng, spec["size"])
            elif spec["kind"] == "fwe":
                sample = gen_count.gen_fwe(sample_rng, spec["size"])
            else:
                sample = gen_count.gen_passage_count(sample_rng, self.paragraphs, spec["size"])
            error = gen_count.verify(spec["kind"], sample)
            if error:
                raise ValueError(f"{spec['id']}: {error}")
            if list(sample["answer"]) != list(info.metadata["gold"]):
                raise ValueError(f"{spec['id']}: count gold mismatch")
            if sample["counts"] != info.metadata["counts"]:
                raise ValueError(f"{spec['id']}: count metadata mismatch")
            self.write(spec["id"], sample["context"], f"VTC_GAP_count_{spec['kind']}")
        print("GAP count", len(count_specs), flush=True)

        rng = random.Random(GAP_SEED)
        needle_specs: list[dict] = []
        for index in range(1000):
            needle_specs.append({
                "id": f"nid_single_{index:05d}", "kind": "single",
                "size": gen_needle.SINGLE_CHARS[index % len(gen_needle.SINGLE_CHARS)],
                "depth": gen_needle.DEPTHS[index % len(gen_needle.DEPTHS)],
                "doc": index % len(self.haystacks), "seed": rng.randrange(1 << 30),
            })
        for index in range(1000):
            needle_specs.append({
                "id": f"nid_multikey_{index:05d}", "kind": "multikey",
                "size": gen_needle.MULTI_LINES[index % len(gen_needle.MULTI_LINES)],
                "depth": gen_needle.DEPTHS[index % len(gen_needle.DEPTHS)],
                "seed": rng.randrange(1 << 30),
            })
        for spec in needle_specs:
            info = self.samples[spec["id"]]
            sample_rng = random.Random(spec["seed"])
            if spec["kind"] == "single":
                doc = self.haystacks[spec["doc"]]
                if len(doc) < spec["size"]:
                    doc = (doc + " ") * (spec["size"] // len(doc) + 1)
                sample = gen_needle.gen_single(sample_rng, doc, spec["size"], spec["depth"])
            else:
                sample = gen_needle.gen_multikey(sample_rng, spec["size"], spec["depth"])
            if sample["answer"][0] != info.metadata["gold"]:
                raise ValueError(f"{spec['id']}: needle gold mismatch")
            self.write(spec["id"], sample["context"], f"VTC_GAP_needle_{spec['kind']}")
        print("GAP needle", len(needle_specs), flush=True)

    def resolve_gap_code(self) -> None:
        code_infos = [
            info for info in self.samples.values()
            if info.origin == "VTC_GAP" and info.group[0] == "code"
        ]
        complete_infos = [info for info in code_infos if info.metadata["kind"] == "complete"]
        for info in code_infos:
            if info.metadata["kind"] == "page":
                text = conversation_target(info.conversation)
                if len(text) != int(info.metadata["chars"]):
                    raise ValueError(f"{info.sid}: code page character count mismatch")
                self.write(info.sid, text, "VTC_GAP_code_page")

        needed_keys: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for info in complete_infos:
            needed_keys[(info.metadata["lang"], info.metadata["repo"], info.metadata["path"])].append(info.sid)
        source_by_id: dict[str, str] = {}
        stack_root = self.root / "data/modelscope/datasets/bigcode/the-stack-smol/data"
        for lang in ("python", "java", "c-sharp"):
            lang_keys = {key for key in needed_keys if key[0] == lang}
            if not lang_keys:
                continue
            with (stack_root / lang / "data.json").open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (lang, row.get("repository_name"), row.get("path"))
                    if key in lang_keys:
                        content = clean_code(row.get("content") or "")
                        for sid in needed_keys[key]:
                            source_by_id[sid] = content
                        lang_keys.remove(key)
                        if not lang_keys:
                            break
            if lang_keys:
                raise ValueError(f"code/{lang}: missing {len(lang_keys)} repo/path sources")

        rng = random.Random(GAP_SEED)
        complete_seeds: dict[int, int] = {}
        for kind, count in (("complete", 1500), ("page", 1500)):
            for index in range(count):
                seed = rng.randrange(1 << 30)
                rng.randrange(4)
                if kind == "complete":
                    complete_seeds[index] = seed
        for info in complete_infos:
            index = int(info.sid.rsplit("_", 1)[1])
            picked = pick_complete_cut(random.Random(complete_seeds[index]), source_by_id[info.sid])
            if picked is None:
                raise ValueError(f"{info.sid}: cannot reproduce complete cut")
            prefix, target = picked
            expected = conversation_target(info.conversation)
            if target != expected:
                raise ValueError(f"{info.sid}: complete target mismatch {target!r} != {expected!r}")
            self.write(info.sid, prefix, "VTC_GAP_code_complete")
        print("GAP code", len(code_infos), flush=True)

    def verify(self) -> None:
        missing = set(self.samples) - self.written
        extra = self.written - set(self.samples)
        if missing or extra:
            raise ValueError(
                f"coverage mismatch: missing={len(missing)} {sorted(missing)[:5]}, "
                f"extra={len(extra)} {sorted(extra)[:5]}"
            )
        zero = [sid for sid, info in self.samples.items() if not info.text_path.is_file() or info.text_path.stat().st_size == 0]
        if zero:
            raise ValueError(f"empty/missing output files: {len(zero)} {zero[:5]}")
        paths = [str(info.text_path) for info in self.samples.values()]
        if len(paths) != len(set(paths)):
            raise ValueError("text_path collision detected")
        print("verified", len(self.samples), "files", self.bytes_written, "bytes", flush=True)

    def update_manifest(self) -> Path | None:
        if not self.update_manifest_requested:
            return None
        temp = self.train_path.with_name(self.train_path.name + ".tmp_text_path")
        with self.train_path.open("r", encoding="utf-8") as src, temp.open("w", encoding="utf-8") as dst:
            for line_no, line in enumerate(src, 1):
                row = json.loads(line)
                sid = row["id"]
                path = self.samples[sid].text_path
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"manifest line {line_no}: invalid text path for {sid}")
                row["text_path"] = str(path)
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            dst.flush()
            os.fsync(dst.fileno())
        backup = self.train_path.with_name(self.train_path.name + ".bak_pre_text_path")
        if not backup.exists():
            shutil.copy2(self.train_path, backup)
        os.replace(temp, self.train_path)
        return backup

    def validate_updated_manifest(self) -> None:
        count = 0
        with self.train_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                expected = str(self.samples[row["id"]].text_path)
                if row.get("text_path") != expected:
                    raise ValueError(f"updated manifest line {line_no}: text_path mismatch")
                path = Path(expected)
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"updated manifest line {line_no}: missing text file")
                count += 1
        if count != len(self.samples):
            raise ValueError(f"updated manifest rows {count} != {len(self.samples)}")
        print("updated manifest validated", count, flush=True)

    def run(self) -> dict:
        self.scan_manifest()
        self.resolve_ruler()
        self.resolve_chatqa2_and_transcribe()
        self.resolve_chatqa_training()
        self.resolve_trivia()
        self.resolve_multihop_and_gap_long()
        self.resolve_gap_count_and_needle()
        self.resolve_gap_code()
        self.verify()
        backup = self.update_manifest()
        if self.update_manifest_requested:
            self.validate_updated_manifest()
        return {
            "rows": len(self.samples),
            "origins": dict(sorted((key, len(value)) for key, value in self.by_origin.items())),
            "resolver_counts": dict(sorted(self.resolver_counts.items())),
            "text_bytes": self.bytes_written,
            "train_jsonl": str(self.train_path),
            "backup": str(backup) if backup else None,
        }


def self_test() -> None:
    payload = "Article:\nAlpha beta.\n\nQuestion: Q"
    assert extract_long_sft_article(payload) == "Alpha beta."
    rng = random.Random(7)
    target, start = planned_chunk(rng, 10000)
    assert 5200 <= target <= 6800 and start is not None
    text = "one\r\n" + "x" * 241 + "\r\n  two  \r\n"
    assert clean_code(text) == "one\n  two"
    body, audit = pad_long_document(random.Random(1), "evidence", ["a" * 100, "b" * 100], 50, 0.5)
    assert body[audit["evidence_start_char"] : audit["evidence_end_char"]] == "evidence"
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--no-update-manifest", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    reconstructor = Reconstructor(
        args.root.resolve(), args.dataset.resolve(),
        update_manifest=not args.no_update_manifest,
    )
    report = reconstructor.run()
    print("REPORT " + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
