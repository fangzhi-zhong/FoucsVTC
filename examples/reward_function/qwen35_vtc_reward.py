"""Local rule-based reward for Qwen3.5-VL VTC zoom GRPO."""

from __future__ import annotations

import json
import re
import ast
import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


ANSWER_WEIGHT = 0.8
FORMAT_WEIGHT = 0.2
TOOL_WEIGHT = 1.0
EXCESS_CALL_WEIGHT = 0.2
INVALID_CALL_WEIGHT = 0.1
# Evidence annotations can be tighter than the useful visual context. Expand
# them around their center before IoU matching. Coordinates are normalized to
# [0, 1000], so 80 is an 8% page-side minimum.
EVIDENCE_EXPANSION = 1.20
MIN_EVIDENCE_WIDTH = 80.0
MIN_EVIDENCE_HEIGHT = 80.0


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _answers(ground_truth: Any) -> list[str]:
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("ground_truth", ground_truth.get("answer", []))
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()
    if ground_truth is None:
        return []
    if isinstance(ground_truth, (list, tuple)):
        return [str(item).strip() for item in ground_truth if str(item).strip()]
    return [str(ground_truth).strip()] if str(ground_truth).strip() else []


def _metadata(extra_info: Any) -> dict:
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            return {}
    return extra_info if isinstance(extra_info, dict) else {}


def _evidence_pairs(extra_info: Any) -> list[tuple[int, list[float]]]:
    extra_info = _metadata(extra_info)

    layout = extra_info.get("evidence_bboxes_by_dpi", {})
    if isinstance(layout, dict):
        candidates = []
        for key, value in layout.items():
            digits = "".join(ch for ch in str(key) if ch.isdigit())
            candidates.append((int(digits or 0), value))
        for _, value in sorted(candidates, key=lambda item: item[0], reverse=True):
            pairs = _unique_valid_pairs(_pairs(value), extra_info.get("num_pages"))
            if pairs:
                return pairs
    return _unique_valid_pairs(_pairs(extra_info.get("evidence_locations", [])), extra_info.get("num_pages"))


def _pairs(layout: Any) -> list[tuple[int, list[float]]]:
    if isinstance(layout, np.ndarray):
        layout = layout.tolist()
    pairs = []
    if isinstance(layout, dict):
        if "page" in layout or "page_id" in layout:
            page = layout.get("page", layout.get("page_id"))
            boxes = layout.get("bbox_2d", layout.get("bbox", layout.get("bboxes", [])))
            pairs.extend(_pairs({str(page): boxes}))
        else:
            for page, boxes in layout.items():
                if isinstance(boxes, np.ndarray):
                    boxes = boxes.tolist()
                if isinstance(boxes, (list, tuple)) and len(boxes) == 4 and all(
                    isinstance(x, (int, float)) for x in boxes
                ):
                    boxes = [boxes]
                if not isinstance(boxes, (list, tuple)):
                    continue
                for box in boxes:
                    if isinstance(box, np.ndarray):
                        box = box.tolist()
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        try:
                            pairs.append((int(page), [float(x) for x in box]))
                        except (TypeError, ValueError, OverflowError):
                            continue
    elif isinstance(layout, (list, tuple)):
        for item in layout:
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if isinstance(item, dict):
                pairs.extend(_pairs(item))
            elif isinstance(item, (list, tuple)) and len(item) == 5:
                try:
                    pairs.append((int(item[0]), [float(x) for x in item[1:]]))
                except (TypeError, ValueError, OverflowError):
                    continue
    return pairs


def _valid_box(box: Any) -> bool:
    return (
        isinstance(box, (list, tuple)) and len(box) == 4
        and all(not isinstance(x, bool) and isinstance(x, (int, float))
                and math.isfinite(x) and 0 <= x <= 1000 for x in box)
        and box[0] < box[2] and box[1] < box[3]
    )


def _unique_valid_pairs(pairs, num_pages=None):
    # The 50K shard has regions, not semantic evidence IDs. Count distinct
    # (page, box) annotations; do not inflate the allowance with DPI copies.
    seen = set()
    result = []
    for page, box in pairs:
        if page < 1 or (num_pages is not None and page > int(num_pages)) or not _valid_box(box):
            continue
        box = _expand_evidence_box(box)
        key = (page, tuple(box))
        if key not in seen:
            seen.add(key)
            result.append((page, box))
    return result


def _expand_evidence_box(box: list[float]) -> list[float]:
    """Expand an evidence box by 20%, enforcing a minimum 80x80 box.

    Expansion is centered and then shifted, when needed, so the result stays
    within the normalized page. A box already touching an edge may therefore
    expand mostly toward the interior rather than symmetrically.
    """
    x1, y1, x2, y2 = map(float, box)
    width = min(1000.0, max(MIN_EVIDENCE_WIDTH, (x2 - x1) * EVIDENCE_EXPANSION))
    height = min(1000.0, max(MIN_EVIDENCE_HEIGHT, (y2 - y1) * EVIDENCE_EXPANSION))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left, right = cx - width / 2.0, cx + width / 2.0
    top, bottom = cy - height / 2.0, cy + height / 2.0
    if left < 0:
        right -= left
        left = 0.0
    if right > 1000:
        left -= right - 1000.0
        right = 1000.0
    if top < 0:
        bottom -= top
        top = 0.0
    if bottom > 1000:
        top -= bottom - 1000.0
        bottom = 1000.0
    return [max(0.0, left), max(0.0, top), min(1000.0, right), min(1000.0, bottom)]


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _tool_calls(solution: str) -> tuple[list[dict], int]:
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", solution, flags=re.DOTALL)
    calls = []
    # Incomplete calls still consume an attempt, including a truncated tail.
    invalid = max(0, solution.count("<tool_call>") - len(blocks))
    for block in blocks:
        try:
            try:
                payload = json.loads(block.strip())
            except json.JSONDecodeError:
                function_match = re.search(r"<function\s*=\s*([A-Za-z0-9_.:-]+)\s*>", block)
                if function_match is None:
                    raise ValueError("unknown tool-call format")
                arguments = {}
                for match in re.finditer(
                    r"<parameter\s*=\s*([A-Za-z0-9_.:-]+)\s*>\s*(.*?)\s*</parameter>",
                    block,
                    flags=re.DOTALL,
                ):
                    value = match.group(2).strip()
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = ast.literal_eval(value)
                    arguments[match.group(1)] = value
                payload = {"name": function_match.group(1), "arguments": arguments}
            if not isinstance(payload, dict):
                raise ValueError("tool call is not an object")
            calls.append(payload)
        except Exception:
            invalid += 1
    return calls, invalid


_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
_CHAT_START = re.compile(r"<\|im_start\|>(assistant|user|tool|system)\n")


def _assistant_segments(solution: str) -> list[str]:
    """Exclude observations and repeated user instructions from all scoring."""
    segments = []
    role, start = "assistant", 0  # The first generation continues the prompt.
    for match in _CHAT_START.finditer(solution):
        if role == "assistant":
            segments.append(solution[start:match.start()])
        role, start = match.group(1), match.end()
    # Keep an empty final segment for trajectories ending on an observation.
    segments.append(solution[start:] if role == "assistant" else "")
    return segments


def _answer_text(solution: str) -> str:
    """Only a terminal answer in the last assistant turn is an answer."""
    final = _TRAILING_CHAT_END.sub("", _assistant_segments(solution)[-1]).strip()
    # A quoted answer inside thinking or a tool payload is not a final answer.
    if "</think>" in final:
        final = final.rsplit("</think>", 1)[-1]
    if "<think>" in final:
        return ""
    matches = list(_ANSWER_BLOCK.finditer(final))
    if not matches or final[matches[-1].end():].strip():
        return ""
    prefix = final[:matches[-1].start()]
    if prefix.rfind("<tool_call>") > prefix.rfind("</tool_call>"):
        return ""
    return matches[-1].group(1).strip()


def _matching_answer_count(answer_text: str, gold: list[str]) -> int:
    """Count one-to-one exact matches under the existing normalization.

    Each occurrence of a normalized gold answer is a possible prediction span.
    A span can be assigned to at most one gold answer, and selected spans may
    not overlap.  The dynamic program therefore handles both repeated gold
    answers and overlapping answers without counting one prediction twice.
    """

    normalized_answer = _normalise(answer_text)
    if not normalized_answer or not gold:
        return 0

    # Preserve word boundaries while retaining the legacy punctuation/space
    # normalization (e.g. 2,521 == 2521). "4" must not match "42", and "no"
    # must not match "unknown". Candidate spans can still cover several words.
    starts, ends = set(), set()
    offset = 0
    for token in re.findall(r"[a-z0-9]+", answer_text.lower()):
        starts.add(offset)
        offset += len(token)
        ends.add(offset)
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for gold_index, answer in enumerate(gold):
        target = _normalise(answer)
        if not target:
            continue
        start = normalized_answer.find(target)
        while start >= 0:
            if start in starts and start + len(target) in ends:
                candidates[start].append((start + len(target), gold_index))
            start = normalized_answer.find(target, start + 1)

    if not candidates:
        return 0

    # dp[position][mask] is the best number of non-overlapping matches after
    # consuming the normalized answer prefix ending at ``position``.  The
    # number of gold answers in VTC rows is small (typically <= 10).
    states: list[dict[int, int]] = [dict() for _ in range(len(normalized_answer) + 1)]
    states[0][0] = 0
    for position in range(len(normalized_answer)):
        if not states[position]:
            continue
        next_states = states[position + 1]
        for mask, count in states[position].items():
            if count > next_states.get(mask, -1):
                next_states[mask] = count
            for end, gold_index in candidates.get(position, ()):
                bit = 1 << gold_index
                if mask & bit:
                    continue
                destination = states[end]
                new_mask = mask | bit
                if count + 1 > destination.get(new_mask, -1):
                    destination[new_mask] = count + 1

    return max((count for state in states for count in state.values()), default=0)


_THINK_PREFILL = re.compile(r"<\|im_start\|>assistant\n<think>\s*\Z")
_TRAILING_CHAT_END = re.compile(r"(?:\s*<\|(?:im_end|endoftext)\|>)+\s*\Z")


def _format_reward(solution: str, prompt_str: str | None = None) -> float:
    """Return 1 for the requested format and 0 for a format violation."""

    # Generation continues after the prompt's opening <think>. Account for
    # that one tag only when the actual assistant prefix confirms it; never
    # count tag examples in user instructions or blindly repair a response.
    prefilled_think = int(bool(_THINK_PREFILL.search(prompt_str or "")))
    # The reward manager decodes EOS tokens. They terminate the message and
    # are not text after the answer. Strip only recognized terminal markers.
    solution = _TRAILING_CHAT_END.sub("", solution)
    if solution.count("<think>") + prefilled_think != solution.count("</think>"):
        return 0.0

    matches = list(_ANSWER_BLOCK.finditer(solution))
    if not matches:
        return 0.0

    # Keep the same "last answer" parsing convention used by the VTC reward
    # path, and require that final answer to contain non-whitespace content.
    answer_match = matches[-1]
    if not answer_match.group(1).strip():
        return 0.0

    # Nothing other than whitespace may follow the closing answer tag.
    if solution[answer_match.end() :].strip():
        return 0.0
    return 1.0


def _dpi(extra_info: Any) -> float:
    """Read the sample DPI, defaulting legacy 72-DPI shards to 72."""

    metadata = extra_info if isinstance(extra_info, dict) else {}
    for source in (metadata, metadata.get("metadata", {}) if isinstance(metadata, dict) else {}):
        if not isinstance(source, dict):
            continue
        for key in ("dpi", "image_dpi"):
            value = source.get(key)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return 72.0


def _tool_quality(calls, evidence, num_pages=None):
    """Maximum one-to-one IoU matching; redundant calls cannot reuse evidence.

    Boxes covering <= half a page retain their IoU. Above half a page, a
    linear area factor decays to zero at a full-page crop. Normalize by the
    smaller of the number of attempts and evidence regions in compute_score,
    so answering correctly with fewer calls does not require reading every
    annotation just to earn the tool bonus.
    """
    valid = []
    invalid = 0
    for call in calls:
        args = call.get("arguments")
        if not isinstance(args, dict):
            invalid += 1
            continue
        page, box = args.get("page"), args.get("bbox_2d")
        if (call.get("name") != "zoom_region" or isinstance(page, bool)
                or not isinstance(page, int) or page < 1
                or (num_pages is not None and page > int(num_pages))
                or not isinstance(box, list) or not _valid_box(box)):
            invalid += 1
            continue
        valid.append((page, box))
    if not valid or not evidence:
        return 0.0, 0.0, invalid, len(valid), 0

    ious = np.array([
        [_iou(box, target) if page == target_page else 0.0 for target_page, target in evidence]
        for page, box in valid
    ])
    area_factors = np.array([
        min(1.0, 2.0 * (1.0 - (box[2] - box[0]) * (box[3] - box[1]) / 1_000_000.0))
        for _, box in valid
    ])
    qualities = ious * area_factors[:, None]
    rows, columns = linear_sum_assignment(qualities, maximize=True)
    matched = qualities[rows, columns]
    return float(matched.sum()), float(ious.max()), invalid, len(valid), int((matched > 0).sum())


def compute_score(data_source, solution_str, ground_truth, extra_info=None, prompt_str=None, **kwargs):
    del data_source, kwargs
    solution = str(solution_str or "")
    metadata = _metadata(extra_info)
    gold = _answers(ground_truth)

    answer_text = _answer_text(solution)
    matched_answers = _matching_answer_count(answer_text, gold)
    acc_reward = matched_answers / len(gold) if gold else 0.0

    # A bounded token-overlap fallback is useful for non-numeric RULER answers.
    partial = 0.0
    if matched_answers == 0 and gold:
        solution_tokens = set(re.findall(r"[a-z0-9]+", answer_text.lower()))
        partial = max(
            (len(solution_tokens & set(re.findall(r"[a-z0-9]+", answer.lower()))) / max(1, len(set(re.findall(r"[a-z0-9]+", answer.lower())))) for answer in gold),
            default=0.0,
        )

    assistant_text = "\n".join(_assistant_segments(solution))
    format_reward = _format_reward(assistant_text, prompt_str=prompt_str) if answer_text else 0.0

    calls, parse_errors = _tool_calls(assistant_text)
    evidence = _evidence_pairs(metadata)
    iou_sum, best_iou, invalid_payloads, valid_calls, matched_evidence = _tool_quality(
        calls, evidence, metadata.get("num_pages")
    )
    num_calls = len(calls) + parse_errors
    invalid_calls = parse_errors + invalid_payloads
    evidence_count = len(evidence)
    iou_reward = iou_sum / min(num_calls, evidence_count) if num_calls and evidence_count else 0.0
    call_efficiency = min(1.0, evidence_count / num_calls) if num_calls else 0.0
    excess_call_ratio = max(0, num_calls - evidence_count) / max(1, num_calls)
    invalid_call_ratio = invalid_calls / max(1, num_calls)

    actual_tool_called = float(valid_calls > 0)
    answer_correct = float(bool(answer_text) and bool(gold) and matched_answers == len(gold))
    dpi = _dpi(metadata)
    dpi_factor = max(0.0, min(1.0, (144.0 - dpi) / (144.0 - 72.0))) ** 2
    # Only full final-answer correctness unlocks positive tool reward.
    # Excess/invalid penalties still apply to incorrect and truncated runs.
    tool_reward = answer_correct * dpi_factor * iou_reward * call_efficiency
    tool_bonus = TOOL_WEIGHT * tool_reward
    tool_penalty = EXCESS_CALL_WEIGHT * excess_call_ratio + INVALID_CALL_WEIGHT * invalid_call_ratio
    score = ANSWER_WEIGHT * acc_reward + FORMAT_WEIGHT * format_reward + tool_bonus - tool_penalty
    return {
        "score": score,
        "acc_reward": acc_reward,
        "answer_exact": acc_reward,
        "answer_matches": matched_answers,
        "answer_count": len(gold),
        "answer_partial": partial,
        "answer_correct": answer_correct,
        "final_answer_present": float(bool(answer_text)),
        "format_reward": format_reward,
        "actual_tool_called": actual_tool_called,
        "dpi": dpi,
        "dpi_factor": dpi_factor,
        "tool_reward": tool_reward,
        "tool_bonus": tool_bonus,
        "tool_penalty": tool_penalty,
        "evidence_count": evidence_count,
        "evidence_expansion": EVIDENCE_EXPANSION,
        "min_evidence_width": MIN_EVIDENCE_WIDTH,
        "min_evidence_height": MIN_EVIDENCE_HEIGHT,
        "matched_evidence": matched_evidence,
        "iou_reward": iou_reward,
        "call_efficiency": call_efficiency,
        "excess_call_ratio": excess_call_ratio,
        "best_iou": best_iou,
        "tool_calls": num_calls,
        "valid_tool_calls": valid_calls,
        "invalid_tool_calls": invalid_calls,
    }
