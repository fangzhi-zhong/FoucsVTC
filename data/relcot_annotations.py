"""Remap REL-CoT page references and boxes after document reflow.

Coordinates use the page-local [0, 1000] convention. These helpers preserve
conversation formats and annotation fields unrelated to page geometry.
"""

from __future__ import annotations

import copy
import json
import math
import re


_BOX_MARKER = re.compile(r"<\|(?:box_start|box_end|begin_of_box|end_of_box)\|>")
_BOX_CITATION = re.compile(
    r"\b(?i:Page)\s+(?P<page>\d+)\s*"
    r"(?P<start><\|box_start\|>|<\|begin_of_box\|>)\s*"
    r"(?P<bbox>\[[^\[\]\n]*\])\s*"
    r"(?P<end><\|box_end\|>|<\|end_of_box\|>)"
)
_PAGE_NUMBER = r"\d+(?:[ \t]*(?:[-–—]|(?i:to))[ \t]*\d+)?"
_PAGE_SEPARATOR = (
    r"(?:[ \t]*,[ \t]*(?:(?i:and|or)[ \t]+)?"
    r"|[ \t]+(?i:and|or)[ \t]+|[ \t]*&[ \t]*)"
)
_BARE_PAGE = re.compile(
    rf"\b(?i:Pages?)[ \t]*(?P<pages>{_PAGE_NUMBER}"
    rf"(?:{_PAGE_SEPARATOR}{_PAGE_NUMBER})*)(?!\w)"
)
_HUMAN_PAGES = re.compile(
    r"\A(?:Page[ \t]+\d+[ \t]*\r?\n[ \t]*<image>[ \t]*\r?\n)+"
)
_END_MARKER = {
    "<|box_start|>": "<|box_end|>",
    "<|begin_of_box|>": "<|end_of_box|>",
}


def _page(value, context):
    if isinstance(value, bool):
        raise ValueError(f"{context}: page must be a positive integer")
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{context}: page must be a positive integer")
    return value


def _location(value, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context}: location must be an object with page and bbox")
    page = _page(value.get("page"), context)
    bbox = value.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"{context}: bbox must contain four coordinates")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float))
        or not math.isfinite(x) or not 0 <= x <= 1000
        for x in bbox
    ):
        raise ValueError(f"{context}: bbox coordinates must be finite numbers in [0, 1000]")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError(f"{context}: bbox must have positive width and height")
    return {"page": page, "bbox": list(bbox)}


def _key(location):
    return location["page"], tuple(location["bbox"])


def _citations(text):
    """Parse all citations, rejecting any orphan or malformed box marker."""
    matches = list(_BOX_CITATION.finditer(text))
    covered_markers = set()
    result = []
    for match in matches:
        if _END_MARKER[match["start"]] != match["end"]:
            raise ValueError(f"mismatched box markers in citation: {match[0]!r}")
        try:
            bbox = json.loads(match["bbox"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid bbox in citation: {match[0]!r}") from exc
        location = _location({"page": match["page"], "bbox": bbox}, "box citation")
        covered_markers.update(
            marker.start() for marker in _BOX_MARKER.finditer(text, match.start(), match.end())
        )
        result.append((match, location))
    for marker in _BOX_MARKER.finditer(text):
        if marker.start() not in covered_markers:
            excerpt = text[max(0, marker.start() - 40):marker.end() + 80]
            raise ValueError(f"cannot parse box citation near {excerpt!r}")
    return result


def _turn_fields(turn):
    if not isinstance(turn, dict):
        raise ValueError("conversation turns must be objects")
    if "from" in turn:
        return str(turn["from"]).lower(), "value"
    return str(turn.get("role", "")).lower(), "content"


def _turn_text(turn, field):
    text = turn.get(field)
    if not isinstance(text, str):
        raise ValueError(f"conversation {field} must be a string")
    return text


def _reasoning_and_answer(text):
    """Keep the final answer, including the closing think tag, byte-for-byte."""
    end = text.find("</think>")
    return (text[:end], text[end:]) if end >= 0 else (text, "")


def _metadata_lists(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    for field in ("all_locations", "evidence_locations"):
        locations = metadata.get(field)
        if locations is None:
            continue
        if not isinstance(locations, list):
            raise ValueError(f"metadata.{field} must be a list")
        yield field, locations


def collect_locations(row, conversation):
    """Collect distinct metadata and assistant boxes in their original order."""
    locations = []
    seen = set()

    def add(location):
        key = _key(location)
        if key not in seen:
            seen.add(key)
            locations.append(location)

    for field, values in _metadata_lists(row.get("metadata") or {}):
        for value in values:
            add(_location(value, f"metadata.{field}"))
    for turn in conversation:
        role, field = _turn_fields(turn)
        if role in ("gpt", "assistant"):
            reasoning, _ = _reasoning_and_answer(_turn_text(turn, field))
            for _, location in _citations(reasoning):
                add(location)
    return locations


def _location_lookup(location_map):
    lookup = {}
    for entry in location_map:
        if not isinstance(entry, dict):
            raise ValueError("location_map entries must be objects")
        source = _location(entry.get("source"), "location_map source")
        targets = entry.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"location_map source {source} has no target locations")
        targets = [_location(target, "location_map target") for target in targets]
        key = _key(source)
        if key in lookup and lookup[key] != targets:
            raise ValueError(f"conflicting location mappings for {source}")
        lookup[key] = targets
    return lookup


def _targets(lookup, location):
    try:
        return lookup[_key(location)]
    except KeyError:
        raise ValueError(f"missing location mapping for {location}") from None


def _page_lookup(page_map, num_pages):
    lookup = {}
    for source, values in page_map.items():
        source = _page(source, "page_map source")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"page_map source {source} has no target pages")
        pages = list(dict.fromkeys(_page(value, "page_map target") for value in values))
        if any(page > num_pages for page in pages):
            raise ValueError(f"page_map target exceeds new page count {num_pages}")
        if source in lookup and lookup[source] != pages:
            raise ValueError(f"conflicting page mappings for Page {source}")
        lookup[source] = pages
    return lookup


def _rewrite_bare_pages(text, lookup):
    def replace(match):
        mapped = []
        for number in re.finditer(_PAGE_NUMBER, match["pages"]):
            bounds = re.split(r"[ \t]*(?:[-–—]|(?i:to))[ \t]*", number[0])
            first, last = int(bounds[0]), int(bounds[-1])
            if first > last:
                raise ValueError(f"descending page range in {match[0]!r}")
            for source in range(first, last + 1):
                if source not in lookup:
                    raise ValueError(f"missing page mapping for Page {source}")
                mapped.extend(lookup[source])
        pages = [str(page) for page in dict.fromkeys(mapped)]
        if len(pages) == 1:
            return "Page " + pages[0]
        return "Pages " + ", ".join(pages[:-1]) + " and " + pages[-1]

    return _BARE_PAGE.sub(replace, text)


def _rewrite_assistant(text, locations, pages):
    text, answer = _reasoning_and_answer(text)
    pieces = []
    offset = 0
    for match, source in _citations(text):
        # Rewrite only the original text between citations. New citation page
        # numbers never pass through the source-page mapping a second time.
        pieces.append(_rewrite_bare_pages(text[offset:match.start()], pages))
        citations = []
        for target in _targets(locations, source):
            bbox = json.dumps(target["bbox"], ensure_ascii=False)
            citations.append(f"Page {target['page']} {match['start']}{bbox}{match['end']}")
        pieces.append(" ".join(citations))
        offset = match.end()
    pieces.append(_rewrite_bare_pages(text[offset:], pages))
    return "".join(pieces) + answer


def rewrite_conversation(conversation, location_map, page_map, num_pages):
    """Return updated turns without changing question, reasoning, or answer text.

    Page references and their associated boxes are the only assistant text
    edits. Text after </think> is preserved verbatim. For older conversations
    without that delimiter, the whole assistant text is treated as reasoning.
    Initial numbered image blocks in user turns are rebuilt.
    """
    num_pages = _page(num_pages, "new page count")
    locations = _location_lookup(location_map)
    if any(target["page"] > num_pages for targets in locations.values() for target in targets):
        raise ValueError(f"location_map target exceeds new page count {num_pages}")
    pages = _page_lookup(page_map, num_pages)
    output = copy.deepcopy(conversation)
    prefix = "".join(f"Page {page}\n<image>\n" for page in range(1, num_pages + 1))
    for turn in output:
        role, field = _turn_fields(turn)
        if role in ("gpt", "assistant"):
            turn[field] = _rewrite_assistant(_turn_text(turn, field), locations, pages)
        elif role in ("human", "user"):
            text = _turn_text(turn, field)
            match = _HUMAN_PAGES.match(text)
            if match:
                if "<image>" in text[match.end():]:
                    raise ValueError("human image placeholders must all belong to the initial Page N blocks")
                turn[field] = prefix + text[match.end():]
            elif "<image>" in text:
                raise ValueError("human image placeholders require initial Page N\\n<image>\\n blocks")
    return output


def remap_metadata(metadata, location_map):
    """Expand mapped locations and remove geometry tied to the old images."""
    locations = _location_lookup(location_map)
    result = copy.deepcopy(metadata)
    for field, values in _metadata_lists(result):
        mapped = []
        for value in values:
            source = _location(value, f"metadata.{field}")
            for target in _targets(locations, source):
                item = copy.deepcopy(value)
                item.update(copy.deepcopy(target))
                item.pop("bbox_abs", None)
                item.pop("evidence_token_id", None)
                mapped.append(item)
        result[field] = mapped
    return result


__all__ = ["collect_locations", "rewrite_conversation", "remap_metadata"]
