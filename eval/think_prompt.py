#!/usr/bin/env python3
"""Reshape evaluation prompts into the system+thinking VTC SFT prompt shape.

The fine-tune learned to answer as `<think>...reasoning with <|box_start|> page
coordinates...</think>\\nanswer`, but it only reaches that mode when the prompt
matches training closely. Two details of lmms-engine's pipeline are invisible to
a normal chat client and were missing from the harnesses:

  * For checkpoints trained with ``add_system_prompt: true``, exactly one system
    turn carrying ``TRAIN_SYSTEM`` must precede the user turn.
  * `TrainUtilities.convert_llava_to_open` calls `.strip()` on every text run
    between `<image>` placeholders, so the rendered prompt has no whitespace
    around the vision blocks.

"""

from __future__ import annotations

import re

TRAIN_SYSTEM = "You are a helpful assistant"
THINK_OPEN = "<think>\n"
THINK_CLOSE = "</think>"

BOX_OPEN = "<|begin_of_box|>"
BOX_CLOSE = "<|end_of_box|>"
_ANSWER_TAGS = re.compile(r"</?answer>")

CONTINUE_ASSISTANT = {"continue_final_message": True, "add_generation_prompt": False}


def strip_text_parts(content):
    """Mirror convert_llava_to_open: strip every text run, drop the empty ones."""
    if isinstance(content, str):
        return content.strip()
    out = []
    for part in content:
        if part.get("type") == "text":
            text = part["text"].strip()
            if text:
                out.append({"type": "text", "text": text})
        else:
            out.append(part)
    return out


def apply_think(messages, extra_body=None):
    """Return (messages, extra_body) rewritten into the SFT training shape.

    The assistant turn is left open on `<think>\\n` so the model continues inside
    its trained reasoning block instead of falling back to the base model's
    "Based on the provided text..." prior.
    """
    # Most benchmark builders already add their system turn.  Older versions of
    # this helper unconditionally prepended another one, yielding two system
    # messages.  Preserve the caller's first system prompt and supply the
    # training default only when none was provided.
    out = []
    saw_system = False
    for msg in messages:
        if msg["role"] == "system":
            if saw_system:
                continue
            saw_system = True
        out.append({**msg, "content": strip_text_parts(msg["content"])})
    if not saw_system:
        out.insert(0, {"role": "system", "content": TRAIN_SYSTEM})
    out.append({"role": "assistant", "content": THINK_OPEN})
    return out, {**(extra_body or {}), **CONTINUE_ASSISTANT}


def unwrap_answer(text):
    """Reduce GLM-4.1V's structured reply to the span it wants graded.

    GLM-4.1V-9B-Base answers as
    `<answer>... <|begin_of_box|>X<|end_of_box|> ...</answer>`, where the box
    marks the final answer and everything around it is restatement. Grading the
    wrapper would dilute every F1/ROUGE metric and leave literal box tokens in
    exact-match comparisons, so the box content wins when present and the
    `<answer>` tags are dropped otherwise.

    A no-op for every other checkpoint here -- neither Qwen3-VL nor Glyph has
    ever emitted these markers.
    """
    if BOX_OPEN in text:
        return text.split(BOX_OPEN, 1)[1].split(BOX_CLOSE, 1)[0].strip()
    return _ANSWER_TAGS.sub("", text).strip()


def strip_think(text):
    """Return the answer that follows the reasoning block.

    Under `--think` the generation starts *inside* `<think>` because of the
    prefill; models that reason on their own open the block themselves. Either
    way a well-formed response closes it and the answer is whatever comes after.
    Responses that never close it are returned unchanged rather than dropped, so
    a degenerate run shows up as a low score instead of an empty file.
    """
    if THINK_CLOSE in text:
        text = text.rsplit(THINK_CLOSE, 1)[1]
    return unwrap_answer(text)
