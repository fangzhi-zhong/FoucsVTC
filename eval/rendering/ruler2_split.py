# Prompt templates adapted from NVIDIA NeMo-Skills RULER v2:
# nemo_skills/dataset/ruler2/{prepare_niah,prepare_mmlu,prepare_qa}.py.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# FocusVTC adaptation: extract literal template boundaries for image rendering;
# dataset generation remains external. See ../RULER/LICENSE for license text.

"""Split a RULER v2 prompt into (header, haystack, tail).

RULER v2 stores the whole prompt in a single `question` field, built upstream as
`{context}\n\n{example}{problem}` where `context` itself is
`<instruction header>\n\n<needles>`. To render only the haystack as images we
recover the three parts from the literal templates in
nemo_skills/dataset/ruler2/{prepare_niah,prepare_mmlu,prepare_qa}.py.

The split is exact, not heuristic: `header + haystack + tail` must reconstruct
the original prompt byte for byte, and the tail marker must occur exactly once.
"""

import re

NIAH_SINGLE_HEADER = (
    "A special magic number is hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the number afterwards.\n"
)
NIAH_MULTI_HEADER = (
    "Some special magic numbers are hidden within the following text. "
    "Make sure to memorize them. I will quiz you about the numbers afterwards.\n"
)
MMLU_RETRIEVE_HEADER = (
    "Below are some questions. I will ask you to copy one of them. "
    "Please copy and paste the question you find.\n\n"
)
MMLU_NIAH_HEADER = (
    "Below are some questions. I will ask you to copy some of them. "
    "Please copy and paste the questions you find.\n\n"
)
MMLU_SOLVE_HEADER = (
    "Below are some questions. I will ask you to solve one of them. "
    "Please solve the question you find and make sure to put the answer "
    "(and only answer) inside \\boxed\\{\\}.\n\n"
)
QA_DOC_HEADER = (
    "Below are some documents. I will give you a text at the end. "
    "Please find the document index of the text. "
    "Only give me the index without any document contents.\n\n"
)
QA_QUESTION_HEADER = (
    "Below are some documents. I will give you a question at the end. "
    "Please find the index of the most relevant document that can help answer the question. "
    "Only give me the index without any document contents.\n\n"
)
QA_SOLVE_HEADER = (
    "Below are some documents. I will ask you to answer a question based on the documents. "
    "Please answer the question.\n\n"
)

FEWSHOT_PREAMBLE = r"\n\nHere are some examples to help you understand the task:\n\n"

# task -> (header literal, regex matching the start of the tail)
SPEC = {
    "mk_niah_basic": (NIAH_SINGLE_HEADER, r"\nWhat is the special magic number for "),
    "mv_niah_basic": (NIAH_MULTI_HEADER, r"\nWhat are all the special magic numbers for "),
    "mk_niah_easy": (MMLU_RETRIEVE_HEADER, r"\n\nPlease copy the Question \d+ from the context\."),
    # medium/hard use 5-shot, so the tail starts at the few-shot preamble
    "mk_niah_medium": (MMLU_SOLVE_HEADER, FEWSHOT_PREAMBLE),
    "mk_niah_hard": (MMLU_SOLVE_HEADER, FEWSHOT_PREAMBLE),
    "mv_niah_easy": (MMLU_NIAH_HEADER, r"\n\nPlease find and copy all the Question \d+ from the context\."),
    "mv_niah_medium": (MMLU_RETRIEVE_HEADER, r"\n\nPlease first copy all the Question \d+ from the context"),
    "mv_niah_hard": (MMLU_RETRIEVE_HEADER, r"\n\nPlease copy the \d+\w+ \(1 indexed\) Question \d+ from the context\."),
    "qa_basic": (QA_DOC_HEADER, r"\n\nText: "),
    "qa_easy": (QA_QUESTION_HEADER, r"\n\nQuestion: "),
    "qa_medium": (QA_SOLVE_HEADER, r"\n\nPlease first find and copy paste the documents relevant"),
    "qa_hard": (QA_SOLVE_HEADER, r"\n\nPlease answer the following question based on the documents\."),
}

_COMPILED = {task: (h, re.compile(t)) for task, (h, t) in SPEC.items()}


class SplitError(ValueError):
    pass


def split_prompt(task, prompt):
    """Return (header, haystack, tail); the three concatenate back to `prompt`."""
    try:
        header, tail_re = _COMPILED[task]
    except KeyError:
        raise SplitError(f"unknown task {task}") from None

    if not prompt.startswith(header):
        raise SplitError(f"{task}: header mismatch")

    matches = list(tail_re.finditer(prompt))
    if len(matches) != 1:
        raise SplitError(f"{task}: tail marker occurs {len(matches)} times, expected 1")

    cut = matches[0].start()
    if cut <= len(header):
        raise SplitError(f"{task}: empty haystack")

    haystack = prompt[len(header) : cut]
    tail = prompt[cut:]
    if header + haystack + tail != prompt:
        raise SplitError(f"{task}: reconstruction mismatch")
    return header, haystack, tail
