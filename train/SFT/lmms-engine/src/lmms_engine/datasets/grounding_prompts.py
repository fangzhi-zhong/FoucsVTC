"""Weak, varied visual-grounding hints for Qwen3.5 SFT prompts.

These deliberately describe the desired behavior semantically without naming a
special bbox token or prescribing a rigid output grammar.  The answer labels
remain the source of supervision for the actual evidence spans.
"""

GROUNDING_PROMPTS = (
    "",
    "Use the images to answer the question. When useful, mention the page and visual evidence that supports your reasoning.",
    "Answer from the provided pages and refer to the relevant visual evidence when it helps.",
    "Inspect the pages carefully and briefly note where supporting evidence appears when relevant.",
    "Base your answer on the images, mentioning the relevant page or region if it clarifies your reasoning.",
    "Use the visual information in the pages as evidence and identify its location when useful.",
    "Answer the question using the page images; a short reference to the supporting location is helpful when needed.",
    "Read the provided pages carefully and point to the supporting evidence when that is useful.",
    "Ground your reasoning in the images and mention the page containing important evidence when appropriate.",
    "Use page-level visual evidence to support the answer whenever it is relevant.",
    "Look at the images closely and briefly indicate where the answer is supported if helpful.",
    "Answer using the document images, with a concise reference to relevant visual evidence when needed.",
    "When the answer depends on a particular part of a page, mention that supporting location briefly.",
    "Rely on the supplied page images and identify useful evidence locations as part of your reasoning.",
    "Check the visual context before answering and mention the supporting page when it adds clarity.",
    "Use the document pages to reason about the answer; refer to the relevant evidence when useful.",
    "Answer carefully from the images and note the page or area that supports the conclusion when appropriate.",
    "Consider the visual evidence in the pages and briefly say where it appears if that helps explain the answer.",
    "Read the pages as evidence for the question and include a short location reference when relevant.",
    "Use the page images to verify the answer, mentioning the supporting evidence only when it is helpful.",
    "Answer based on the supplied visual context and point out the relevant page or evidence location when needed.",
)
