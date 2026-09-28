"""Reuse lmms-eval's official OCRBench prompt and scoring implementation."""

from lmms_eval.tasks.ocrbench.utils import (  # noqa: F401
    ocrbench_aggregate_accuracy,
    ocrbench_doc_to_text,
    ocrbench_doc_to_visual,
    ocrbench_process_results,
)
