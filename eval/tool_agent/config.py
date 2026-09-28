"""JSON inheritance with environment expansion and release-relative paths."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = {"model", "tool_schema_path", "grpo_root", "chat_template_path", "data_root", "results_dir", "allowed_local_media_path", "vtcbench_root"}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        merged[key] = _merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def _read(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular config inheritance: {path}")
    seen = seen | {path}
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    return _merge(_read(path.parent / parent, seen), config) if parent else config


def load_config(path: str | Path) -> dict[str, Any]:
    defaults = {
        "FOCUSVTC_ROOT": str(ROOT),
        "FOCUSVTC_DATA_ROOT": str(ROOT / "datasets"),
        "FOCUSVTC_MODEL": str(ROOT / "models/FocusVTC"),
        "FOCUSVTC_OUTPUT_ROOT": str(ROOT / "outputs"),
        "VTCBENCH_ROOT": str(ROOT / "external/VTCBench"),
    }
    substitutions = {key: os.environ.get(key, value) for key, value in defaults.items()}

    def expand(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: expand(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        if not isinstance(value, str):
            return value
        for name, replacement in substitutions.items():
            value = value.replace("${" + name + "}", replacement)
        value = os.path.expandvars(os.path.expanduser(value))
        if "${" in value:
            raise ValueError(f"unresolved environment variable in {key}: {value}")
        if key in PATH_KEYS:
            return str((ROOT / value).resolve())
        return value

    return expand(_read(Path(path), set()))
