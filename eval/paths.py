"""Portable locations shared by benchmark clients."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def data_path(relative: str) -> Path:
    return Path(os.environ.get("FOCUSVTC_DATA_ROOT", ROOT / "datasets")).expanduser().resolve() / relative


def resolve_images(row: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    """Resolve manifest-relative image paths before constructing file URLs."""
    row = dict(row)
    for field in ("images", "image"):
        if field in row:
            row[field] = [str((manifest_dir / Path(image)).resolve()) for image in row[field]]
    return row
