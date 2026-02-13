"""Shared persistence utilities."""

import json
from pathlib import Path
from typing import Any


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Replaces any character that isn't alphanumeric, hyphen, or underscore with underscore.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON data to a file atomically.

    Writes to a temporary file first, then renames to the target path.
    This prevents data corruption if the process crashes mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=indent))
    tmp_path.replace(path)


def atomic_write_text(path: Path, content: str) -> None:
    """Write text to a file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content)
    tmp_path.replace(path)
