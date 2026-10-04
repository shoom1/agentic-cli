"""Atomic writes for the memory package's own files.

The files are private (0600): they hold the package's state, not files a user
edits. Same behavior as ``agentic_cli.file_utils`` without ``preserve_mode``;
the package keeps its own copy so it imports nothing from agentic-cli.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to a temp file beside ``path``, fsync, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write ``data`` as JSON, atomically."""
    _atomic_write(path, json.dumps(data, indent=indent))


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content``, atomically."""
    _atomic_write(path, content)
