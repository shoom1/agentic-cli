"""Which embedding model built an index, recorded beside it.

A knowledge base keeps ``embeddings/model.json`` and the memory store
``embedding_model.json``: the model's name and its vector dimension. On
open, each compares the record with the model it has now and re-embeds what
another model built. 0.6.3 ignores both files.

A project can ship its knowledge base, so a record is untrusted input: it
is read only from a small regular file (never through a symlink, never
blocking on a FIFO), and anything but a non-empty model name and a positive
integer dimension counts as no record.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_cli.memory._core.io import atomic_write_json

# Two short fields; anything bigger is not a record.
_MAX_RECORD_BYTES = 4096


@dataclass(frozen=True)
class ModelRecord:
    """An embedding model's name and the dimension of its vectors."""

    model_name: str
    embedding_dim: int


def _valid(name: Any, dim: Any) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and isinstance(dim, int)
        and not isinstance(dim, bool)
        and dim > 0
    )


def _read_small_file(path: Path) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        data = os.read(fd, _MAX_RECORD_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    return None if len(data) > _MAX_RECORD_BYTES else data


def read_model_record(path: Path) -> ModelRecord | None:
    """The record at ``path``, or None when it is missing or not a valid record."""
    data = _read_small_file(path)
    if data is None:
        return None
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    name, dim = parsed.get("model_name"), parsed.get("embedding_dim")
    return ModelRecord(name, dim) if _valid(name, dim) else None


def write_model_record(path: Path, record: ModelRecord) -> None:
    """Write ``record`` to ``path``, atomically."""
    atomic_write_json(path, {"model_name": record.model_name, "embedding_dim": record.embedding_dim})


def record_for(service: Any) -> ModelRecord | None:
    """The record ``service`` would write, or None if it does not name its model.

    The memory store takes any object with ``embed_text`` and ``embed_batch``;
    one without ``model_name`` and ``embedding_dim`` keeps no record.
    """
    name = getattr(service, "model_name", None)
    try:
        dim = service.embedding_dim
    except AttributeError:
        return None
    return ModelRecord(name, dim) if _valid(name, dim) else None
