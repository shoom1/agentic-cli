"""Helpers for the memory store's tests: writing files as a person would."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

T0 = datetime(2026, 11, 8, 14, 3, 22, tzinfo=timezone.utc)


class Clock:
    """A ``now`` the test moves by hand."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def memory_text(
    memory_id: str = "4f2a9c1e",
    description: str = "Prefers pytest fixtures over unittest classes",
    content: str = "New tests use pytest fixtures.",
    *,
    type: str = "feedback",
    source: str = "agent",
    created: str = "2026-11-08T14:03:22Z",
    updated: str = "2026-11-08T14:03:22Z",
    extra: str = "",
) -> str:
    """A memory file's text, the way the store writes one (plus ``extra`` lines)."""
    return (
        f"---\nid: {memory_id}\ndescription: {description}\ntype: {type}\n"
        f"source: {source}\ncreated: {created}\nupdated: {updated}\n{extra}---\n{content}\n"
    )


def write_memory(folder: Path, name: str, text: str) -> Path:
    """Write ``text`` as ``folder/name`` (the folder created 0700)."""
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path
