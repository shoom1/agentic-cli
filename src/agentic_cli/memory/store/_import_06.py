"""0.6's ``memories.json``, imported once into the user scope (spec §10).

0.6 kept one store per app and OS user in the same folder the user scope now
uses. Each memory the model saved becomes a live memory (its id kept, so a
second run adds nothing); each one the extractor wrote, which nobody ever
reviewed, becomes a proposal. Then the old files move into ``pre-0.7.0/``.
A 0.6 session still running rewrites ``memories.json`` with every item it
holds (released 0.6 takes no lock, and rewrites it on every change, a search
included), so what such a session writes later is imported at the next open
with new ids only: an id found in any file already in ``pre-0.7.0/`` was
imported before, and a memory deleted since stays deleted. The ``.lock`` the
import takes orders two 0.7 processes.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.memory._core.io import read_regular_file
from agentic_cli.memory._core.log import get_logger
from agentic_cli.memory.store._files import valid_id
from agentic_cli.memory.store._scope import ScopeFolder
from agentic_cli.memory.store.models import Candidate, MemoryBusy, MemoryStoreError, Problem

if TYPE_CHECKING:
    from agentic_cli.memory.store.store import MemoryStore

logger = get_logger("agentic_cli.memory.store")

LEGACY_DIR = "pre-0.7.0"
OLD_FILES = ("memories.json", "memories_embeddings.json", "embedding_model.json")
DESCRIPTION_CUT = 120


class _Unreadable(Exception):
    """``memories.json`` cannot be read as 0.6's store."""


def needs_import(user_dir: Path) -> bool:
    """Whether 0.6's ``memories.json`` is in the user scope's folder."""
    return os.path.lexists(user_dir / "memories.json")


def run(store: "MemoryStore", folder: ScopeFolder) -> list[Problem]:
    """Import ``memories.json``; the problems it leaves (an unreadable file)."""
    path = folder.path / "memories.json"
    try:
        with folder.lock():
            return _import(store, folder, path)
    except MemoryBusy:
        logger.info("memory_import_waiting", path=str(path))
        return []


def _entries(path: Path) -> list[Any]:
    try:
        data = read_regular_file(path)
    except OSError as e:
        raise _Unreadable(str(e)) from None
    try:
        parsed = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise _Unreadable(f"not JSON: {e}") from None
    items = parsed.get("items") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        raise _Unreadable('neither a list of memories nor {"items": [...]}')
    return items


def _imported_ids(folder: Path) -> set[str]:
    """The ids in the files already moved to ``pre-0.7.0/``: imported before, whatever became of them.

    A file that cannot be read says nothing.
    """
    legacy = folder / LEGACY_DIR
    try:
        if not stat.S_ISDIR(os.lstat(legacy).st_mode):
            return set()
        names = sorted(item.name for item in os.scandir(legacy) if item.name.startswith("memories.json"))
    except OSError:
        return set()
    ids: set[str] = set()
    for name in names:
        try:
            entries = _entries(legacy / name)
        except _Unreadable:
            continue
        ids.update(entry["id"] for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str))
    return ids


def _description(content: str) -> str:
    line = next((part.strip() for part in content.splitlines() if part.strip()), "")
    if len(line) <= DESCRIPTION_CUT:
        return line
    cut = line[: DESCRIPTION_CUT - 1]
    space = cut.rfind(" ")
    if space > DESCRIPTION_CUT // 3:
        cut = cut[:space]
    return cut.rstrip() + "…"


def _utc(value: Any) -> datetime | None:
    """0.6's naive local time, in UTC to the second; None when it is not one."""
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.astimezone()  # 0.6 wrote local time
    return moment.astimezone(timezone.utc).replace(microsecond=0)


def _import(store: "MemoryStore", folder: ScopeFolder, path: Path) -> list[Problem]:
    try:
        entries = _entries(path)
    except _Unreadable as e:
        logger.warning("memory_import_unreadable", path=str(path), reason=str(e))
        return [Problem(path, "import_unreadable", str(e))]
    counts = {"memories": 0, "proposals": 0, "skipped": 0}
    imported = _imported_ids(folder.path)
    for entry in entries:
        outcome = _one(store, folder, entry, imported)
        if outcome in counts:
            counts[outcome] += 1
    _move_aside(folder.path)
    logger.info("memory_imported", path=str(path), **counts)
    return []


def _one(store: "MemoryStore", folder: ScopeFolder, entry: Any, imported: set[str]) -> str:
    """Import one entry: ``memories``, ``proposals``, ``skipped``, or another word."""
    if not isinstance(entry, dict) or entry.get("archived"):
        return "archived"
    content = entry.get("content")
    old_id = entry.get("id")
    if isinstance(old_id, str) and old_id in imported:
        return "already"
    if not isinstance(content, str) or not content.strip():
        logger.info("memory_import_item_skipped", id=str(old_id), reason="no content")
        return "skipped"
    tags = entry.get("tags")
    tags = [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []
    text = content.strip() + (f"\n\nTags: {', '.join(tags)}" if tags else "")
    description = _description(content)
    if "auto-extracted" in tags:
        candidate = Candidate("new", None, description, text, "user", "user")
        reason = store._candidate_problem(candidate)
        if reason is not None:
            logger.info("memory_import_item_skipped", id=str(old_id), reason=reason)
            return "skipped"
        proposal = store._propose_locked(folder, candidate, None, None, "imported")
        return "proposals" if proposal is not None else "repeated"
    memory_id = old_id if valid_id(old_id) else None
    if memory_id is not None and memory_id in store._entries()[0]:
        return "already"
    try:
        description, text = store._checked(description, text, "user")
        store._screened(description, text)
        created = _utc(entry.get("created_at"))
        store._save_locked(
            folder, description, text, "user", "imported", None,
            memory_id=memory_id, created=created, updated=_utc(entry.get("updated_at")) or created,
        )
    except MemoryStoreError as e:
        logger.info("memory_import_item_skipped", id=str(old_id), reason=str(e))
        return "skipped"
    return "memories"


def _move_aside(folder: Path) -> None:
    legacy = folder / LEGACY_DIR
    try:
        mode = os.lstat(legacy).st_mode
    except FileNotFoundError:
        legacy.mkdir(mode=0o700)
        mode = os.lstat(legacy).st_mode
    if not stat.S_ISDIR(mode):
        logger.warning("memory_legacy_folder_refused", path=str(legacy))
        return
    names = list(OLD_FILES) + sorted(
        item.name for item in os.scandir(folder) if item.name.startswith("memories.json.unreadable-")
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    moved: list[str] = []
    for name in names:
        source = folder / name
        if not os.path.lexists(source):
            continue
        destination = legacy / name
        if os.path.lexists(destination):
            destination = legacy / f"{name}.{stamp}"
        os.replace(source, destination)
        moved.append(name)
    if moved:
        logger.info("memory_legacy_files_moved", path=str(legacy), files=moved)
