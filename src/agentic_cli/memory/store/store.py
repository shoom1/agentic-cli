"""The memory store: markdown files in a user scope and a project scope (spec §6).

Each memory is one file (``_files``) in its scope's folder (``_scope``); the
files are the store's only state, read again as they change, so a person's
edit is seen at the next call. A change holds the scope's lock, keeps the
previous version in ``.history`` and replaces files atomically. Every call is
also serialized within the process (tools run on worker threads).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentic_cli.memory._core.log import get_logger
from agentic_cli.memory._core.secrets import looks_like_secret
from agentic_cli.memory.store import _files
from agentic_cli.memory.store._scope import INBOX, PROJECT_NOTE, Entry, ScopeFolder, resolve_duplicates
from agentic_cli.memory.store.models import (
    MEMORY_TYPES,
    SCOPES,
    SOURCES,
    DuplicateMemory,
    InvalidMemory,
    Memory,
    MemoryBusy,
    MemoryStoreFull,
    NoProjectScope,
    Problem,
    Version,
    newest_first,
)

logger = get_logger("agentic_cli.memory.store")

SCOPE_MAX = 1_000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStore:
    """Memories kept across conversations, as markdown files (provisional API).

    Args:
        user_dir: The user scope's folder (memories that hold across projects).
        project_dir: The project scope's folder, or None for no project scope.
        project_name: The project's name in the index and ``/memory``; by
            default the name of ``project_dir``'s parent folder.
        project_path: The project's absolute path, written to
            ``project-path.txt`` beside the project scope when it is created.
        screen: Returns why a text must not be kept (a credential), or None;
            None for no screen.
        now: The clock (UTC); for tests.
    """

    def __init__(
        self,
        user_dir: Path | str,
        project_dir: Path | str | None = None,
        *,
        project_name: str | None = None,
        project_path: Path | str | None = None,
        screen: Callable[[str], str | None] | None = looks_like_secret,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._folders: dict[str, ScopeFolder] = {"user": ScopeFolder(Path(user_dir), "user")}
        self.project_name: str | None = None
        if project_dir is not None:
            project_dir = Path(project_dir)
            note = (project_dir.parent / PROJECT_NOTE, str(project_path)) if project_path is not None else None
            self._folders["project"] = ScopeFolder(project_dir, "project", note=note)
            self.project_name = project_name or project_dir.parent.name
        self._screen = screen
        self._now = now or _utc_now
        self._lock = threading.RLock()
        self._import_problems: list[Problem] = []
        for folder in self._folders.values():
            folder.check()
        self._tidy()

    @property
    def has_project_scope(self) -> bool:
        """Whether the store has a project scope."""
        return "project" in self._folders

    # -- helpers ------------------------------------------------------------

    def _now_s(self) -> datetime:
        """Now, in UTC, to the second: what a file records."""
        return self._now().astimezone(timezone.utc).replace(microsecond=0)

    def _folder(self, scope: str) -> ScopeFolder:
        if scope not in SCOPES:
            raise InvalidMemory(f"scope: {scope!r} is not one of {', '.join(SCOPES)}")
        folder = self._folders.get(scope)
        if folder is None:
            raise NoProjectScope(
                "there is no project scope here (the app started in the home folder, / or above it)"
            )
        return folder

    def _entries(self) -> tuple[dict[str, Entry], list[Problem]]:
        """Each id's memory across both scopes, and every problem."""
        entries: list[Entry] = []
        problems: list[Problem] = []
        for folder in self._folders.values():
            found, wrong = folder.scan()
            entries += found
            problems += wrong
        winners, duplicates = resolve_duplicates(entries)
        return winners, problems + duplicates

    def _checked(
        self, description: str, content: str, memory_type: str, scope: str | None = None
    ) -> tuple[str, str]:
        """``description`` and ``content`` stripped, after save rules 1-3."""
        reason = _files.description_problem(description)
        if reason:
            raise InvalidMemory(f"description: {reason}")
        reason = _files.content_problem(content)
        if reason:
            raise InvalidMemory(f"content: {reason}")
        if memory_type not in MEMORY_TYPES:
            raise InvalidMemory(f"type: {memory_type!r} is not one of {', '.join(MEMORY_TYPES)}")
        if scope is not None and scope not in SCOPES:
            raise InvalidMemory(f"scope: {scope!r} is not one of {', '.join(SCOPES)}")
        return description.strip(), content.strip()

    def _screened(self, description: str, content: str) -> None:
        """Save rule 5: refuse what looks like a credential."""
        if self._screen is None:
            return
        reason = self._screen(description + "\n" + content)
        if reason:
            raise InvalidMemory(f"looks like a secret ({reason}); memories must not hold credentials")

    @staticmethod
    def _check_unique(description: str, winners: dict[str, Entry], exclude: str | None = None) -> None:
        """Save rule 6: no other memory has this description."""
        key = _files.normalized(description)
        for entry in winners.values():
            if entry.memory.id != exclude and _files.normalized(entry.memory.description) == key:
                raise DuplicateMemory(entry.memory.id)

    def _taken(self, winners: dict[str, Entry]) -> Callable[[str], bool]:
        """Whether a new id is taken by a memory, a proposal or a history entry."""
        def taken(candidate: str) -> bool:
            if candidate in winners:
                return True
            return any(
                candidate in folder.history_ids() or (folder.path / INBOX / f"{candidate}.md").exists()
                for folder in self._folders.values()
            )
        return taken

    def _count(self, winners: dict[str, Entry], scope: str) -> int:
        return sum(1 for entry in winners.values() if entry.memory.scope == scope)

    def _tidy(self) -> None:
        """Prune each existing scope's history; a busy scope is left for the next open."""
        now = self._now()
        for folder in self._folders.values():
            if not folder.path.is_dir():
                continue
            try:
                with self._lock, folder.lock():
                    folder.prune_history(now)
            except MemoryBusy:
                continue

    # -- memories ------------------------------------------------------------

    def save(
        self,
        description: str,
        content: str,
        type: str,
        *,
        scope: str = "project",
        source: str = "agent",
        conversation: str | None = None,
    ) -> Memory:
        """Save a new memory (spec §6.3's rules, in order).

        Raises:
            InvalidMemory: a field fails its rule, or the text looks like a secret.
            NoProjectScope: ``scope="project"`` and there is none.
            DuplicateMemory: another memory has the same description.
            MemoryStoreFull: the scope holds ``SCOPE_MAX`` memories.
            MemoryBusy: another process held the scope's lock too long.
        """
        description, content = self._checked(description, content, type, scope)
        if source not in SOURCES:
            raise InvalidMemory(f"source: {source!r} is not one of {', '.join(SOURCES)}")
        folder = self._folder(scope)
        self._screened(description, content)
        with self._lock, folder.lock():
            return self._save_locked(folder, description, content, type, source, conversation)

    def _save_locked(
        self,
        folder: ScopeFolder,
        description: str,
        content: str,
        memory_type: str,
        source: str,
        conversation: str | None,
        *,
        memory_id: str | None = None,
        created: datetime | None = None,
        updated: datetime | None = None,
    ) -> Memory:
        """Rules 6 and 7, then write. The caller checked the fields and holds the lock."""
        winners, _ = self._entries()
        self._check_unique(description, winners)
        if self._count(winners, folder.scope) >= SCOPE_MAX:
            raise MemoryStoreFull(
                f"the {folder.scope} scope holds {SCOPE_MAX} memories: delete or merge some first"
            )
        now = self._now_s()
        memory_id = memory_id or _files.new_id(self._taken(winners))
        memory = Memory(
            id=memory_id,
            description=description,
            content=content,
            type=memory_type,  # type: ignore[arg-type]
            scope=folder.scope,
            source=source,  # type: ignore[arg-type]
            conversation=conversation,
            created=created or now,
            updated=updated or created or now,
            path=folder.target(description, memory_id),
        )
        folder.write(memory, {})
        return memory

    def read(self, memory_id: str) -> Memory | None:
        """The memory with this id, or None."""
        if not _files.valid_id(memory_id):
            return None
        with self._lock:
            entry = self._entries()[0].get(memory_id)
        return entry.memory if entry is not None else None

    def update(
        self,
        memory_id: str,
        *,
        description: str | None = None,
        content: str | None = None,
        type: str | None = None,
    ) -> Memory | None:
        """Change a memory; its previous version goes to history. None: no such memory.

        ``id``, ``created``, ``source`` and ``conversation`` stay; ``updated``
        is now; the file is renamed when the description changes.

        Raises:
            InvalidMemory: no field given, a field fails its rule, or the text
                looks like a secret.
            DuplicateMemory: another memory has the new description.
            MemoryBusy: another process held the scope's lock too long.
        """
        if description is None and content is None and type is None:
            raise InvalidMemory("give a new description, content or type")
        if description is not None:
            reason = _files.description_problem(description)
            if reason:
                raise InvalidMemory(f"description: {reason}")
            description = description.strip()
        if content is not None:
            reason = _files.content_problem(content)
            if reason:
                raise InvalidMemory(f"content: {reason}")
            content = content.strip()
        if type is not None and type not in MEMORY_TYPES:
            raise InvalidMemory(f"type: {type!r} is not one of {', '.join(MEMORY_TYPES)}")
        if not _files.valid_id(memory_id):
            return None
        with self._lock:
            entry = self._entries()[0].get(memory_id)
            if entry is None:
                return None
            folder = self._folders[entry.memory.scope]
            with folder.lock():
                return self._update_locked(folder, memory_id, description, content, type)

    def _update_locked(
        self,
        folder: ScopeFolder,
        memory_id: str,
        description: str | None,
        content: str | None,
        memory_type: str | None,
    ) -> Memory | None:
        """Rules 5 and 6, then the change. The caller checked the fields and holds the lock."""
        winners, _ = self._entries()
        entry = winners.get(memory_id)
        if entry is None:
            return None
        new_description = description if description is not None else entry.memory.description
        new_content = content if content is not None else entry.memory.content
        self._screened(new_description, new_content)
        self._check_unique(new_description, winners, exclude=memory_id)
        changed = replace(
            entry.memory,
            description=new_description,
            content=new_content,
            type=memory_type if memory_type is not None else entry.memory.type,
            updated=self._now_s(),
            path=folder.target(new_description, memory_id),
        )
        folder.keep_version(entry, self._now())
        folder.write(changed, entry.extra)
        if entry.memory.path != changed.path:
            entry.memory.path.unlink(missing_ok=True)
        folder.prune_history(self._now(), memory_id)
        return changed

    def delete(self, memory_id: str) -> bool:
        """Delete a memory; it stays in history (restorable for 30 days). False: no such memory."""
        if not _files.valid_id(memory_id):
            return False
        with self._lock:
            entry = self._entries()[0].get(memory_id)
            if entry is None:
                return False
            folder = self._folders[entry.memory.scope]
            with folder.lock():
                entry = self._entries()[0].get(memory_id)
                if entry is None:
                    return False
                folder.keep_version(entry, self._now())
                entry.memory.path.unlink(missing_ok=True)
                folder.prune_history(self._now(), memory_id)
                return True

    def memories(self, scope: str | None = None) -> list[Memory]:
        """Every memory (of ``scope`` only, if given), newest ``updated`` first, then by id."""
        with self._lock:
            winners = self._entries()[0]
        found = [e.memory for e in winners.values() if scope is None or e.memory.scope == scope]
        return sorted(found, key=newest_first)

    def problems(self) -> list[Problem]:
        """The files the store cannot use (it never changes them)."""
        with self._lock:
            return self._entries()[1] + list(self._import_problems)

    # -- history -------------------------------------------------------------

    def history(self, memory_id: str | None = None, *, days: int = 30) -> list[Version]:
        """Versions saved in the last ``days`` days, newest first (of one memory, if given)."""
        with self._lock:
            winners = self._entries()[0]
            cutoff = self._now() - timedelta(days=days)
            found: list[tuple[str, datetime, Memory]] = []
            for folder in self._folders.values():
                for one, saved, path in folder.versions(memory_id):
                    if saved < cutoff:
                        continue
                    entry = folder.read_version(path)
                    if entry is not None and entry.memory.id == one:
                        found.append((one, saved, entry.memory))
        newest: dict[str, datetime] = {}
        for one, saved, _ in found:
            newest[one] = max(saved, newest.get(one, saved))
        versions = [
            Version(memory_id=one, saved=saved, memory=memory,
                    deleted=one not in winners and saved == newest[one])
            for one, saved, memory in found
        ]
        return sorted(versions, key=lambda v: (-v.saved.timestamp(), v.memory_id))

    def _find_version(self, memory_id: str, saved: datetime) -> tuple[ScopeFolder, Entry]:
        for folder in self._folders.values():
            for one, when, path in folder.versions(memory_id):
                if when == saved:
                    entry = folder.read_version(path)
                    if entry is not None:
                        return folder, entry
        raise InvalidMemory(f"no version of {memory_id} saved at {saved.isoformat()}")

    def restore(self, memory_id: str, saved: datetime) -> Memory:
        """Bring back the version of ``memory_id`` saved at ``saved``.

        The current version, if any, goes to history first, so a restore can
        itself be undone.

        Raises:
            InvalidMemory: no such version.
            DuplicateMemory: another memory now has its description.
            MemoryStoreFull: it was deleted and its scope is full.
            MemoryBusy: another process held the scope's lock too long.
        """
        with self._lock:
            folder, old = self._find_version(memory_id, saved)
            with folder.lock():
                winners, _ = self._entries()
                current = winners.get(memory_id)
                self._check_unique(old.memory.description, winners, exclude=memory_id)
                if current is None and self._count(winners, folder.scope) >= SCOPE_MAX:
                    raise MemoryStoreFull(
                        f"the {folder.scope} scope holds {SCOPE_MAX} memories: delete or merge some first"
                    )
                restored = replace(
                    old.memory,
                    scope=folder.scope,
                    updated=self._now_s(),
                    path=folder.target(old.memory.description, memory_id),
                )
                if current is not None:
                    folder.keep_version(current, self._now())
                folder.write(restored, old.extra)
                if current is not None and current.memory.path != restored.path:
                    current.memory.path.unlink(missing_ok=True)
                folder.prune_history(self._now(), memory_id)
                return restored
