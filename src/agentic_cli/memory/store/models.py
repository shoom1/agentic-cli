"""The memory store's types: a memory, a proposal, a version, a problem, the errors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

MemoryType = Literal["user", "feedback", "project", "reference"]
Scope = Literal["user", "project"]
Source = Literal["user", "agent", "extractor", "imported"]
Action = Literal["new", "update"]

MEMORY_TYPES: tuple[str, ...] = ("user", "feedback", "project", "reference")
SCOPES: tuple[str, ...] = ("user", "project")
SOURCES: tuple[str, ...] = ("user", "agent", "extractor", "imported")
ACTIONS: tuple[str, ...] = ("new", "update")
#: The memory tools an agent can have; the index names only those it has.
ALL_MEMORY_TOOLS: tuple[str, ...] = (
    "save_memory", "read_memory", "search_memory", "update_memory", "delete_memory",
)


@dataclass(frozen=True)
class Memory:
    """One memory: a markdown file in a scope's folder."""

    id: str
    description: str
    content: str
    type: MemoryType
    scope: Scope
    source: Source
    conversation: str | None
    created: datetime
    updated: datetime
    path: Path


def newest_first(memory: Memory) -> tuple[float, str]:
    """The order of every list of memories: newest ``updated`` first, then by id."""
    return (-memory.updated.timestamp(), memory.id)


@dataclass(frozen=True)
class Candidate:
    """A memory the extractor proposes: a new one, or a change to one by id."""

    action: Action
    target: str | None
    description: str
    content: str
    type: MemoryType
    scope: Scope


@dataclass(frozen=True)
class Proposal:
    """A candidate filed in a scope's inbox, waiting for a person."""

    id: str
    action: Action
    target: str | None
    description: str
    content: str
    type: MemoryType
    scope: Scope
    source: Source
    conversation: str | None
    outside_content: str | None
    created: datetime
    path: Path


@dataclass(frozen=True)
class Version:
    """A memory as it was before a change or a delete."""

    memory_id: str
    saved: datetime
    memory: Memory
    deleted: bool


@dataclass(frozen=True)
class Problem:
    """A file the store cannot use, and why. The store never changes it.

    ``memory_id`` is the id its frontmatter holds when that is a valid id (a
    ``bad_field`` file), so the memory it was is not taken for a deleted one.
    """

    path: Path
    kind: str
    detail: str
    memory_id: str | None = None


class MemoryStoreError(Exception):
    """A memory store call that cannot be done; the message says why."""


class InvalidMemory(MemoryStoreError, ValueError):
    """A field fails its rule; the message names the field and the rule."""


class DuplicateMemory(MemoryStoreError):
    """Another memory has the same description."""

    def __init__(self, existing_id: str) -> None:
        super().__init__(f"memory {existing_id} has the same description")
        self.existing_id = existing_id


class MemoryStoreFull(MemoryStoreError):
    """The scope holds as many memories as it may."""


class NoProjectScope(MemoryStoreError):
    """The store has no project scope (the app started at home, ``/`` or above home)."""


class MemoryBusy(MemoryStoreError, TimeoutError):
    """Another process held the scope's lock for longer than the store waits."""


class FileInTheWay(MemoryStoreError):
    """A change would write over a file that is not the memory's own.

    The file is one the store cannot use (a person's edit that broke it) or
    one it does not own; it is never rewritten, moved or deleted.
    """


class UnsafeMemoryStore(MemoryStoreError):
    """A scope folder, or one inside it, is a symbolic link or not a folder."""
