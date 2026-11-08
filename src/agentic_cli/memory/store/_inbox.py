"""A scope's inbox: proposals waiting for a person, and what they discarded (spec §8.3).

A proposal is a memory file plus ``action``, ``target`` and
``outside_content``, in ``<scope>/.inbox/<id>.md``. The model never sees one.
``discarded.jsonl`` keeps the normalized descriptions a person discarded, so
the extractor is not shown the same line after every similar conversation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from agentic_cli.memory._core.io import append_text, atomic_write_text, read_if_regular, read_regular_file
from agentic_cli.memory.store._files import (
    FILE_MAX_BYTES,
    MEMORY_KEYS,
    PROPOSAL_KEYS,
    FieldError,
    FormatError,
    format_time,
    parse,
    proposal_fields,
    proposal_from,
    render,
)
from agentic_cli.memory.store._scope import INBOX, ScopeFolder
from agentic_cli.memory.store.models import Problem, Proposal

INBOX_MAX = 100
PROPOSAL_DAYS = 30
DISCARDED_KEEP = 1_000
DISCARDED = "discarded.jsonl"


class Inbox:
    """The inbox of one scope's folder. Changes hold the folder's lock."""

    def __init__(self, folder: ScopeFolder) -> None:
        self._folder = folder

    @property
    def path(self) -> Path:
        return self._folder.path / INBOX

    def proposals(self) -> tuple[list[Proposal], list[Problem]]:
        """Every proposal in the inbox, and every file that is not one."""
        try:
            listing = list(os.scandir(self.path))
        except FileNotFoundError:
            return [], []
        found: list[Proposal] = []
        problems: list[Problem] = []
        for item in listing:
            if item.name.startswith(".") or not item.name.endswith(".md"):
                continue
            path = Path(item.path)
            try:
                text = read_regular_file(path, limit=FILE_MAX_BYTES).decode("utf-8")
                fields, content = parse(text)
                found.append(proposal_from(fields, content, scope=self._folder.scope, path=path))
            except (OSError, UnicodeDecodeError, FormatError, FieldError) as e:
                problems.append(Problem(path, "bad_proposal", str(e)))
        return found, problems

    def write(self, proposal: Proposal) -> None:
        self._folder.ensure(INBOX)
        atomic_write_text(
            proposal.path,
            render(proposal_fields(proposal), proposal.content, order=MEMORY_KEYS + PROPOSAL_KEYS),
        )

    def remove(self, proposal: Proposal) -> None:
        proposal.path.unlink(missing_ok=True)

    def discarded(self) -> list[str]:
        """The normalized descriptions a person discarded, oldest first."""
        data = read_if_regular(self.path / DISCARDED)
        if data is None:
            return []
        found: list[str] = []
        for line in data.decode("utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("description"), str):
                found.append(entry["description"])
        return found

    def add_discarded(self, description: str, now: datetime) -> None:
        """Remember ``description`` (normalized by the caller); keep the newest ``DISCARDED_KEEP``."""
        folder = self._folder.ensure(INBOX)
        line = json.dumps({"description": description, "date": format_time(now)}, ensure_ascii=False)
        append_text(folder / DISCARDED, line + "\n")
        entries = self.discarded()
        if len(entries) > DISCARDED_KEEP:
            kept = (read_if_regular(folder / DISCARDED) or b"").decode("utf-8").splitlines()[-DISCARDED_KEEP:]
            atomic_write_text(folder / DISCARDED, "\n".join(kept) + "\n")

    def prune(self, now: datetime) -> None:
        """Remove proposals older than ``PROPOSAL_DAYS``; they are not counted as discarded."""
        cutoff = now - timedelta(days=PROPOSAL_DAYS)
        found, _ = self.proposals()
        for proposal in found:
            if proposal.created < cutoff:
                self.remove(proposal)
