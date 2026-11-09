"""/memory: review what memory holds (spec §12).

The list of both scopes is a picker like /resume's; one memory is a dialog
with Delete and, when it has earlier versions, History. ``/memory history``
restores a version, ``/memory inbox`` steps through the extractor's
proposals, ``/memory stale`` lists memories not updated for 90 days. The
command acts as the user, through the store: no permission prompt, and every
change keeps the previous version. Store calls read files, so they run on a
worker thread.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from thinking_prompt import ButtonConfig, ChoiceField, Dialog

from agentic_cli.cli.commands import Command, CommandCategory

if TYPE_CHECKING:
    from agentic_cli.memory import Memory, MemoryStore, Problem

STALE_DAYS = 90
NOT_READY = (
    "Memory is not available yet — the workflow is still initializing. Try /memory again in a moment."
)
NO_MEMORY = "This app has no memory."
UNAVAILABLE = (
    "Memory is unavailable: its folder could not be opened. "
    "The log says why (memory_store_unavailable)."
)
_PROBLEMS = "\0problems"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _when(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def proposals_line(count: int) -> str | None:
    """``N memory proposals: /memory inbox``, or None when none wait."""
    if not count:
        return None
    return f"{count} memory proposal{'s' if count != 1 else ''}: /memory inbox"


def memory_store(app: Any) -> "MemoryStore | None":
    """The app's memory store; None (and the user told why) when there is none.

    A workflow whose tools need memory but has no store could not open its
    folder (the log says why); one whose tools do not need it has no memory.
    """
    try:
        workflow = app.workflow
    except (RuntimeError, AttributeError):
        app.session.add_warning(NOT_READY)
        return None
    store = getattr(workflow, "memory_store", None)
    if store is None:
        needed = "memory_store" in getattr(workflow, "required_managers", ())
        app.session.add_message("system", UNAVAILABLE if needed else NO_MEMORY)
    return store


def _row(memory: "Memory") -> str:
    return (
        f"{memory.updated:%Y-%m-%d} · {memory.scope} · {memory.type} · {memory.source} · "
        f"{memory.description} · {memory.id}"
    )


def memory_picker(
    memories: list["Memory"], problems: list["Problem"], waiting: int, title: str
) -> tuple[Dialog, dict[str, str]]:
    """The list: one option a memory (the id last, so rows stay unique on a
    narrow terminal), then ``Problems (N)``; and what each option stands for."""
    rows = {_row(memory): memory.id for memory in memories}
    if problems:
        rows[f"Problems ({len(problems)})"] = _PROBLEMS
    project = sum(1 for m in memories if m.scope == "project")
    user = sum(1 for m in memories if m.scope == "user")
    heading = f"{title}: {project} project, {user} user"
    line = proposals_line(waiting)
    if line:
        heading += f" · {line}"
    return Dialog(heading, fields=[ChoiceField("memory", "", options=list(rows))]), rows


def memory_view(memory: "Memory", has_history: bool) -> Dialog:
    """One memory: its fields, its file, its text; Delete, History, Close."""
    body = "\n".join([
        memory.description,
        "",
        f"Scope: {memory.scope}   Type: {memory.type}   Source: {memory.source}",
        f"Conversation: {memory.conversation or '-'}",
        f"Created: {_when(memory.created)}   Updated: {_when(memory.updated)}",
        f"File: {memory.path} (edit it in your editor)",
        "",
        memory.content,
    ])
    buttons = [ButtonConfig("Delete", result="delete")]
    if has_history:
        buttons.append(ButtonConfig("History", result="history"))
    buttons.append(ButtonConfig("Close", result=None))
    return Dialog(f"Memory {memory.id}", body=body, buttons=buttons)


class MemoryCommand(Command):
    """Review saved memories: the list, one memory, history, the inbox, stale ones.

    Args:
        now: The clock ``stale`` measures against (UTC); for tests.
    """

    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        super().__init__(
            name="memory",
            description="Review saved memories: the list, history, the inbox, stale ones",
            usage="/memory [inbox | history | stale | <id>]",
            examples=["/memory", "/memory inbox", "/memory history", "/memory stale", "/memory 4f2a9c1e"],
            category=CommandCategory.GENERAL,
        )
        self._now = now or _utc_now

    async def execute(self, args: str, app: Any) -> None:
        store = memory_store(app)
        if store is None:
            return
        word = self.parse_args(args).positional.strip()
        if word == "inbox":
            await self._inbox(app, store)
        elif word == "history":
            await self._history(app, store, None)
        elif word == "stale":
            await self._list(app, store, stale=True)
        elif word:
            await self._show(app, store, word)
        else:
            await self._list(app, store)

    async def _list(self, app: Any, store: "MemoryStore", *, stale: bool = False) -> None:
        memories = await asyncio.to_thread(store.memories)
        problems = await asyncio.to_thread(store.problems)
        waiting = len(await asyncio.to_thread(store.proposals))
        if stale:
            cutoff = self._now() - timedelta(days=STALE_DAYS)
            memories, problems = [m for m in memories if m.updated <= cutoff], []
            if not memories:
                app.session.add_message("system", f"No memory has gone {STALE_DAYS} days without an update.")
                return
        elif not memories and not problems:
            line = proposals_line(waiting)
            app.session.add_message("system", "No memories saved yet." + (f" {line}" if line else ""))
            return
        dialog, rows = memory_picker(memories, problems, waiting, "Stale memories" if stale else "Memory")
        picked = await app.session.show_dialog(dialog)
        key = rows.get(picked) if picked is not None else None
        if key == _PROBLEMS:
            await self._problems(app, problems)
        elif key is not None:
            await self._show(app, store, key)

    async def _problems(self, app: Any, problems: list["Problem"]) -> None:
        body = "\n".join(f"{p.path}\n  {p.kind}: {p.detail}" for p in problems)
        await app.session.show_dialog(Dialog(
            "Memory files that cannot be used", body=body, buttons=[ButtonConfig("Close", result=None)],
        ))

    async def _show(self, app: Any, store: "MemoryStore", memory_id: str) -> None:
        from agentic_cli.memory import MemoryStoreError

        memory = await asyncio.to_thread(store.read, memory_id)
        if memory is None:
            app.session.add_message("system", f"No memory with id {memory_id}.")
            return
        versions = await asyncio.to_thread(store.history, memory.id)
        choice = await app.session.show_dialog(memory_view(memory, bool(versions)))
        if choice == "delete":
            try:
                deleted = await asyncio.to_thread(store.delete, memory.id)
            except (MemoryStoreError, OSError) as e:
                app.session.add_error(f"Could not delete memory {memory.id}: {e}")
                return
            if deleted:
                app.session.add_success(
                    f"Deleted memory {memory.id} (restorable for 30 days: /memory history)."
                )
        elif choice == "history":
            await self._history(app, store, memory.id)

    async def _history(self, app: Any, store: "MemoryStore", memory_id: str | None) -> None:
        raise NotImplementedError  # Task 20

    async def _inbox(self, app: Any, store: "MemoryStore") -> None:
        raise NotImplementedError  # Task 20
