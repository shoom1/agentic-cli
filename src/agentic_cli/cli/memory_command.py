"""/memory: review what memory holds (spec §12).

The list of both scopes is a picker like /resume's; one memory is a dialog
with Delete and, when it has earlier versions, History. ``/memory history``
restores a version, ``/memory inbox`` steps through the extractor's
proposals, ``/memory stale`` lists memories not updated for 90 days. The
command acts as the user, through the store: no permission prompt, and every
change keeps the previous version. Store calls read files, so they run on a
worker thread.

Enter must never be a destructive press: the button a dialog opens on is the
harmless one (Close, Skip), and Delete acts only on the memory as it was shown.
Each dialog is drawn where the ``dialog_placement`` setting says now: the
session keeps the placement it started with.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from thinking_prompt import ButtonConfig, ChoiceField, Dialog

from agentic_cli.cli.commands import Command, CommandCategory

if TYPE_CHECKING:
    from thinking_prompt import Placement

    from agentic_cli.memory import Memory, MemoryStore, Problem, Proposal, Version

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


def _placement(app: Any) -> "Placement":
    """Where ``/memory``'s dialogs are drawn: the ``dialog_placement`` setting
    now, which ``/settings`` may have changed since the session started."""
    return app.settings.dialog_placement


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
    memories: list["Memory"], problems: list["Problem"], waiting: int, title: str,
    *, placement: "Placement | None" = None,
) -> tuple[Dialog, dict[str, str]]:
    """The list: one option a memory (the id last, so rows stay unique on a
    narrow terminal), then ``Problems (N)``; and what each option stands for.
    ``placement`` is where it is drawn (None: the session's)."""
    rows = {_row(memory): memory.id for memory in memories}
    if problems:
        rows[f"Problems ({len(problems)})"] = _PROBLEMS
    project = sum(1 for m in memories if m.scope == "project")
    user = sum(1 for m in memories if m.scope == "user")
    heading = f"{title}: {project} project, {user} user"
    line = proposals_line(waiting)
    if line:
        heading += f" · {line}"
    fields = [ChoiceField("memory", "", options=list(rows))]
    return Dialog(heading, fields=fields, placement=placement), rows


def memory_view(
    memory: "Memory", has_history: bool, *, placement: "Placement | None" = None
) -> Dialog:
    """One memory: its fields, its file, its text; Delete, History, Close.
    ``placement`` is where it is drawn (None: the session's)."""
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
    buttons.append(ButtonConfig("Close", result=None, focused=True))
    return Dialog(f"Memory {memory.id}", body=body, buttons=buttons, placement=placement)


def _find(store: "MemoryStore", memory_id: str) -> "Memory | None":
    """The memory with this id, whatever the case it is typed in."""
    memory = store.read(memory_id)
    if memory is None:
        wanted = memory_id.casefold()
        memory = next((m for m in store.memories() if m.id.casefold() == wanted), None)
    return memory


def _claiming(store: "MemoryStore", memory_id: str) -> "Problem | None":
    """The file the store cannot use that still holds ``memory_id``, if any."""
    wanted = memory_id.casefold()
    return next(
        (p for p in store.problems() if p.memory_id and p.memory_id.casefold() == wanted), None
    )


def version_body(version: "Version", current: "Memory | None", unusable: "Problem | None" = None) -> str:
    """A saved version beside the memory as it is now: ``deleted``, or the
    file a person broke (``unusable``) when that holds its id."""
    lines = [f"Saved {_when(version.saved)}:", version.memory.description, "", version.memory.content, "", "Now:"]
    if current is not None:
        lines += [current.description, "", current.content]
    elif unusable is not None:
        lines += [_unusable(unusable)]
    else:
        lines += ["deleted"]
    return "\n".join(lines)


def _unusable(problem: "Problem") -> str:
    """Where a memory is, when a person's broken file holds its id."""
    return f"its file cannot be used (see Problems in /memory): {problem.path}"


def proposal_body(
    proposal: "Proposal", current: "Memory | None", unusable: "Problem | None" = None
) -> str:
    """What a proposal would do: a new memory, or a change beside the memory now.

    When that memory is gone, the body says so: deleted (the proposal can be
    discarded), or its file is one a person broke (``unusable``: fix it, and
    nothing is to be discarded).
    """
    lines: list[str] = []
    if proposal.outside_content:
        lines += [f"From a conversation that read outside content ({proposal.outside_content}).", ""]
    if proposal.action == "update":
        lines += ["A change to a memory.", "", "Now:"]
        if current is not None:
            lines += [current.description, f"Type: {current.type}", "", current.content]
        elif unusable is not None:
            lines += [_unusable(unusable)]
        else:
            lines += ["(deleted: discard this proposal)"]
        lines += ["", "Proposed:", proposal.description, f"Type: {proposal.type}", "", proposal.content]
    else:
        lines += [f"A new {proposal.scope} memory ({proposal.type}).", "", proposal.description, "", proposal.content]
    lines += ["", f"Conversation: {proposal.conversation or '-'}"]
    return "\n".join(lines)


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
        dialog, rows = memory_picker(
            memories, problems, waiting, "Stale memories" if stale else "Memory",
            placement=_placement(app),
        )
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
            placement=_placement(app),
        ))

    async def _show(self, app: Any, store: "MemoryStore", memory_id: str) -> None:
        memory = await asyncio.to_thread(_find, store, memory_id)
        if memory is None:
            await self._missing(app, store, memory_id, shown=False)
            return
        versions = await asyncio.to_thread(store.history, memory.id)
        choice = await app.session.show_dialog(
            memory_view(memory, bool(versions), placement=_placement(app))
        )
        if choice == "delete":
            await self._delete(app, store, memory)
        elif choice == "history":
            await self._history(app, store, memory.id)

    async def _missing(self, app: Any, store: "MemoryStore", memory_id: str, *, shown: bool) -> None:
        """Say why there is no memory ``memory_id``: its file cannot be used (and
        where it is listed), it was deleted since it was shown, or it never was."""
        problem = await asyncio.to_thread(_claiming, store, memory_id)
        if problem is not None:
            line = (
                f"Memory {memory_id} is in a file that cannot be used: {problem.path} "
                f"({problem.kind}: {problem.detail}). It is listed under Problems in /memory."
            )
        elif shown:
            line = f"Memory {memory_id} was already deleted."
        else:
            line = f"No memory with id {memory_id}."
        app.session.add_message("system", line)

    async def _delete(self, app: Any, store: "MemoryStore", shown: "Memory") -> None:
        """Delete ``shown`` only if it is still what the person saw: another
        process may have changed or removed it since the view opened."""
        from agentic_cli.memory import MemoryStoreError

        current = await asyncio.to_thread(store.read, shown.id)
        if current is None:
            await self._missing(app, store, shown.id, shown=True)
            return
        if current != shown:
            app.session.add_message(
                "system", f"Memory {shown.id} changed since it was shown, so it was not deleted. It is shown again."
            )
            await self._show(app, store, shown.id)
            return
        try:
            deleted = await asyncio.to_thread(store.delete, shown.id)
        except (MemoryStoreError, OSError) as e:
            app.session.add_error(f"Could not delete memory {shown.id}: {e}")
            return
        if deleted:
            app.session.add_success(f"Deleted memory {shown.id} (restorable for 30 days: /memory history).")
        else:
            app.session.add_message("system", f"Memory {shown.id} was already deleted.")

    async def _history(self, app: Any, store: "MemoryStore", memory_id: str | None) -> None:
        from agentic_cli.memory import MemoryStoreError

        versions = await asyncio.to_thread(store.history, memory_id)
        if not versions:
            app.session.add_message("system", "No changes in the last 30 days.")
            return
        rows: dict[str, Any] = {}
        for version in versions:
            label = (
                f"{version.saved:%Y-%m-%d %H:%M} · {'deleted' if version.deleted else 'updated'} · "
                f"{version.memory.description} · {version.memory_id}"
            )
            while label in rows:
                label += " ·"
            rows[label] = version
        title = "Memory history" + (f" of {memory_id}" if memory_id else "")
        picked = await app.session.show_dialog(Dialog(
            title, fields=[ChoiceField("version", "", options=list(rows))],
            placement=_placement(app),
        ))
        version = rows.get(picked) if picked is not None else None
        if version is None:
            return
        current = await asyncio.to_thread(store.read, version.memory_id)
        unusable = None if current is not None else await asyncio.to_thread(_claiming, store, version.memory_id)
        choice = await app.session.show_dialog(Dialog(
            f"Version of {version.memory_id}",
            body=version_body(version, current, unusable),
            buttons=[ButtonConfig("Restore", result="restore"), ButtonConfig("Close", result=None, focused=True)],
            placement=_placement(app),
        ))
        if choice != "restore":
            return
        try:
            restored = await asyncio.to_thread(store.restore, version.memory_id, version.saved)
        except (MemoryStoreError, OSError) as e:
            app.session.add_error(f"Could not restore memory {version.memory_id}: {e}")
            return
        app.session.add_success(f"Restored memory {restored.id}: {restored.description}.")

    async def _inbox(self, app: Any, store: "MemoryStore") -> None:
        from agentic_cli.memory import MemoryStoreError

        proposals = await asyncio.to_thread(store.proposals)
        if not proposals:
            app.session.add_message("system", "No memory proposals waiting.")
            return
        accepted = discarded = 0
        for number, proposal in enumerate(proposals, start=1):
            current = unusable = None
            if proposal.action == "update":
                current = await asyncio.to_thread(store.read, proposal.target or "")
                if current is None:
                    unusable = await asyncio.to_thread(_claiming, store, proposal.target or "")
            choice = await app.session.show_dialog(Dialog(
                f"Memory proposal {number} of {len(proposals)}",
                body=proposal_body(proposal, current, unusable),
                buttons=[
                    ButtonConfig("Accept", result="accept"),
                    ButtonConfig("Discard", result="discard"),
                    ButtonConfig("Skip", result="skip", focused=True),
                ],
                placement=_placement(app),
            ))
            if choice is None:  # Esc stops; what is left waits
                break
            try:
                if choice == "accept":
                    await asyncio.to_thread(store.accept, proposal.id)
                    accepted += 1
                elif choice == "discard":
                    if await asyncio.to_thread(store.discard, proposal.id):
                        discarded += 1
                    else:
                        app.session.add_message("system", "That proposal was already handled.")
            except (MemoryStoreError, OSError) as e:
                app.session.add_error(f"Could not {choice} {proposal.description!r}: {e}")
        waiting = len(await asyncio.to_thread(store.proposals))
        app.session.add_message("system", f"Accepted {accepted}, discarded {discarded}, {waiting} waiting.")
