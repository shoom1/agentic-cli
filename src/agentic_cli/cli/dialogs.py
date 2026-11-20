"""One dialog at a time on a session.

thinking-prompt shows one dialog and raises on a second. A turn and a command
used to run one at a time, in the input handler, so they never met. A
sub-agent's view runs its run outside the handler (the prompt stays open to
steer it), so a command may open a dialog while the run asks its question, or
the run may ask while a command's dialog is open: a second dialog that raised
would be a permission prompt answered by a denial the user never saw, or a
command that fails.

Every dialog of the app's session takes the session's one lock
(``dialog_lock``) for as long as it is open: whichever comes second waits for
the user to answer the first. ``serialize_dialogs`` makes that true of the
session's own dialog methods, whoever calls them (the framework's commands, an
app's own, a question's dialog in the message processor), so no caller has a
path of its own to remember.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import weakref
from typing import Any

#: The session methods that open a dialog (thinking-prompt's public ones).
DIALOG_METHODS = (
    "show_dialog",
    "yes_no_dialog",
    "message_dialog",
    "choice_dialog",
    "dropdown_dialog",
    "checklist_dialog",
)


class DialogLock:
    """A lock a task may take again while it holds it.

    The message processor holds it for a question's whole dialog, and the
    session's wrapped dialog method takes it again inside: the same task is
    let through, so the two never wait for each other.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._depth = 0

    async def __aenter__(self) -> None:
        task = asyncio.current_task()
        if self._owner is not None and self._owner is task:
            self._depth += 1
            return
        await self._lock.acquire()
        self._owner = task
        self._depth = 1

    async def __aexit__(self, *exc: Any) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()


_LOCKS: "weakref.WeakKeyDictionary[Any, DialogLock]" = weakref.WeakKeyDictionary()


def dialog_lock(ui: Any) -> DialogLock:
    """The lock every dialog of the session ``ui`` takes while it is open."""
    lock = _LOCKS.get(ui)
    if lock is None:
        lock = _LOCKS[ui] = DialogLock()
    return lock


def serialize_dialogs(session: Any) -> Any:
    """Make each dialog method of ``session`` take its ``dialog_lock``.

    The methods are replaced on the instance. Only a real method is wrapped,
    so a session already done is left as it is (its wrappers are functions),
    and so is a stand-in whose attributes are not methods (a test's mock).

    Returns:
        ``session``.
    """
    for name in DIALOG_METHODS:
        original = getattr(session, name, None)
        if not inspect.ismethod(original):
            continue

        @functools.wraps(original)
        async def locked(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            async with dialog_lock(session):
                return await _original(*args, **kwargs)

        setattr(session, name, locked)
    return session
