"""One dialog at a time on a session.

thinking-prompt shows one dialog and raises on a second ``show_dialog``. A
turn and a command used to run one at a time, in the input handler, so they
never met. A sub-agent's view runs its run outside the handler (the prompt
stays open to steer it), so a command may open a dialog while the run asks
its question, or the run may ask while a command's dialog is open: a second
dialog that raised would be a permission prompt answered by a denial the
user never saw, or a command that fails.

Every dialog of the app takes the session's one lock (``dialog_lock``) for as
long as it is open: whichever comes second waits for the user to answer the
first. The commands show theirs through ``show_dialog``; the message
processor holds the lock around a question's dialog.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import Any

_LOCKS: "weakref.WeakKeyDictionary[Any, asyncio.Lock]" = weakref.WeakKeyDictionary()


def dialog_lock(ui: Any) -> asyncio.Lock:
    """The lock every dialog of the session ``ui`` takes while it is open."""
    lock = _LOCKS.get(ui)
    if lock is None:
        lock = _LOCKS[ui] = asyncio.Lock()
    return lock


async def show_dialog(ui: Any, dialog: Any) -> Any:
    """``ui.show_dialog(dialog)``, once no other dialog of the session is open.

    Returns:
        What the dialog returns (the button's result, or None on Escape).
    """
    async with dialog_lock(ui):
        return await ui.show_dialog(dialog)
