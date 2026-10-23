"""``compact()`` run as ``/compact`` runs it, for the backends' tests.

The CLI (``MessageProcessor.process_compact``) runs a manager's ``compact``
as the source of a merged stream, inside ``live_events.reporting``, so what
it reports arrives while it runs. ``compact_now`` does the same and collects
what was reported.
"""

from __future__ import annotations

import contextlib
from typing import Any

from agentic_cli.workflow import live_events


async def compact_now(
    manager: Any,
    focus: str | None = None,
    *,
    events: list | None = None,
    session_id: str = "s",
    user_id: str = "u",
) -> tuple[Any, list]:
    """``manager.compact`` on ``(user_id, session_id)``, its reports merged in.

    Every event reported goes to ``events`` (a new list unless given), so a
    test still reads them when ``compact`` raises.

    Returns:
        What ``compact`` returned, and the events it reported.
    """
    events = [] if events is None else events
    outcome: dict[str, Any] = {}

    async def source():
        outcome["result"] = await manager.compact(
            manager.session_ref(session_id, user_id), focus
        )
        if False:  # pragma: no cover - an async generator that yields nothing
            yield

    live = live_events.LiveEvents()
    with live_events.reporting(live):
        async with contextlib.aclosing(live_events.merged(source(), live)) as items:
            async for event in items:
                events.append(event)
    return outcome["result"], events
