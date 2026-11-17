"""A delegation's result envelope: how a sub-agent run's end reaches its parent.

A blocking delegation's tool result is its run's envelope::

    {"success": true, "agent": "researcher-2", "run": "researcher-2#1",
     "status": "completed", "partial": false, "result": "<final text>", "error": null}

``success`` is true only for a completed run, partial or not; any other end
(``failed``, ``refused``, ``cancelled``, ``interrupted``) has a one-line
``error`` and whatever text the run produced in ``result``, and a cancelled
run's also says who cancelled it (``cancelled_by``: ``parent``, ``user`` or
``limit``). ``read_envelope`` finds an envelope in a delegation's result as
either backend's ``TOOL_RESULT`` carries it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agentic_cli.workflow.agents.runs import RUN_CANCELLED, RUN_COMPLETED, RUN_END_STATUSES
from agentic_cli.workflow.sessions import RunRef

#: A run that reached its most model calls: completed, its answer partial.
STEP_LIMIT_ERROR = "step limit reached"
#: A run cancelled at its time limit.
TIME_LIMIT_ERROR = "time limit reached"
#: A run cancelled at its token limit.
TOKEN_LIMIT_ERROR = "token limit reached"
#: A run its turn stopped (Ctrl+C), or a process or a manager that ended under it.
INTERRUPTED_ERROR = "interrupted before it finished"
#: Who may cancel a run: its parent, the user, or one of its limits.
CANCELLED_BY = frozenset({"parent", "user", "limit"})

#: The keys every envelope has, in the order it lists them.
_KEYS = ("success", "agent", "run", "status", "partial", "result", "error")


def envelope(
    ref: RunRef,
    status: str,
    *,
    result: str = "",
    partial: bool = False,
    error: str | None = None,
    cancelled_by: str | None = None,
) -> dict[str, Any]:
    """The envelope of the run ``ref``, ended with ``status``.

    ``error`` is made one line (its whitespace runs, line breaks among them,
    each one space). ``cancelled_by`` is given exactly when ``status`` is
    ``cancelled``.

    Raises:
        ValueError: ``ref`` is the root's own work (no run), ``status`` is
            no run's end, or ``cancelled_by`` is missing, misplaced or not
            one of ``CANCELLED_BY``.
    """
    if ref.run is None:
        raise ValueError("An envelope is a sub-agent run's; the root's own work has none.")
    if status not in RUN_END_STATUSES:
        raise ValueError(
            f"An envelope is made at a run's end: {status!r} is not one of "
            f"{', '.join(sorted(RUN_END_STATUSES))}."
        )
    if (status == RUN_CANCELLED) != (cancelled_by is not None):
        raise ValueError(
            "cancelled_by names who cancelled a cancelled run, and only a cancelled run: "
            f"status {status!r}, cancelled_by {cancelled_by!r}."
        )
    if cancelled_by is not None and cancelled_by not in CANCELLED_BY:
        raise ValueError(
            f"cancelled_by must be one of {', '.join(sorted(CANCELLED_BY))}, not {cancelled_by!r}."
        )
    value: dict[str, Any] = {
        "success": status == RUN_COMPLETED,
        "agent": ref.instance,
        "run": ref.run,
        "status": status,
        "partial": partial,
        "result": result,
        "error": None if error is None else " ".join(error.split()),
    }
    if cancelled_by is not None:
        value["cancelled_by"] = cancelled_by
    return value


def read_envelope(value: Any) -> dict[str, Any] | None:
    """The envelope in a delegation's result, as either backend's ``TOOL_RESULT`` carries it.

    ``value`` is a dict (ADK's function response), or the JSON text of one
    (a Deep Agents ``ToolMessage``'s content). A copy of it when it has the
    envelope's keys; None for anything else: the error result of a call
    that started no run, any other result, text that does not parse. Never
    raises.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return None
    if not isinstance(value, Mapping) or not all(key in value for key in _KEYS):
        return None
    return dict(value)
