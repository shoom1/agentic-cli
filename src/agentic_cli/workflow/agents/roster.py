"""The main agent's roster of its open sub-agents.

A main agent that can continue its sub-agents (``message_agent``) is told which
of its own are open: a list after the memory index, one line per instance that
is idle, stopped or running, newest first, with its type, status and title and
the first line of its last result. Closed instances are left out, and so is any
line that would take the text past ``ROSTER_MAX`` characters (the oldest go;
``check_agents`` lists them all).

The roster is a snapshot per conversation (:class:`Rosters`), taken when the
conversation's first request asks for it and dropped where the memory index's
is, so the head of the request stays the same for prompt caching. Between
snapshots it is stale: the tool results and notices in the conversation carry
what changed. ADK puts it in a user content of its own (``RosterPlugin``) and
Deep Agents in the first human message's text blocks (``RosterMiddleware``),
the index ahead of it on both.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from agentic_cli.workflow.agents.runs import (
    INSTANCE_IDLE,
    INSTANCE_RUNNING,
    INSTANCE_STOPPED,
    AgentRuns,
    InstanceRecord,
    RunRecord,
)
from agentic_cli.workflow.sessions import SessionRef, get_active_turn

ROSTER_OPEN = "<agents>"
ROSTER_CLOSE = "</agents>"
ROSTER_HEADER = "Your sub-agents in this conversation (message_agent continues one):"
#: The most characters the roster takes, tags and header included.
ROSTER_MAX = 2_000

#: The statuses of the instances it lists.
_LISTED = frozenset({INSTANCE_RUNNING, INSTANCE_IDLE, INSTANCE_STOPPED})
#: The main agent's instance id: the parent of its own sub-agents.
_ROOT = "root"


def _last_result(instance: InstanceRecord, records: Mapping[str, RunRecord]) -> str:
    """The first non-blank line of the newest run's result that has one."""
    for run in reversed(instance.runs):
        record = records.get(run)
        if record is None:
            continue
        for line in record.result.splitlines():
            if line.strip():
                return line.strip()
    return ""


def _line(instance: InstanceRecord, records: Mapping[str, RunRecord]) -> str:
    line = f"- {instance.instance} ({instance.agent}, {instance.status}): {instance.title}"
    result = _last_result(instance, records)
    return f"{line} — {result}" if result else line


def roster_text(instances: Sequence[InstanceRecord], records: Mapping[str, RunRecord]) -> str:
    """The roster of ``instances`` (in the order they started), or ``""`` when none is open.

    Args:
        instances: The instances to list.
        records: Their runs' records, by run id.
    """
    size = len(ROSTER_OPEN) + len(ROSTER_HEADER) + len(ROSTER_CLOSE) + 2
    lines: list[str] = []
    for instance in reversed(instances):
        if instance.status not in _LISTED:
            continue
        line = _line(instance, records)
        if size + len(line) + 1 > ROSTER_MAX:
            break
        lines.append(line)
        size += len(line) + 1
    if not lines:
        return ""
    return "\n".join([ROSTER_OPEN, ROSTER_HEADER, *lines, ROSTER_CLOSE])


class Rosters:
    """The roster of each conversation, taken once and kept.

    Attributes:
        snapshots: The roster each conversation took, by conversation. ``""``
            is a snapshot too (no open instance): the conversation does not
            ask again until the snapshot is dropped.
    """

    def __init__(self, runs: Callable[[], AgentRuns]) -> None:
        """Hold no snapshot yet; ``runs`` gives the records a snapshot is taken from."""
        self._runs = runs
        self.snapshots: dict[SessionRef, str] = {}

    def now(self) -> str:
        """The roster the turn's conversation carries, taken the first time it is
        asked for; ``""`` for none (and outside a turn).

        It reads the records in memory only.
        """
        ref = get_active_turn()
        if ref is None:
            return ""
        text = self.snapshots.get(ref)
        if text is None:
            runs = self._runs()
            text = roster_text(
                [i for i in runs.instances(ref) if i.parent == _ROOT],
                {r.ref.run: r for r in runs.records(ref) if r.ref.run},
            )
            self.snapshots[ref] = text
        return text

    def forget(self, ref: SessionRef | None = None) -> None:
        """Drop a conversation's snapshot (the turn's by default): its next request takes a new one."""
        ref = ref or get_active_turn()
        if ref is not None:
            self.snapshots.pop(ref, None)

    def clear(self) -> None:
        """Drop every conversation's snapshot."""
        self.snapshots.clear()
