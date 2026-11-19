"""A manager's sub-agent runs, kept by conversation.

Each run is named readably within its conversation: its instance
``<agent>-<n>`` (``researcher-2``, numbered per agent, past the instances
the conversation's store already keeps: ``number_from``) and its run
``<instance>#<n>`` (each instance runs once: ``#1``). The framework names a
run as it starts it (``agents/delegate.py::Delegations``) with ``run_for``,
keyed by (conversation, parent run, the parent's call id): the same key
names the same run until it has ended, and a new one after (a provider that
numbers its call ids repeats them every turn).

The records are written from the runs' own events (``AGENT_RUN``: started,
then how it works while it does, then one end status), which
``Delegations`` hands to ``observe`` as it makes them, under the turn they
are reported in. The one other writer is ``interrupt_open``, for the runs a
turn leaves open when it is stopped or fails, when no event can reach its
consumer any more. A conversation is a ``SessionRef``, or None for runs
started outside a manager's turn. All of it is in memory, for as long as
its manager lives: a backend switch builds a new manager, and the runs
recorded before it do not follow (as with its usage ledgers).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentic_cli.workflow.events import EventType, WorkflowEvent
from agentic_cli.workflow.sessions import RunRef, SessionRef, get_active_turn
from agentic_cli.workflow.tool_summaries import delegation_summary

#: A run's statuses: started, then how it works while it does, then one end.
RUN_STARTED = "started"
#: A working run's status: waiting for a free working slot, working, or
#: waiting on the user (a question, a permission prompt). A record says
#: ``running`` from its start; its events say ``queued`` when it must wait
#: for a slot (to start, or, once the user has answered, to take its own
#: back while another run works in it), and ``running`` once it has one.
RUN_QUEUED = "queued"
RUN_RUNNING = "running"
RUN_INPUT_REQUIRED = "input_required"
RUN_WORKING_STATUSES = frozenset({RUN_QUEUED, RUN_RUNNING, RUN_INPUT_REQUIRED})
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_REFUSED = "refused"
RUN_CANCELLED = "cancelled"
RUN_INTERRUPTED = "interrupted"
RUN_END_STATUSES = frozenset(
    {RUN_COMPLETED, RUN_FAILED, RUN_REFUSED, RUN_CANCELLED, RUN_INTERRUPTED}
)

#: The longest title a run gets: the first line of its brief's title.
TITLE_MAX = 80
#: How much of its result a run's end event, and its record, keep.
RESULT_MAX = 200
#: How ADK's own delegation tool told a parent its sub-agent failed. Kept
#: for callers that read it; the framework's delegation answers with the
#: result envelope (``agents/envelope.py``), its ``error`` the reason.
SUBAGENT_ERROR_PREFIX = "Error running sub-agent: "


def run_title(request: str) -> str:
    """A run's title: the first non-blank line of ``request``, cut to ``TITLE_MAX`` characters."""
    line = next((line.strip() for line in request.splitlines() if line.strip()), "")
    return line[:TITLE_MAX]


def result_head(result: str) -> str:
    """What a run keeps of its result: its first ``RESULT_MAX`` characters."""
    return result[:RESULT_MAX]


def run_label(ref: RunRef | None) -> str | None:
    """How a dialog names a sub-agent run: ``researcher-2 (researcher, run 1)``.

    Its instance, its agent, and its number within the instance (what follows
    ``#`` in ``ref.run``). None for the root's own work (and for none): the
    root's dialogs name no agent.
    """
    if ref is None or ref.is_root:
        return None
    return f"{ref.instance} ({ref.agent}, run {run_number(ref.run)})"


def show_as_delegation(result: WorkflowEvent, envelope: Mapping[str, Any]) -> None:
    """Show a delegation's ``TOOL_RESULT`` from its envelope, the same on every backend.

    ``envelope`` is the run's result envelope (``agents/envelope.py``), as the
    parent read it. The content becomes ``tool_summaries.delegation_summary``'s:
    the result's first line for a run that completed or was refused (the
    refusal notice says why), else ``Failed: <error>``. The result succeeds
    as the envelope does (``success``), its ``error`` the envelope's when it
    does not, and ``metadata["result"]`` is the envelope.
    """
    text = envelope.get("result")
    text = text if isinstance(text, str) else ""
    succeeded = envelope.get("success") is True
    error = envelope.get("error")
    if succeeded or envelope.get("status") == RUN_REFUSED:
        result.content = delegation_summary(text)
    else:
        result.content = delegation_summary(text, error=error if isinstance(error, str) else "")
    result.metadata["success"] = succeeded
    if succeeded:
        result.metadata.pop("error", None)
    elif isinstance(error, str):
        result.metadata["error"] = error
    result.metadata["result"] = dict(envelope)


@dataclass
class RunRecord:
    """One sub-agent run, as its events told it.

    Attributes:
        ref: The run.
        call_id: The parent's call that started it.
        title: Its brief's title.
        status: While it works, ``running``, ``queued`` (waiting for a
            free working slot) or ``input_required`` (waiting on the user);
            then its end status.
        started: When it started.
        ended: When it ended; None while it runs.
        result: The first ``RESULT_MAX`` characters of its result.
        error: Why it ended without completing.
        partial: Whether its result was cut short.
        usage: Its token totals by model (``{"input": ..., "output": ...}``).
        brief: Its brief's filled fields, the title first.
        started_by: Who started it: ``agent:<instance>`` (``agent:root``
            for the root).
        mode: How it runs: ``blocking`` (its parent waits for it).
        cancelled_by: Who cancelled it, for a cancelled run.
        plan: Its own last plan (``""`` for none).
        tasks: Its own last task list (the state tools' list of dicts).
    """

    ref: RunRef
    call_id: str | None
    title: str
    status: str
    started: datetime
    ended: datetime | None = None
    result: str = ""
    error: str | None = None
    partial: bool = False
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    brief: dict[str, str] = field(default_factory=dict)
    started_by: str | None = None
    mode: str = "blocking"
    cancelled_by: str | None = None
    plan: str = ""
    tasks: list[dict[str, Any]] = field(default_factory=list)


def own_plan_and_tasks(state: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """A run's own plan and task list out of ``state``, as both backends read them.

    The plan only when it is a string, the tasks only when they are a list
    of dicts (the state tools' shape): an app's tool may write any value
    under those keys, and ``/agents`` draws what is kept.
    """
    plan = state.get("plan")
    tasks = state.get("tasks")
    return (
        plan if isinstance(plan, str) else "",
        list(tasks) if isinstance(tasks, list) and all(isinstance(t, dict) for t in tasks) else [],
    )


def _run_id(instance: str, number: int) -> str:
    """A run's id: its instance and the run's number there (``researcher-2#1``)."""
    return f"{instance}#{number}"


def run_instance(run: str) -> str:
    """The instance a run id names (``researcher-2#1``: ``researcher-2``).

    The id ``AgentRuns.run_for`` gives, read back: what a job's
    ``started_by`` is matched by when a call's run is named by its id
    (``workflow/interrupts.py``).
    """
    return run.rpartition("#")[0]


def run_number(run: str) -> str:
    """A run's number within its instance, as its id writes it (``researcher-2#1``: ``1``).

    The other half of the id ``AgentRuns.run_for`` gives: how a dialog
    (``run_label``) and ``/agents`` number a run.
    """
    return run.rpartition("#")[2]


class AgentRuns:
    """A manager's sub-agent runs, kept by conversation, in memory for the manager's life."""

    def __init__(self) -> None:
        # (conversation, parent run, call id) -> the run that call started.
        self._runs: dict[tuple[Any, str | None, str], RunRef] = {}
        # (conversation, agent) -> how many instances of it are numbered.
        self._numbered: dict[tuple[Any, str], int] = {}
        # Each conversation's records, by run id, in start order.
        self._records: dict[Any, dict[str, RunRecord]] = {}

    def run_for(
        self, conversation: SessionRef | None, parent: RunRef | None, call_id: str, agent: str
    ) -> RunRef:
        """The run the call ``call_id`` of ``parent``'s starts, of ``agent``.

        The same key names the same run while it is open; one that has ended
        is followed by a new instance of ``agent`` (numbered per agent, per
        conversation, from 1) and its run ``#1``. ``parent`` None, or the
        root's own (``root_run``), is the root.
        """
        parent_run = None if parent is None or parent.is_root else parent.run
        key = (conversation, parent_run, call_id)
        ref = self._runs.get(key)
        if ref is not None and not self._ended(conversation, ref):
            return ref
        count = self._numbered.get((conversation, agent), 0) + 1
        self._numbered[(conversation, agent)] = count
        instance = f"{agent}-{count}"
        ref = RunRef(
            agent=agent,
            instance=instance,
            run=_run_id(instance, 1),
            parent_run=parent_run,
            depth=(parent.depth if parent is not None else 0) + 1,
        )
        self._runs[key] = ref
        return ref

    def number_from(self, conversation: SessionRef | None, counts: Mapping[str, int]) -> None:
        """Number ``conversation``'s next instances of each agent past ``counts``.

        ``counts[agent]`` is the highest instance number of ``agent`` the
        conversation's store already keeps (another manager's: before a
        restart or a switch of backend or store, or another process's between
        two turns), so ``run_for``'s next instance of it
        is ``<agent>-<counts[agent] + 1>`` at the earliest. A count is never
        lowered: an instance this manager numbered already stays numbered,
        and an open run keeps its name.
        """
        for agent, count in counts.items():
            key = (conversation, agent)
            if count > self._numbered.get(key, 0):
                self._numbered[key] = count

    def observe(self, event: WorkflowEvent) -> None:
        """Record what a run's ``AGENT_RUN`` event says, under the turn it is reported in.

        A start makes the run's record (its call, title, brief, who started
        it and its mode), ``running``; a working status (``queued``,
        ``running``, ``input_required``) becomes a working record's status;
        an end gives it its status, time, result, error, whether it is
        partial, what it used, who cancelled it and its own last plan and
        tasks (an end whose start was not seen makes the record too). Any other event changes nothing, and
        neither does a working status of a run that has ended or was never
        started.
        """
        ref = event.run
        if event.type is not EventType.AGENT_RUN or ref is None or ref.run is None:
            return
        records = self._records.setdefault(get_active_turn(), {})
        metadata = event.metadata
        status = metadata.get("status")
        if status == RUN_STARTED:
            records[ref.run] = RunRecord(
                ref=ref,
                call_id=metadata.get("call_id"),
                title=metadata.get("title") or "",
                status=RUN_RUNNING,
                started=event.timestamp,
                brief=dict(metadata.get("brief") or {}),
                started_by=metadata.get("started_by"),
                mode=metadata.get("mode") or "blocking",
            )
            return
        if status in RUN_WORKING_STATUSES:
            working = records.get(ref.run)
            if working is not None and working.status not in RUN_END_STATUSES:
                working.status = status
            return
        if status not in RUN_END_STATUSES:
            return
        record = records.get(ref.run)
        if record is None:
            record = records[ref.run] = RunRecord(
                ref=ref,
                call_id=metadata.get("call_id"),
                title="",
                status=RUN_STARTED,
                started=event.timestamp,
            )
        record.status = status
        record.ended = event.timestamp
        record.result = metadata.get("result") or ""
        record.error = metadata.get("error")
        record.partial = bool(metadata.get("partial"))
        record.usage = dict(metadata.get("usage") or {})
        record.cancelled_by = metadata.get("cancelled_by")
        record.plan, record.tasks = own_plan_and_tasks(metadata)

    def records(self, conversation: SessionRef | None) -> list[RunRecord]:
        """``conversation``'s runs, in the order they started."""
        return list(self._records.get(conversation, {}).values())

    def record(self, conversation: SessionRef | None, run_or_instance: str) -> RunRecord | None:
        """The record of a run (``researcher-1#1``), or of an instance's latest run (``researcher-1``)."""
        records = self._records.get(conversation, {})
        if run_or_instance in records:
            return records[run_or_instance]
        found = [r for r in records.values() if r.ref.instance == run_or_instance]
        return found[-1] if found else None

    def interrupt_open(
        self,
        conversation: SessionRef | None,
        *,
        usage: Callable[[str], dict[str, dict[str, int]]] | None = None,
    ) -> list[RunRef]:
        """End ``conversation``'s runs still open as interrupted; the runs ended.

        For a turn that ends with runs open (a cancel, a failure): no event
        can tell its consumer any more. ``usage`` gives each run's tokens by
        model, as an end event carries them (the conversation's ledger's
        ``run_usage``).
        """
        now = datetime.now()
        ended: list[RunRef] = []
        for record in self._records.get(conversation, {}).values():
            if record.status not in RUN_END_STATUSES:
                record.status = RUN_INTERRUPTED
                record.ended = now
                if usage is not None and record.ref.run is not None:
                    record.usage = usage(record.ref.run)
                ended.append(record.ref)
        return ended

    def forget(self, conversation: SessionRef | None) -> None:
        """Drop ``conversation``'s runs, records and numbering."""
        self._records.pop(conversation, None)
        self._runs = {key: ref for key, ref in self._runs.items() if key[0] != conversation}
        self._numbered = {
            key: count for key, count in self._numbered.items() if key[0] != conversation
        }

    def _ended(self, conversation: SessionRef | None, ref: RunRef) -> bool:
        record = self._records.get(conversation, {}).get(ref.run or "")
        return record is not None and record.status in RUN_END_STATUSES
