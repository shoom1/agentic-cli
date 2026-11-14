"""A manager's sub-agent runs, kept by conversation.

Each run is named readably within its conversation: its instance
``<agent>-<n>`` (``researcher-2``, numbered per agent) and its run
``<instance>#<n>`` (each instance runs once: ``#1``). Both backends find a
run where its events come from (ADK's invocation branch, Deep Agents'
stream namespace) and ask ``run_for`` for its name, keyed by (conversation,
parent run, the parent's call id): the same key names the same run until it
has ended, and a new one after (a provider that numbers its call ids repeats
them every turn).

The records are written from the runs' own events (``AGENT_RUN``: started,
then one end status), which the event processors hand to ``observe`` as they
make them, under the turn they are reported in. The one other writer is
``interrupt_open``, for the runs a turn leaves open when it is stopped or
fails, when no event can reach its consumer any more. A conversation is a
``SessionRef``, or None for a processor driven outside a manager. All of it
is in memory, for as long as its manager lives: a backend switch builds a
new manager, and the runs recorded before it do not follow (as with its
usage ledgers).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentic_cli.workflow.agents.outcome import CUT, REFUSAL_ERROR, REFUSED
from agentic_cli.workflow.events import EventType, WorkflowEvent
from agentic_cli.workflow.sessions import RunRef, SessionRef, get_active_turn
from agentic_cli.workflow.tool_summaries import delegation_summary

#: A run's statuses: started, then one end.
RUN_STARTED = "started"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_REFUSED = "refused"
RUN_CANCELLED = "cancelled"
RUN_INTERRUPTED = "interrupted"
RUN_END_STATUSES = frozenset(
    {RUN_COMPLETED, RUN_FAILED, RUN_REFUSED, RUN_CANCELLED, RUN_INTERRUPTED}
)

#: The longest title a run gets: the first line of the request its parent wrote.
TITLE_MAX = 80
#: How much of its result a run's end event, and its record, keep.
RESULT_MAX = 200
#: How a parent is told its sub-agent failed: ADK's delegation tool's words
#: (``_SingleTurnAgentTool``), which Deep Agents' delegation gives too.
SUBAGENT_ERROR_PREFIX = "Error running sub-agent: "


def run_title(request: str) -> str:
    """A run's title: the first non-blank line of its request, cut to ``TITLE_MAX`` characters."""
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


def run_ended(
    ref: RunRef,
    call_id: str | None,
    answer: str,
    *,
    failed: bool = False,
    error: str | None = None,
    outcome: str | None = None,
    usage: dict[str, dict[str, int]] | None = None,
    sealed: bool = False,
) -> WorkflowEvent:
    """The ``AGENT_RUN`` that ends ``ref``'s run, decided as on every backend.

    ``answer`` is the delegation's result as its parent read it; ``failed``
    says it is the framework's own error result (``error``: its error), and
    ``outcome`` is the mark the backend's hook put on the run's last answer
    (``agents/outcome.py``). The run ends:

    - ``interrupted`` for the seal of a call a stopped turn left open
      (``sealed``), with its result;
    - ``failed`` for ``Error running sub-agent: <reason>`` (its error the
      reason), or for the framework's error result (its error, else the
      answer);
    - ``refused`` when the outcome is ``REFUSED``;
    - else ``completed``, ``partial`` when the outcome is ``CUT``.

    The sub-agent's answer is never read for success: each backend tells
    ``failed`` from its own error result alone, so an answer that looks
    like a failed result completed.
    """
    if sealed:
        return WorkflowEvent.agent_run(
            ref, RUN_INTERRUPTED, call_id=call_id, result=result_head(answer), usage=usage,
        )
    if answer.startswith(SUBAGENT_ERROR_PREFIX):
        return WorkflowEvent.agent_run(
            ref, RUN_FAILED, call_id=call_id, result="",
            error=answer.removeprefix(SUBAGENT_ERROR_PREFIX), usage=usage,
        )
    if failed:
        return WorkflowEvent.agent_run(
            ref, RUN_FAILED, call_id=call_id, result="", error=error or answer, usage=usage,
        )
    if outcome == REFUSED:
        return WorkflowEvent.agent_run(
            ref, RUN_REFUSED, call_id=call_id, result=result_head(answer),
            error=REFUSAL_ERROR, usage=usage,
        )
    return WorkflowEvent.agent_run(
        ref, RUN_COMPLETED, call_id=call_id, result=result_head(answer),
        partial=outcome == CUT, usage=usage,
    )


def show_as_delegation(result: WorkflowEvent, answer: str, ended: WorkflowEvent) -> None:
    """Show a delegation's ``TOOL_RESULT`` as every backend shows it.

    ``ended`` is the ``AGENT_RUN`` the result ends the run with, and
    ``answer`` what the sub-agent answered, as its parent read it. The
    content becomes ``tool_summaries.delegation_summary``'s: the answer's
    first line, or, for a sub-agent that failed, ``Failed: <reason>``, the
    result failing too (``success`` false; its ``error`` the text the parent
    read, as Deep Agents' error result has it). ``metadata["result"]`` stays
    as the backend gave it.

    A result that ends its run ``interrupted`` (ADK's seal of a call a
    stopped turn left open) is left as the tool result made it. No turn
    shows one: ADK writes its seals as the stopped run closes, after the
    turn has stopped reading its events, so only a processor handed a seal
    directly meets the case.
    """
    status = ended.metadata.get("status")
    if status == RUN_INTERRUPTED:
        return
    if status == RUN_FAILED:
        result.content = delegation_summary(answer, error=ended.metadata.get("error") or "")
        result.metadata["success"] = False
        result.metadata["error"] = result.metadata.get("error") or answer
        return
    result.content = delegation_summary(answer)


@dataclass
class RunRecord:
    """One sub-agent run, as its events told it.

    Attributes:
        ref: The run.
        call_id: The parent's call that started it.
        title: The first line of its request.
        status: ``started`` while it runs, then its end status.
        started: When it started.
        ended: When it ended; None while it runs.
        result: The first ``RESULT_MAX`` characters of its result.
        error: Why it ended without completing.
        partial: Whether its result was cut short.
        usage: Its token totals by model (``{"input": ..., "output": ...}``).
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

    def observe(self, event: WorkflowEvent) -> None:
        """Record what a run's ``AGENT_RUN`` event says, under the turn it is reported in.

        A start makes the run's record; an end gives it its status, time,
        result, error, whether it is partial and what it used (an end whose
        start was not seen makes the record too). Any other event changes
        nothing.
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
                status=RUN_STARTED,
                started=event.timestamp,
            )
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
