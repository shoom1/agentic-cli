"""A manager's sub-agent runs, kept by conversation.

Each run is named readably within its conversation: its instance
``<agent>-<n>`` (``researcher-2``, numbered per agent, past the instances
the conversation's store already keeps: ``number_from``) and its run
``<instance>#<n>`` (a first run is ``#1``, each follow-up the next number).
A run is one piece of work; an instance outlives its runs and is ``idle``
between them (``InstanceRecord``: ``running``, ``idle``, ``stopped`` when the
user stopped it, ``closed``). The framework names a first run as it starts it
(``agents/delegate.py::Delegations``) with ``run_for``, keyed by
(conversation, parent run, the parent's call id): the same key names the same
run until it has ended, and a new one after (a provider that numbers its call
ids repeats them every turn); an instance's next run is named by ``next_run``.

The records are written from the runs' own events (``AGENT_RUN``: started,
then how it works while it does, then one end status, carrying the status it
leaves its instance in; ``AGENT_INSTANCE`` a change without a run), which
``Delegations`` hands to ``observe`` as it makes them, under the turn they
are reported in. The one other writer is ``interrupt_open``, for the runs a
turn leaves open when it is stopped or fails, when no event can reach its
consumer any more. A conversation is a ``SessionRef``, or None for runs
started outside a manager's turn.

The registry is in memory; a conversation's records are kept beside it in
its store as well (``BaseWorkflowManager``'s store hooks): ``on_change`` is
told each time a conversation's records change, ``export`` gives them as
JSON-safe data (every instance, every run's record, the id counters,
``RECORDS_VERSION``) and ``restore`` reads that back when the conversation
is loaded again (after a restart, or by another manager); a run found still
working ends ``interrupted``, since the conversation's lock means the
process that ran it is gone.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
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

#: An instance's statuses. It is ``running`` while a run of it works, ``idle``
#: between runs, ``stopped`` once the user stopped its run (only the user
#: continues it) and ``closed`` when it can take no more.
INSTANCE_RUNNING = "running"
INSTANCE_IDLE = "idle"
INSTANCE_STOPPED = "stopped"
INSTANCE_CLOSED = "closed"
INSTANCE_STATUSES = frozenset(
    {INSTANCE_RUNNING, INSTANCE_IDLE, INSTANCE_STOPPED, INSTANCE_CLOSED}
)

#: The format of a conversation's records as ``AgentRuns.export`` writes them
#: (their ``v``); records of another are not read (``restore``).
RECORDS_VERSION = 1

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
        input: A follow-up's message as written; ``""`` for a first run,
            whose input is its brief.
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
    input: str = ""


@dataclass
class InstanceRecord:
    """One sub-agent instance of a conversation, as its runs' events told it.

    Attributes:
        instance: Its id (``researcher-2``).
        agent: Its type.
        parent: The instance that started it (``root`` for the root's
            children and the user's).
        depth: How far below the root it is.
        inherit_context: Whether it sees its parent's conversation.
        title: Its first brief's title.
        status: One of ``INSTANCE_STATUSES``.
        runs: Its run ids, oldest first.
        continuable: Whether its type takes follow-ups, as it was when it started.
        tools: The type's own tools its first run's delegation narrowed it to
            (``task``'s ``tools``; empty for none of them); None for all of
            them. Every run of it keeps them.
        model: The model choice its first run's delegation ran it on, as the
            type writes it (``task``'s ``model``); None for the type's own
            model. Every run of it keeps it.
        based_on: For an instance of the built-in ``general_purpose`` type,
            the type that started it, whose tools and model it has; None for
            any other. Every run of it keeps it.
    """

    instance: str
    agent: str
    parent: str
    depth: int
    inherit_context: bool
    title: str
    status: str
    runs: list[str] = field(default_factory=list)
    continuable: bool = True
    tools: tuple[str, ...] | None = None
    model: str | None = None
    based_on: str | None = None


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
    """A manager's sub-agent runs, kept by conversation, in memory, and beside the conversation in its store.

    Args:
        on_change: Told the conversation each time its records change (a
            run's event that changes them, ``interrupt_open``), never by
            ``restore``: what a store's writes follow. None: nobody is told.
    """

    def __init__(self, *, on_change: Callable[[SessionRef | None], None] | None = None) -> None:
        self._on_change = on_change
        # (conversation, parent run, call id) -> the run that call started (a
        # start with no call, the user's, has no key).
        self._runs: dict[tuple[Any, str | None, str], RunRef] = {}
        # (conversation, agent) -> how many instances of it are numbered.
        self._numbered: dict[tuple[Any, str], int] = {}
        # Each conversation's records, by run id, in start order.
        self._records: dict[Any, dict[str, RunRecord]] = {}
        # Each conversation's instances, by id, in start order.
        self._instances: dict[Any, dict[str, InstanceRecord]] = {}
        # Each conversation's runs by the order their ends were recorded in
        # (1, 2, ...): what ranks instances by when their last run ended; and
        # how many numbers each conversation has given (only ever growing).
        self._ended_in: dict[Any, dict[str, int]] = {}
        self._end_numbers: dict[Any, int] = {}

    def run_for(
        self,
        conversation: SessionRef | None,
        parent: RunRef | None,
        call_id: str | None,
        agent: str,
    ) -> RunRef:
        """The run the call ``call_id`` of ``parent``'s starts, of ``agent``.

        The same key names the same run while it is open; one that has ended
        is followed by a new instance of ``agent`` (numbered per agent, per
        conversation, from 1) and its run ``#1``. ``parent`` None, or the
        root's own (``root_run``), is the root. A start with no call
        (``call_id`` None: the user's) has no key: each is a new instance.
        """
        parent_run = None if parent is None or parent.is_root else parent.run
        key = (conversation, parent_run, call_id)
        ref = self._runs.get(key) if call_id is not None else None
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
        if call_id is not None:
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

    def next_run(
        self, conversation: SessionRef | None, instance: str, *, parent_run: str | None
    ) -> RunRef:
        """The next run of ``instance``: ``<instance>#<n+1>``, of the same agent and depth.

        ``parent_run`` is the run that sends it (None for the root's own
        message, or the user's). Raises ``KeyError`` for an unknown instance.
        """
        record = self._instances[conversation][instance]
        last = max((int(n) for n in map(run_number, record.runs) if n.isdigit()), default=0)
        return RunRef(
            agent=record.agent,
            instance=instance,
            run=_run_id(instance, last + 1),
            parent_run=parent_run,
            depth=record.depth,
        )

    def instances(self, conversation: SessionRef | None) -> list[InstanceRecord]:
        """``conversation``'s instances, in the order they started."""
        return list(self._instances.get(conversation, {}).values())

    def instance(self, conversation: SessionRef | None, instance: str) -> InstanceRecord | None:
        """One instance of ``conversation``, or None."""
        return self._instances.get(conversation, {}).get(instance)

    def observe(self, event: WorkflowEvent) -> None:
        """Record what a run's ``AGENT_RUN`` event says, under the turn it is reported in.

        A start makes the run's record (its call, title, brief, input, who
        started it and its mode), ``running``, and makes its instance or
        reopens it (``running``); a working status (``queued``,
        ``running``, ``input_required``) becomes a working record's status;
        an end gives it its status, time, result, error, whether it is
        partial, what it used, who cancelled it and its own last plan and
        tasks, and sets its instance's status from the end's
        ``instance_status`` (``idle`` without one; an end whose start was not
        seen makes the record and the instance too). ``AGENT_INSTANCE`` sets a
        known instance's status without a run. Any other event changes
        nothing, and neither does a working status of a run that has ended
        or was never started.
        """
        ref = event.run
        if ref is None or ref.run is None:
            return
        conversation = get_active_turn()
        metadata = event.metadata
        if event.type is EventType.AGENT_INSTANCE:
            known = self._instances.get(conversation, {}).get(ref.instance)
            if (
                known is not None and metadata.get("status") in INSTANCE_STATUSES
                and known.status != metadata["status"]
            ):
                known.status = metadata["status"]
                self._changed(conversation)
            return
        if event.type is not EventType.AGENT_RUN:
            return
        records = self._records.setdefault(conversation, {})
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
                input=metadata.get("input") or "",
            )
            instance = self._instance_of(conversation, ref, metadata)
            if ref.run not in instance.runs:
                instance.runs.append(ref.run)
            instance.status = INSTANCE_RUNNING
            self._changed(conversation)
            return
        if status in RUN_WORKING_STATUSES:
            working = records.get(ref.run)
            if (
                working is not None and working.status not in RUN_END_STATUSES
                and working.status != status
            ):
                working.status = status
                self._changed(conversation)
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
        self._number_end(conversation, ref.run)
        instance = self._instance_of(conversation, ref, metadata)
        if ref.run not in instance.runs:
            instance.runs.append(ref.run)
        left = metadata.get("instance_status")
        instance.status = left if left in INSTANCE_STATUSES else INSTANCE_IDLE
        self._changed(conversation)

    def end_order(self, conversation: SessionRef | None, run: str) -> int:
        """Where ``run``'s end was recorded in ``conversation``: 1 for the first, 0 for none yet.

        The order the ends were recorded in, not their times: the clock may
        repeat or step back.
        """
        return self._ended_in.get(conversation, {}).get(run, 0)

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
        ``run_usage``). Their instances are left ``idle``, or ``closed`` for
        a type that is not continuable.
        """
        now = datetime.now()
        ended: list[RunRef] = []
        for record in self._records.get(conversation, {}).values():
            if record.status not in RUN_END_STATUSES:
                record.status = RUN_INTERRUPTED
                record.ended = now
                self._number_end(conversation, record.ref.run or "")
                known = self._instances.get(conversation, {}).get(record.ref.instance)
                if known is not None and known.status == INSTANCE_RUNNING:
                    known.status = INSTANCE_IDLE if known.continuable else INSTANCE_CLOSED
                if usage is not None and record.ref.run is not None:
                    record.usage = usage(record.ref.run)
                ended.append(record.ref)
        if ended:
            self._changed(conversation)
        return ended

    def export(self, conversation: SessionRef | None) -> dict[str, Any]:
        """``conversation``'s records as JSON-safe data, for its store.

        ``{"v": RECORDS_VERSION, "saved": <now, ISO>, "numbered": {type: n},
        "instances": [...], "runs": [...]}``: each instance's fields
        (``InstanceRecord``, its ``tools`` a list) and each run's (``RunRecord``, its
        ``ref``'s fields first, its times as ISO text, and ``end``: where its end was
        recorded, 0 for none yet), in the order they started, and the
        highest instance number of each type. ``saved`` is when this was
        taken: a run still working then is restored as ended at that time.
        """
        return {
            "v": RECORDS_VERSION,
            "saved": datetime.now().isoformat(),
            "numbered": {
                agent: count for (owner, agent), count in self._numbered.items()
                if owner == conversation
            },
            "instances": [_instance_value(one) for one in self.instances(conversation)],
            "runs": [
                _run_value(record, self.end_order(conversation, record.ref.run or ""))
                for record in self.records(conversation)
            ],
        }

    def restore(self, conversation: SessionRef | None, data: Mapping[str, Any]) -> list[str]:
        """Replace ``conversation``'s records with what ``data`` (an ``export``) holds.

        A run found working (``queued``, ``running``, ``input_required``, or
        never past ``started``) ends ``interrupted`` (``INTERRUPTED_ERROR``)
        at the later of its start and the time ``data`` was saved, and an
        instance found ``running`` becomes ``idle`` (``closed`` for a type
        that is not continuable, as ``interrupt_open`` leaves it): the
        process that ran them is gone. The counters
        are the records', never below an instance they keep. Nobody is told
        (``on_change``): this is no run's change. Other conversations'
        records are left as they are.

        Returns:
            The runs it found working, in the order they started.

        Raises:
            ValueError: ``data`` is no records of ``RECORDS_VERSION`` (another
                version, or one that does not read): nothing is changed.
        """
        from agentic_cli.workflow.agents.envelope import INTERRUPTED_ERROR

        if not isinstance(data, Mapping) or data.get("v") != RECORDS_VERSION:
            version = data.get("v") if isinstance(data, Mapping) else None
            raise ValueError(
                f"Sub-agent records of version {version!r}; this version reads {RECORDS_VERSION}."
            )
        try:
            saved = _time(data.get("saved"))
            numbered = _counts(data.get("numbered"))
            instances = [_instance_from(value) for value in _list(data.get("instances"))]
            ends = {}
            records = []
            for value in _list(data.get("runs")):
                record = _run_from(value)
                records.append(record)
                ends[record.ref.run or ""] = _get(value, "end", int)
            interrupted: list[str] = []
            for record in records:
                if record.status not in RUN_END_STATUSES:
                    record.status = RUN_INTERRUPTED
                    record.error = INTERRUPTED_ERROR
                    record.ended = max(record.started, saved)
                    interrupted.append(record.ref.run or "")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Sub-agent records that do not read: {error}") from None
        for instance in instances:
            if instance.status == INSTANCE_RUNNING:
                instance.status = INSTANCE_IDLE if instance.continuable else INSTANCE_CLOSED
            agent, _, number = instance.instance.rpartition("-")
            if agent == instance.agent and number.isascii() and number.isdigit():
                numbered[agent] = max(numbered.get(agent, 0), int(number))
        self.forget(conversation)
        self._records[conversation] = {record.ref.run or "": record for record in records}
        self._instances[conversation] = {instance.instance: instance for instance in instances}
        for agent, count in numbered.items():
            self._numbered[(conversation, agent)] = count
        # The ends are numbered as they were recorded (a run found working
        # ends after them, in start order): what ranks instances by their last
        # run's end, which the clock may not tell.
        recorded = sorted(
            (r for r in records if ends[r.ref.run or ""] > 0),
            key=lambda r: ends[r.ref.run or ""],
        )
        for record in [*recorded, *(r for r in records if r.ref.run in interrupted)]:
            self._number_end(conversation, record.ref.run or "")
        return interrupted

    def forget(self, conversation: SessionRef | None) -> None:
        """Drop ``conversation``'s runs, records, instances and numbering."""
        self._records.pop(conversation, None)
        self._instances.pop(conversation, None)
        self._ended_in.pop(conversation, None)
        self._end_numbers.pop(conversation, None)
        self._runs = {key: ref for key, ref in self._runs.items() if key[0] != conversation}
        self._numbered = {
            key: count for key, count in self._numbered.items() if key[0] != conversation
        }

    def _number_end(self, conversation: SessionRef | None, run: str) -> None:
        """Give ``run``'s end the next number of ``conversation``: a run recorded
        ended twice takes a new one, and no end shares another's."""
        number = self._end_numbers.get(conversation, 0) + 1
        self._end_numbers[conversation] = number
        self._ended_in.setdefault(conversation, {})[run] = number

    def _instance_of(
        self, conversation: SessionRef | None, ref: RunRef, metadata: Mapping[str, Any]
    ) -> InstanceRecord:
        """``ref``'s instance, made from a first run's event when it is new."""
        instances = self._instances.setdefault(conversation, {})
        record = instances.get(ref.instance)
        if record is None:
            tools = metadata.get("tools")
            record = instances[ref.instance] = InstanceRecord(
                instance=ref.instance,
                agent=ref.agent,
                parent=run_instance(ref.parent_run) if ref.parent_run else "root",
                depth=ref.depth,
                inherit_context=bool(metadata.get("inherit_context", False)),
                title=metadata.get("title") or "",
                status=INSTANCE_RUNNING,
                continuable=bool(metadata.get("continuable", True)),
                tools=tuple(tools) if tools is not None else None,
                model=metadata.get("model"),
                based_on=metadata.get("based_on"),
            )
        return record

    def _changed(self, conversation: SessionRef | None) -> None:
        if self._on_change is not None:
            self._on_change(conversation)

    def _ended(self, conversation: SessionRef | None, ref: RunRef) -> bool:
        record = self._records.get(conversation, {}).get(ref.run or "")
        return record is not None and record.status in RUN_END_STATUSES


def stored_instance_ids(data: Any) -> list[str] | None:
    """The instance ids a conversation's stored records name (an ``export``), in
    the order they started; None for data that is no records of
    ``RECORDS_VERSION``. What a store reads to find the conversation's
    transcripts without listing every session it keeps."""
    if not isinstance(data, Mapping) or data.get("v") != RECORDS_VERSION:
        return None
    instances = data.get("instances")
    if not isinstance(instances, list):
        return None
    return [
        value["instance"] for value in instances
        if isinstance(value, Mapping) and isinstance(value.get("instance"), str)
    ]


def _instance_value(instance: InstanceRecord) -> dict[str, Any]:
    """An instance's record as ``export`` writes it: JSON-safe, its tools a list."""
    value = asdict(instance)
    if instance.tools is not None:
        value["tools"] = list(instance.tools)
    return value


def _run_value(record: RunRecord, end: int) -> dict[str, Any]:
    """A run's record as ``export`` writes it: JSON-safe, ``end`` where its end was recorded."""
    ref = record.ref
    return {
        "run": ref.run,
        "agent": ref.agent,
        "instance": ref.instance,
        "parent_run": ref.parent_run,
        "depth": ref.depth,
        "call_id": record.call_id,
        "title": record.title,
        "status": record.status,
        "started": record.started.isoformat(),
        "ended": record.ended.isoformat() if record.ended is not None else None,
        "result": record.result,
        "error": record.error,
        "partial": record.partial,
        "usage": {model: dict(totals) for model, totals in record.usage.items()},
        "brief": dict(record.brief),
        "started_by": record.started_by,
        "mode": record.mode,
        "cancelled_by": record.cancelled_by,
        "plan": record.plan,
        # An app's tool may write any value in a task: kept as JSON says it.
        "tasks": json.loads(json.dumps(record.tasks, default=str)),
        "input": record.input,
        "end": end,
    }


_NONE = type(None)


def _get(value: Mapping[str, Any], name: str, *kinds: type) -> Any:
    """``value[name]``, which must be of one of ``kinds`` exactly (no bool for an int)."""
    found = value[name]
    if type(found) not in kinds:
        raise TypeError(f"{name} is {type(found).__name__}")
    return found


def _list(value: Any) -> list[Any]:
    if type(value) is not list:
        raise TypeError("not a list")
    return value


def _time(value: Any) -> datetime:
    if type(value) is not str:
        raise TypeError("a time that is not text")
    found = datetime.fromisoformat(value)
    if found.tzinfo is not None:
        raise ValueError("a time with a zone")  # every time kept is the app's own clock
    return found


def _counts(value: Any) -> dict[str, int]:
    if type(value) is not dict or not all(
        type(key) is str and type(count) is int for key, count in value.items()
    ):
        raise TypeError("counters that are not numbers by type")
    return dict(value)


def _instance_from(value: Any) -> InstanceRecord:
    if not isinstance(value, Mapping):
        raise TypeError("an instance that is not an object")
    status = _get(value, "status", str)
    runs = _list(value["runs"])
    if status not in INSTANCE_STATUSES or not all(type(run) is str for run in runs):
        raise ValueError(f"an instance's status {status!r} or runs")
    return InstanceRecord(
        instance=_get(value, "instance", str),
        agent=_get(value, "agent", str),
        parent=_get(value, "parent", str),
        depth=_get(value, "depth", int),
        inherit_context=_get(value, "inherit_context", bool),
        title=_get(value, "title", str),
        status=status,
        runs=list(runs),
        continuable=_get(value, "continuable", bool),
        # Written since records first kept them: missing in older ones (all
        # of the type's tools, its own model, no general-purpose spawner).
        tools=_tools(value.get("tools")),
        model=_get_new(value, "model", str, _NONE),
        based_on=_get_new(value, "based_on", str, _NONE),
    )


def _tools(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if type(value) is not list or not all(type(name) is str for name in value):
        raise TypeError("tools that are not a list of names")
    return tuple(value)


def _get_new(value: Mapping[str, Any], name: str, *kinds: type) -> Any:
    """``value[name]`` as ``_get`` reads it, None when it is missing (a field
    added since a version's first records)."""
    return _get(value, name, *kinds) if name in value else None


def _run_from(value: Any) -> RunRecord:
    if not isinstance(value, Mapping):
        raise TypeError("a run that is not an object")
    usage = _get(value, "usage", dict)
    brief = _get(value, "brief", dict)
    tasks = _list(value["tasks"])
    if not all(
        type(model) is str and type(totals) is dict
        and all(type(k) is str and type(n) is int for k, n in totals.items())
        for model, totals in usage.items()
    ) or not all(type(k) is str and type(v) is str for k, v in brief.items()) or not all(
        type(task) is dict for task in tasks
    ):
        raise TypeError("a run's usage, brief or tasks")
    ended = _get(value, "ended", str, _NONE)
    return RunRecord(
        ref=RunRef(
            agent=_get(value, "agent", str),
            instance=_get(value, "instance", str),
            run=_get(value, "run", str),
            parent_run=_get(value, "parent_run", str, _NONE),
            depth=_get(value, "depth", int),
        ),
        call_id=_get(value, "call_id", str, _NONE),
        title=_get(value, "title", str),
        status=_get(value, "status", str),
        started=_time(value["started"]),
        ended=_time(ended) if ended is not None else None,
        result=_get(value, "result", str),
        error=_get(value, "error", str, _NONE),
        partial=_get(value, "partial", bool),
        usage={model: dict(totals) for model, totals in usage.items()},
        brief=dict(brief),
        started_by=_get(value, "started_by", str, _NONE),
        mode=_get(value, "mode", str),
        cancelled_by=_get(value, "cancelled_by", str, _NONE),
        plan=_get(value, "plan", str),
        tasks=[dict(task) for task in tasks],
        input=_get(value, "input", str),
    )
