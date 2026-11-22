"""A delegation: the one place a sub-agent run starts, ends and is decided.

A parent's ``task`` call reaches ``Delegations.delegate`` from its backend's
task tool (ADK's ``adk/task_tool.py::TaskTool``, Deep Agents'
``deepagents/task_tool.py::build_task_tool``) as a ``DelegationRequest``.
``delegate``:

- reads the brief against what the calling run may start (``brief.read_brief``;
  ``offered``: its own sub-agents, then, above ``max_spawn_depth``, the types
  agents start at run time, ``agents/spawn.py``): a brief that cannot start a run is the call's error result, and nothing
  else happens (no run, no record, no event);

Then ``start``, the one entry for an instance's first run, whoever starts it
(a ``StartRequest``: the type, the brief, who starts it):

- counts the run against its turn's (``SubAgentPolicy.max_runs_per_turn``,
  runs at any depth: ``sessions.runs_this_turn``): a call past it is the
  error result ``RUNS_PER_TURN_ERROR`` says, and nothing else happens;
- names the run (``AgentRuns.run_for``: its instance ``<type>-<n>`` and its run
  ``<instance>#1``, under the caller's run; a turn's first run is numbered
  past the instances its conversation's store keeps, which ``stored`` says
  once per conversation per turn) and reports its ``AGENT_RUN``
  ``started``: the title, the call, the brief, who started it,
  ``blocking``;
- runs it through the backend's hook (``BaseWorkflowManager._run_agent``) in
  an asyncio task of its own, in which the run is the active run
  (``get_active_run``) and no tool call is (``get_active_tool_call``): the
  run's model calls, tool calls, permission checks and questions are its
  own, never its caller's ``task`` call's;
- ends the run as the hook's answer says, reports the end (and a cut run's
  or a failed run's one warning after it), and returns the result envelope
  (``envelope.envelope``) with the shared state the run changed.

A sub-agent is an instance that outlives its run: it is ``idle`` when the run
ends, and ``Delegations.follow_up`` starts its next run (``<instance>#2``)
through the same hook, working slots, limits, events and envelope, with
``AgentRun.follow_up`` set: the instance's transcript exists, and the run's
message is the follow-up as written. One run per instance at a time: a
follow-up to a running instance is refused. A run's end says the status it
leaves its instance in: ``stopped`` for the user's cancel, ``closed`` for a
type that is not continuable (``AgentConfig.continuable``) or when more
instances are open than ``SubAgentPolicy.max_open`` allows (the oldest idle
one is closed), else ``idle``. ``Delegations.close`` closes an idle or
stopped instance and its descendants.

The parent's tools (``agents/follow_ups.py``) call ``Delegations.message``
(an idle instance's follow-up, or a running one's run steered, or with
``interrupt`` cancelled by its parent and followed up), ``stop`` (a running
instance's run cancelled by its parent, leaving it idle; with ``close``, the
instance and its descendants closed, a running one's at its run's end) and
``check`` (a caller's sub-agents), and the user's lines reach a running run
through ``steer``. A message steered to a run that its transcript does not
hold when the run ends (``RunControl.unstored``) is kept for the instance's
next run, which starts with it steered: a message is never lost.

Each run works under its ``RunControl`` (``agents/control.py``,
``AgentRun.control``): its limits (its type's over the policy's), the time
limit running while the hook does, and a cancel of the run alone
(``Delegations.cancel``), which its limits use too.

Each run works in a slot of its conversation's ``SlotPool``
(``agents/slots.py``, ``SubAgentPolicy.max_active`` slots; the root takes
none): it takes one before its hook runs, waiting ``queued`` (then
``running``) when every slot is held, and gives it back at its end. A run
that delegates lends its slot while it waits for its child (one lend for
all the children it waits on at once), and the child that ends last hands
it straight back, so a parent goes on ahead of the runs that have not
started, and nested delegations never wait on each other's slots. A cancel
of a queued run ends it as a cancel of a working run does, and its place
in the queue goes with it. The time a run waits for its slot is not
counted against its time limit; the time it lends its slot is.

A run that waits on the user (``waiting_on_user``: a question, a permission
prompt, ``BaseWorkflowManager.request_user_input`` and the permission
engine) says ``input_required``, lends its slot for the wait, says
``queued`` once answered while it waits for its slot back (another run
works in it), and ``running`` once it has it; a prompt no one can be shown
(no input callback, or its turn ended) lends nothing.

Each ``AGENT_RUN`` is recorded (``AgentRuns.observe``) as it is made and
reported to the turn's live events, so it reaches the turn's stream while
the run works. A run ends:

- ``failed`` when the hook raised (not a cancel) or answered an error: the
  parent's envelope carries the reason (the provider's, for a failed model
  call) and the text the run wrote before it failed, and the user is warned
  once, a recoverable ``ERROR``: ``<instance> failed: <reason>``, or the
  hook's own words (``RunFailed``: a context overflow,
  ``context.subagent_overflow_message``);
- ``refused`` when a model refused its last answer (``agents/outcome.py``);
- ``completed`` otherwise, ``partial`` for an answer cut at its output
  limit, followed by the cut's one warning (``outcome.cut_off_warning``),
  and for a run that made its last allowed model call (its step limit:
  ``error`` "step limit reached", no warning);
- ``cancelled`` when its control cancelled it (``cancelled_by`` its parent,
  the user or a limit: the time limit, the token limit), with what the hook
  had of its work, ``partial`` when that is any text;
- ``interrupted`` when its caller's task is cancelled (Ctrl+C stops the
  turn): the cancel goes on once the run has stopped, whatever else
  stopped it, unless a run above it was cancelled by its control (its
  parent's stop, a limit, the user's stop of that run): then it ends
  ``cancelled`` by its parent, saying so (``_stopped_with_caller``). A
  run's own task cancelled by anything but its control while its caller
  goes on ends ``interrupted`` too, and its caller reads that envelope.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers
from agentic_cli.workflow import live_events
from agentic_cli.workflow.agents.brief import (
    Brief,
    BriefError,
    TaskOffer,
    read_brief,
    task_message,
    task_offer,
)
from agentic_cli.workflow.agents.control import PARENT_CAUSE, RunControl
from agentic_cli.workflow.agents.envelope import (
    CANCELLED_BY,
    INTERRUPTED_ERROR,
    STEP_LIMIT_ERROR,
    envelope,
    open_task_descriptions,
)
from agentic_cli.workflow.agents.outcome import CUT, REFUSAL_ERROR, REFUSED, cut_off_warning
from agentic_cli.workflow.agents.policy import AgentLimits
from agentic_cli.workflow.agents.runs import (
    INSTANCE_CLOSED,
    INSTANCE_IDLE,
    INSTANCE_RUNNING,
    INSTANCE_STOPPED,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_END_STATUSES,
    RUN_FAILED,
    RUN_INPUT_REQUIRED,
    RUN_INTERRUPTED,
    RUN_QUEUED,
    RUN_REFUSED,
    RUN_RUNNING,
    RUN_STARTED,
    AgentRuns,
    InstanceRecord,
    result_head,
    run_title,
)
from agentic_cli.workflow.agents.slots import NEW, Slot, SlotPool
from agentic_cli.workflow.agents.spawn import (
    AGENT_SOURCE,
    depth_sentence_applies,
    may_spawn,
    spawnable,
)
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    count_run_this_turn,
    current_turn_scope,
    get_active_turn,
    release_run_this_turn,
    reset_active_turn,
    runs_this_turn,
    set_active_run,
    set_active_tool_call,
    set_active_turn,
)

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.policy import SubAgentPolicy
    from agentic_cli.workflow.interrupts import OpenCalls
    from agentic_cli.workflow.config import AgentConfig
    from agentic_cli.workflow.usage import UsageLedgers

logger = Loggers.workflow()

#: How a run its parent's ``task`` call starts runs: the parent waits for it.
BLOCKING = "blocking"
#: The longest error text a failed run's log line carries.
_MAX_LOGGED_ERROR = 500
#: A ``task`` call's error result once its turn has started the most runs it
#: may (``SubAgentPolicy.max_runs_per_turn``, the number).
RUNS_PER_TURN_ERROR = (
    "This turn has already started {limit} sub-agent runs, the most one turn may start. "
    "Answer with what you have."
)


#: ``started_by`` of a run the user starts (a follow-up from a view).
USER = "user"
#: What a follow-up to an instance its sender may not continue is told.
NOT_YOURS_ERROR = "'{instance}' is not one of your sub-agents. {yours}"
YOURS = "Yours: {instances}."
NO_SUB_AGENTS = "You have no sub-agents yet."
CLOSED_ERROR = "{instance} is closed; start a new one with task."
STOPPED_ERROR = "{instance} was stopped by the user; only the user can continue it."
RUNNING_ERROR = "{instance} is running: an instance runs one run at a time."
TYPE_GONE_ERROR = "{instance}'s type '{agent}' is not one of this app's agents."
NO_MESSAGE_ERROR = "A follow-up needs a message."
#: A message that is not text, or is blank.
MESSAGE_NEEDED_ERROR = "message_agent needs a message."
#: The note of a steered message's result.
STEERED_NOTE = "Your message was given to its run before its next step."
#: A run its parent stopped (``stop_agent``), and one it replaced with a message.
STOPPED_BY_PARENT_ERROR = "stopped by its parent"
REPLACED_ERROR = "stopped by its parent for a new message"
#: A run the user replaced with a message of theirs.
REPLACED_BY_USER_ERROR = "stopped by the user for a new message"
#: Why a parent's ``stop_agent`` closes an instance.
CLOSED_BY_PARENT = "closed by its parent"
#: Why the user stopped an instance: its run's envelope's ``error``, or, for
#: an idle one, its ``AGENT_INSTANCE``'s reason (``stop_by_user``).
STOPPED_BY_USER_ERROR = "stopped by the user"
#: Why the user closed an instance (``AGENT_INSTANCE``'s reason; ``close_by_user``).
CLOSED_BY_USER = "closed by the user"
#: The error of a run that ended with a parent's run its control stopped.
WITH_ITS_PARENT_ERROR = "its parent's run was stopped"
#: Who cancels a run for its parent's tools (``cancelled_by``).
PARENT = "parent"
#: How many lines of a run's result ``check`` shows.
_CHECK_LINES = 3
#: Why an instance past ``SubAgentPolicy.max_open`` is closed.
MAX_OPEN_REASON = "more than {limit} sub-agents open: the oldest idle one is closed"
#: The statuses of an instance that waits between its runs.
_OPEN = frozenset({INSTANCE_IDLE, INSTANCE_STOPPED})


def started_by(caller: RunRef | None) -> str:
    """Who starts a run its parent delegated: ``agent:<the caller's instance>``.

    ``agent:root`` for the root's own calls (no run, or ``root_run``).
    """
    return f"agent:{caller.instance if caller is not None else 'root'}"


def _refused(error: str) -> Delegated:
    """The error result of a call that starts no run and steers none."""
    return Delegated(result={"success": False, "error": error}, state={}, run=None)


def _not_yours(instance: str, yours: Sequence[InstanceRecord]) -> str:
    """What a caller is told of an instance that is not its own sub-agent."""
    names = ", ".join(i.instance for i in yours)
    return NOT_YOURS_ERROR.format(
        instance=instance, yours=YOURS.format(instances=names) if yours else NO_SUB_AGENTS
    )


@dataclass(frozen=True)
class DelegationRequest:
    """A ``task`` call, as its backend's task tool hands it over.

    Attributes:
        caller: The run that made the call (``get_active_run()`` at the
            call): None, or ``root_run``, for the root.
        caller_agent: The calling agent's type name.
        call_id: The caller's ``task`` call.
        arguments: ``task``'s arguments as the model wrote them.
        state: The shared state at the call (the outside-content mark),
            which the run starts from.
        history: Called once, only when the chosen type inherits its
            parent's conversation: that conversation, as the backend reads it.
    """

    caller: RunRef | None
    caller_agent: str
    call_id: str
    arguments: Mapping[str, Any]
    state: Mapping[str, Any] = field(default_factory=dict)
    history: Callable[[], Any] | None = None


@dataclass(frozen=True)
class StartRequest:
    """A first run of an agent type, as whoever starts it hands it over.

    Who may start the type is the asker's to check (``Delegations.offered``
    for a ``task`` call); ``Delegations.start`` runs what it is given.

    Attributes:
        agent_type: The type to run: one of the manager's agent types.
        brief: The run's brief (a title and an objective at least).
        started_by: Who starts it: ``agent:<instance>`` (``agent:root`` for
            the root), ``skill:<name>`` or ``user``.
        caller: The run that starts it; None for the root, or the user.
        call_id: The caller's tool call; None for the user's start.
        conversation: The conversation; None: the active turn's.
        state: The caller's shared state (the outside-content mark), which
            the run starts from.
        history: Called once, only when the type inherits its parent's
            conversation: that conversation, as the backend reads it.
        counted: Whether the run counts toward its turn's runs
            (``SubAgentPolicy.max_runs_per_turn``); False for a run the
            user starts.
    """

    agent_type: str
    brief: Brief
    started_by: str
    caller: RunRef | None = None
    call_id: str | None = None
    conversation: SessionRef | None = None
    state: Mapping[str, Any] = field(default_factory=dict)
    history: Callable[[], Any] | None = None
    counted: bool = True


@dataclass(frozen=True)
class AgentRun:
    """What the backend hook is asked to run.

    Attributes:
        ref: The run.
        config: Its agent type.
        message: The task message (``brief.task_message``): the run's first
            user message; a follow-up's, the message as written.
        conversation: The conversation it runs in.
        state: The shared state it starts from.
        history: An inheriting run's parent conversation, as the backend
            reads it; None for a fresh one.
        control: Its control (``agents/control.py``): its limits, a cancel
            of the run alone, the messages steered to it. A run made
            without one gets one with no limits.
        follow_up: Whether this is an instance's next run: its transcript
            exists, ``message`` is the follow-up as written (no task
            message), ``history`` is None, and ``state`` is the caller's
            shared state (the outside-content mark).
    """

    ref: RunRef
    config: AgentConfig
    message: str
    conversation: SessionRef | None
    state: Mapping[str, Any]
    history: Any = None
    control: RunControl = None  # type: ignore[assignment]  # filled below when None
    follow_up: bool = False

    def __post_init__(self) -> None:
        if self.control is None:
            object.__setattr__(self, "control", RunControl(self.ref, AgentLimits()))


@dataclass(frozen=True)
class FollowUpRequest:
    """A follow-up to an instance: the message that starts its next run.

    Attributes:
        instance: The instance to continue (``researcher-2``).
        message: The follow-up, as written.
        started_by: Who sends it: ``agent:<instance>`` (``agent:root`` for
            the root) or ``USER``.
        caller: The run that sends it; None for the root's own call or the user.
        call_id: The caller's tool call; None for the user.
        conversation: The conversation; None: the active turn's.
        state: The caller's shared state (the outside-content mark), which
            the run starts from.
    """

    instance: str
    message: str
    started_by: str
    caller: RunRef | None = None
    call_id: str | None = None
    conversation: SessionRef | None = None
    state: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunAnswer:
    """What the backend hook returns for a run it ran to its end.

    Attributes:
        text: Its last non-empty answer (``""`` for none), one it wrote
            before it failed too: a failed run's envelope carries it.
        outcome: How that answer ended (``agents/outcome.py``: ``REFUSED``,
            ``CUT``), None when it ended as an answer should.
        error: Why it failed, when it did (the provider's reason for a
            failed model call, ``failure_reason``); None when it answered.
        state: The shared state keys it changed, with their new values (the
            outside-content mark: a run's plan and tasks are its own).
        plan: Its plan as it left it (``""`` for none): a run's own, never
            written back to its caller.
        tasks: Its task list as it left it (the state tools' list of dicts).
            Both are filled for a run that completed and for one the hook
            answered as failed (``error``); a run that raised (``RunFailed``),
            was cancelled or was interrupted carries none.
        error_code: The failure's code, the user's warning's ``error_code``:
            the provider error's type for a failed model call, the reply's
            code for one that is no answer; None for none.
    """

    text: str = ""
    outcome: str | None = None
    error: str | None = None
    state: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    plan: str = ""
    tasks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Delegated:
    """What the backend's task tool gets back for a ``task`` call.

    Attributes:
        result: The envelope, or the error result of a call that started no run.
        state: What the run changed of the shared state; empty when none started.
        run: The run; for a steer, the run the message was given to; None when
            no run started.
    """

    result: dict[str, Any]
    state: Mapping[str, Any]
    run: RunRef | None


RunHook = Callable[[AgentRun], Awaitable[RunAnswer]]
#: How many instances of each type a conversation's store already keeps, by
#: type: the highest instance number (``BaseWorkflowManager._stored_instances``).
StoredInstances = Callable[[SessionRef], Awaitable[Mapping[str, int]]]


@dataclass
class _Working:
    """A run at work, as its own task sees it: its slot in its conversation's
    pool, the child runs it waits on (by run id), its waits on the user, and
    the working status it said last (``running`` from its start)."""

    ref: RunRef
    call_id: str | None
    pool: SlotPool
    slot: Slot
    emit: Callable[[WorkflowEvent], None]
    children: set[str] = field(default_factory=set)
    asking: int = 0
    said: str = RUN_RUNNING

    def waits_on_only(self, child: str) -> bool:
        """Whether ``child`` is all the run waits on: its last child, the user not asked."""
        return self.asking == 0 and self.children <= {child}

    def say(self, status: str) -> None:
        """Report ``status``, a working one, unless the run said it last."""
        if status != self.said:
            self.said = status
            self.emit(WorkflowEvent.agent_run(self.ref, status, call_id=self.call_id))

    @asynccontextmanager
    async def waiting_on_user(self) -> AsyncIterator[None]:
        """``input_required``, its slot lent for the block, ``queued`` while a take-back
        waits for a slot, then ``running``; nested waits once.

        A wait that starts while an earlier one waits for the slot back
        (two calls of one response that each ask) says ``input_required``
        again: the run waits on the user.
        """
        self.asking += 1
        self.say(RUN_INPUT_REQUIRED)
        stopped = False
        try:
            async with self.pool.lent(self.slot, on_wait=lambda: self.say(RUN_QUEUED)):
                yield
        except asyncio.CancelledError:
            stopped = True
            raise
        finally:
            self.asking -= 1
            if self.asking == 0 and not stopped:
                self.say(RUN_RUNNING)


#: The run at work in this task (``_in_run`` sets it in the run's own task,
#: whose copies its tool calls run in); None in the root's own work.
_working: ContextVar[_Working | None] = ContextVar("agentic_cli_working_run", default=None)


@asynccontextmanager
async def waiting_on_user() -> AsyncIterator[None]:
    """The block waits on the user (a question, a permission prompt): the run lends its slot.

    In a sub-agent run's own work, the run says ``input_required``, its slot
    works for another run meanwhile, and once the block ends it takes the
    slot back, ahead of the runs that have not started, saying ``queued``
    while it waits for it (another run holds it), and says ``running``;
    nested waits (a prompt waiting its turn behind another's, then the
    prompt) count once. A wait a cancel ends takes nothing back: the run is
    being stopped. Nothing in the root's own work, nor outside a run.

    Only a wait that can block belongs in it: a lent slot goes to the first
    run queued for one, which may work its whole run in it before this run
    takes it back. ``request_user_input`` and the permission engine enter it
    only when someone can answer (``can_request_user_input``).
    """
    working = _working.get()
    if working is None:
        yield
        return
    async with working.waiting_on_user():
        yield


class RunFailed(Exception):
    """A run's failure in its backend's words: what its parent and its user read.

    A hook raises it for a failure it has words for (a sub-agent's context
    overflow, ``context.subagent_overflow_message``): its text is the
    envelope's ``error`` and the user's warning, as given, ``code`` that
    warning's error code, and ``text`` the last text the run wrote before
    it failed (``""`` for none), the envelope's ``result``. A failure in
    the provider's words (a failed model call) a hook answers instead
    (``RunAnswer.error``, ``failure_reason``, with the run's last text and
    the provider error's type as ``RunAnswer.error_code``).
    Any other exception a hook lets out fails the run with its text (its
    type's name, for none), warned of as ``<instance> failed: <reason>``
    with the type's name as its code, and no result.
    """

    def __init__(self, error: str, *, code: str, text: str = "") -> None:
        super().__init__(error)
        self.code = code
        self.text = text


def failure_reason(error: BaseException) -> str:
    """A failure's reason, as a failed run's envelope gives it: its text, else its type's name."""
    return str(error) or type(error).__name__


class Delegations:
    """A manager's delegations: the one place a sub-agent run starts, ends and is decided.

    Args:
        runs: The manager's runs, which name and record each run.
        policy: How the manager's sub-agents run.
        types: Every agent type, by name: a caller may start its own
            sub-agents (``AgentConfig.sub_agents``) and, as the policy
            allows, the types agents may start at run time (``offered``).
        hook: Runs one run (``BaseWorkflowManager._run_agent``).
        ledgers: The manager's usage ledgers: a run's end carries what it
            used, by model, and its token limit counts its side calls
            (``RunControl``'s ``side_tokens``). None: an end carries no
            usage, and a run's own calls alone count toward its limit.
        open_calls: The manager's record of its calls (``OpenCalls``): the
            stop recorded for an instance that is left closed is taken, so
            its calls are freed (it can never run again to answer them).
            None: no record.
        stored: How many instances of each type a conversation's store
            already keeps (``BaseWorkflowManager._stored_instances``), asked
            once per conversation per turn, before the turn's first run is
            named, so what another manager ran there (before a restart, or
            in another process between two turns) is numbered past. None:
            every conversation is numbered from 1.
    """

    def __init__(
        self,
        *,
        runs: AgentRuns,
        policy: SubAgentPolicy,
        types: Mapping[str, AgentConfig],
        hook: RunHook,
        ledgers: UsageLedgers | None = None,
        stored: StoredInstances | None = None,
        open_calls: OpenCalls | None = None,
    ) -> None:
        self._runs = runs
        self.policy = policy
        self._types = dict(types)
        self._hook = hook
        self._ledgers = ledgers
        self._stored = stored
        self._open_calls = open_calls
        # Each conversation's turn its stored instances were last read in
        # (its span, ``current_turn_scope``: held here, so no later turn's
        # can be the same object), and each conversation's lock, which a
        # turn's parallel first runs wait at while one reads them.
        self._numbered: dict[SessionRef, object] = {}
        self._numbering: dict[SessionRef, asyncio.Lock] = {}
        # The working runs' controls, by (conversation, run id).
        self._controls: dict[tuple[SessionRef | None, str], RunControl] = {}
        # Each working run's end (set once it has ended), the messages steered
        # to an instance's run that its transcript did not hold (by conversation
        # and instance), and the instances to close at their runs' ends, with why.
        self._ends: dict[tuple[SessionRef | None, str], asyncio.Event] = {}
        self._unread: dict[tuple[SessionRef | None, str], list[str]] = {}
        self._closing: dict[tuple[SessionRef | None, str], str] = {}
        # Each conversation's working slots, made on first use and kept: a
        # slot lent while its run waits must come back to the pool it left.
        self._pools: dict[SessionRef | None, SlotPool] = {}

    def slots(self, conversation: SessionRef | None) -> SlotPool:
        """``conversation``'s working slots: ``policy.max_active`` of them, made on first use."""
        pool = self._pools.get(conversation)
        if pool is None:
            pool = self._pools[conversation] = SlotPool(self.policy.max_active)
        return pool

    def sub_agents(self, agent: str) -> dict[str, AgentConfig]:
        """The types ``agent`` may start, by name, in its ``sub_agents``' order."""
        caller = self._types.get(agent)
        names = caller.sub_agents if caller is not None else ()
        return {name: self._types[name] for name in names if name in self._types}

    def offered(self, caller_agent: str, depth: int) -> dict[str, AgentConfig]:
        """The types a run of ``caller_agent`` at ``depth`` may start with ``task``.

        Its own sub-agents in their order, then, while ``depth`` is below
        ``max_spawn_depth`` and the caller does not inherit its parent's
        conversation, the types agents may start at run time
        (``spawnable``), those not already its own, in config order. An
        inheriting type is offered to the root (depth 0) alone: only the
        root's own sub-agents inherit.
        """
        offered = self.sub_agents(caller_agent)
        caller = self._types.get(caller_agent)
        if caller is None or caller.inherit_context or not may_spawn(depth, self.policy):
            return offered
        for name in spawnable(list(self._types.values()), self.policy, AGENT_SOURCE):
            config = self._types[name]
            if name not in offered and (depth == 0 or not config.inherit_context):
                offered[name] = config
        return offered

    def task_offer(
        self, caller_agent: str, depth: int, *, warn: bool = False
    ) -> TaskOffer | None:
        """What ``task`` offers a run of ``caller_agent`` at ``depth``; None when it may start nothing.

        Both backends declare ``task`` from this: the schema and description
        of ``offered``'s types, with the sentence that sub-agents cannot
        start sub-agents when what the run starts could not.
        """
        types = self.offered(caller_agent, depth)
        if not types:
            return None
        own = self._types[caller_agent].sub_agents
        return task_offer(
            types, self.policy,
            depth_sentence=depth_sentence_applies(types, own, depth, self.policy),
            warn=warn,
        )

    async def delegate(self, request: DelegationRequest) -> Delegated:
        """Run the sub-agent ``request`` asks for, to its end; its envelope.

        A brief that cannot start a run is the error result, and nothing
        else happens; otherwise it ends in ``start``, started by the
        calling run.
        """
        depth = request.caller.depth if request.caller is not None else 0
        try:
            config, brief = read_brief(
                request.arguments, self.offered(request.caller_agent, depth)
            )
        except BriefError as error:
            return _refused(str(error))
        return await self.start(StartRequest(
            agent_type=config.name, brief=brief, started_by=started_by(request.caller),
            caller=request.caller, call_id=request.call_id, state=request.state,
            history=request.history,
        ))

    async def start(self, request: StartRequest) -> Delegated:
        """Run a first run of ``request.agent_type`` to its end; its envelope.

        A call past the runs its turn may start (when ``request.counted``)
        is the error result, and nothing else happens. A caller that is a
        run lends its slot while it waits for this one. A cancel of the
        caller's task (the turn's) ends the run ``interrupted`` once it has
        stopped, and goes on.

        Raises:
            KeyError: ``request.agent_type`` is not one of the manager's types.
        """
        config = self._types[request.agent_type]
        conversation = (
            request.conversation if request.conversation is not None else get_active_turn()
        )
        with _as_turn(conversation):
            if request.counted:
                refused = self._past_runs_per_turn(config.name)
                if refused is not None:
                    return refused
                count_run_this_turn()
            await self._number_past_stored(conversation)
            ref = self._runs.run_for(conversation, request.caller, request.call_id, config.name)
            brief = request.brief
            self._emit(WorkflowEvent.agent_run(
                ref, RUN_STARTED, call_id=request.call_id, title=run_title(brief.title),
                brief=brief.fields(), started_by=request.started_by, mode=BLOCKING,
                inherit_context=config.inherit_context, continuable=config.continuable,
            ))
            return await self._run(
                ref, config, conversation, request.call_id,
                message=task_message(brief, inherit=config.inherit_context),
                state=request.state,
                history=request.history if config.inherit_context else None,
            )

    async def follow_up(
        self, request: FollowUpRequest, *, _reserved: list[bool] | None = None
    ) -> Delegated:
        """Run the instance ``request`` names once more, to its end; its envelope.

        The instance's next run (``AgentRuns.next_run``), through the same
        hook, slots, limits, events and envelope as a first run, its
        ``AGENT_RUN`` start carrying the message as its ``input``. A
        follow-up that cannot run is the error result and nothing else
        happens: an unknown instance (an agent may continue only its own
        sub-agents, the user any), a closed one, one the user stopped (only
        the user continues it), a running one (one run per instance at a
        time) and a blank message. An agent's follow-up counts toward its
        turn's runs (``SubAgentPolicy.max_runs_per_turn``), the user's does
        not. Everything up to the run's start event is one synchronous step,
        so two follow-ups of one instance cannot both start.
        """
        conversation = (
            request.conversation if request.conversation is not None else get_active_turn()
        )
        with _as_turn(conversation):
            error = self._follow_up_refused(conversation, request)
            if error is not None:
                return Delegated(result={"success": False, "error": error}, state={}, run=None)
            instance = self._runs.instance(conversation, request.instance)
            assert instance is not None
            config = self._types[instance.agent]
            if request.started_by != USER:
                if _reserved is None:
                    refused = self._past_runs_per_turn(instance.agent)
                    if refused is not None:
                        return refused
                    count_run_this_turn()
                else:
                    _reserved[0] = False  # the reserved run starts now
            caller = request.caller
            ref = self._runs.next_run(
                conversation, request.instance,
                parent_run=None if caller is None or caller.is_root else caller.run,
            )
            self._emit(WorkflowEvent.agent_run(
                ref, RUN_STARTED, call_id=request.call_id, title=run_title(request.message),
                started_by=request.started_by, mode=BLOCKING, input=request.message,
            ))
            return await self._run(
                ref, config, conversation, request.call_id,
                message=request.message, state=request.state, history=None, follow_up=True,
            )

    def close(self, conversation: SessionRef | None, instance: str, *, reason: str) -> list[str]:
        """Close the idle or stopped ``instance`` of ``conversation`` and its descendants.

        Each closed instance is reported (``AGENT_INSTANCE``, ``closed``,
        with ``reason``). A running, closed or unknown instance is left as it
        is.

        Returns:
            The ids closed, the instance first, then its descendants in the
            order they started; empty when it could not be closed. A
            descendant that works is not among them: it is closed as its run
            ends.
        """
        with _as_turn(conversation):
            record = self._runs.instance(conversation, instance)
            if record is None or record.status not in _OPEN:
                return []
            below = self._descendants(conversation, instance)
            closed: list[str] = []
            for one in [record, *below]:
                if one.status in _OPEN:
                    self._close(conversation, one, reason)
                    closed.append(one.instance)
            # A descendant that works is closed as its run ends.
            self._close_at_end(conversation, below, reason)
            return closed

    async def message(
        self,
        request: FollowUpRequest,
        *,
        interrupt: bool = False,
        _reserved: list[bool] | None = None,
    ) -> Delegated:
        """Send a message to a sub-agent instance: ``message_agent``'s.

        An idle instance runs it as a follow-up (``follow_up``). A running
        one of the sender's (the user's: any) is steered: the message
        reaches its run before its next step (``Delegated.result`` says so,
        with the run), or, with ``interrupt``, its run is cancelled by its
        parent and the message starts a follow-up once it has ended. One run
        per instance at a time: everything up to a follow-up's start is one
        synchronous step, so of two messages to one idle instance in one
        response the first starts its run and the second is steered into it.
        A run that takes no more steps (cancelled, its work ended, or an
        instance its parent is closing) is not steered: its end is waited for,
        and the message then follows up the idle instance or is refused as
        ``follow_up`` refuses (closed, stopped by the user); to an instance
        being closed it is refused as closed at once, since the answer is
        known. An interrupt of an agent's keeps the turn's run it checked
        for: the run is counted as the check is made, so no call in the same
        answer can take it before the follow-up starts (it is given back when
        no run starts from it, whatever stops the call). A blank message,
        and anything ``follow_up`` refuses (``interrupt`` included: nothing is
        cancelled for a message that cannot start a run), is the error result
        and nothing else happens. The sender's interrupt is its own stop: the
        user's reads ``cancelled_by`` ``user``, a parent's ``parent``.

        Returns:
            The follow-up's result, or the steer's; ``state`` is the shared
            state a follow-up run changed.
        """
        conversation = (
            request.conversation if request.conversation is not None else get_active_turn()
        )
        with _as_turn(conversation):
            if not isinstance(request.message, str) or not request.message.strip():
                return _refused(MESSAGE_NEEDED_ERROR)
            control = self._working_control(conversation, request)
            if control is None:
                return await self.follow_up(request, _reserved=_reserved)
            ref = control.ref
            if (conversation, ref.instance) in self._closing:
                return _refused(CLOSED_ERROR.format(instance=ref.instance))
            stopping = control.stopping
            if not interrupt and not stopping:
                control.steer(request.message)
                logger.info("subagent_steered", run=ref.run, started_by=request.started_by)
                return Delegated(
                    result={"success": True, "agent": ref.instance, "run": ref.run,
                            "status": RUN_RUNNING, "note": STEERED_NOTE},
                    state={}, run=ref,
                )
            ended = self._ends[(conversation, ref.run or "")]
            made_here = False
            if interrupt and not stopping:
                error = self._follow_up_refused(conversation, request, running_ok=True)
                if error is None and request.started_by != USER:
                    refused = self._past_runs_per_turn(ref.agent)
                    error = refused.result["error"] if refused is not None else None
                if error is not None:
                    return _refused(error)
                if request.started_by != USER and _reserved is None:
                    count_run_this_turn()
                    _reserved, made_here = [True], True
                by, why = (USER, REPLACED_BY_USER_ERROR) if request.started_by == USER else (
                    PARENT, REPLACED_ERROR
                )
                self.cancel(ref.run or "", by=by, error=why, conversation=conversation)
        try:
            await ended.wait()
            # Idle now: its follow-up (or, should another run have started
            # meanwhile, a steer of that one).
            return await self.message(request, _reserved=_reserved)
        finally:
            if made_here and _reserved is not None and _reserved[0]:
                release_run_this_turn()  # no run started from it: give its place back

    def stop(
        self,
        conversation: SessionRef | None,
        caller: RunRef | None,
        instance: str,
        *,
        close: bool = False,
    ) -> dict[str, Any]:
        """Stop one of ``caller``'s sub-agents: ``stop_agent``'s result.

        A running instance's run is cancelled at once, by its parent
        (``cancel``); the instance is idle again once the run ends. With
        ``close`` the instance and its descendants are closed: an idle one at
        once, a running one (and any descendant that works) as its run ends
        (``closed`` names them all). An idle instance without ``close`` is
        left as it is. The instance's status is the one it is left in once
        its run has ended; ``stopped`` is the run cancelled, None when none
        worked or its work had ended.

        Returns:
            The result, or the error result for an instance that is not the
            caller's, one that is closed, or one the user stopped (only the
            user continues it).
        """
        conversation = conversation if conversation is not None else get_active_turn()
        with _as_turn(conversation):
            yours = self._yours(conversation, caller)
            record = next((i for i in yours if i.instance == instance), None)
            error = self._refused_to(instance, yours, record)
            if error is not None:
                return {"success": False, "error": error}
            assert record is not None
            below = self._descendants(conversation, instance)
            reaching = [record.instance, *(i.instance for i in below if i.status != INSTANCE_CLOSED)]
            control = self._working_control(conversation, record=record)
            stopped: str | None = None
            closed: list[str] = []
            if control is not None:
                run = control.ref.run or ""
                if self.cancel(run, by=PARENT, error=STOPPED_BY_PARENT_ERROR,
                               conversation=conversation):
                    stopped = run
                if close:
                    self._close_at_end(conversation, [record, *below], CLOSED_BY_PARENT)
                    closed = reaching
                left = INSTANCE_CLOSED if close or not record.continuable else INSTANCE_IDLE
            elif close:
                self.close(conversation, instance, reason=CLOSED_BY_PARENT)
                closed = reaching
                left = INSTANCE_CLOSED
            else:
                left = record.status
            logger.info("subagent_stopped", instance=instance, run=stopped, closed=closed)
            return {"success": True, "agent": instance, "status": left, "stopped": stopped,
                    "closed": closed}

    def check(
        self, conversation: SessionRef | None, caller: RunRef | None, instance: str | None = None
    ) -> dict[str, Any]:
        """``caller``'s sub-agents, or one of them: ``check_agents``' result.

        Each row is an instance's status, title, last run and its status,
        the first lines of that run's result (none while it works) and the
        input and output tokens of all its runs: an ended run's from its
        record, a working one's from the conversation's ledger. Newest
        first; closed ones too. A name that is not the caller's sub-agent is
        the error result.
        """
        conversation = conversation if conversation is not None else get_active_turn()
        with _as_turn(conversation):
            yours = self._yours(conversation, caller)
            if instance is not None:
                found = next((i for i in yours if i.instance == instance), None)
                if found is None:
                    return {"success": False, "error": _not_yours(instance, yours)}
                yours = [found]
            return {
                "success": True,
                "agents": [self._checked(conversation, one) for one in reversed(yours)],
            }

    def steer(self, conversation: SessionRef | None, run_id: str, message: str) -> bool:
        """Give the working run ``run_id`` of ``conversation`` (the active turn's, for None)
        ``message``, before its next model call.

        Returns:
            False when no such run works, or it takes no more steps
            (cancelled, or its work has ended).

        Raises:
            ValueError: ``message`` is not text, or blank.
        """
        key = (conversation if conversation is not None else get_active_turn(), run_id)
        control = self._controls.get(key)
        if control is None or control.stopping:
            return False
        control.steer(message)
        return True

    async def stop_by_user(self, conversation: SessionRef | None, instance: str) -> bool:
        """The user's stop of ``conversation``'s ``instance``, any of its instances.

        A running instance's run is cancelled by the user
        (``STOPPED_BY_USER_ERROR``) and waited for to its end, which leaves
        the instance ``stopped`` (``closed`` for a type that is not
        ``continuable``); a run whose work had ended as the stop came
        (``RunControl.finish``) ends as it went, and the instance is stopped
        after its end. An idle instance is stopped at once (``AGENT_INSTANCE``
        ``stopped``, reason ``STOPPED_BY_USER_ERROR``). Only the user may
        start a stopped instance's next run. ``conversation``, when not the
        active turn's, is made the active turn for the events; None is the
        active turn's.

        Returns:
            True when it stopped the instance; False, doing nothing, for an
            instance that is unknown, closed or stopped already.
        """
        conversation = conversation if conversation is not None else get_active_turn()
        with _as_turn(conversation):
            record = self._runs.instance(conversation, instance)
            if record is None or record.status in (INSTANCE_CLOSED, INSTANCE_STOPPED):
                return False
            stopped = await self._user_stops_run(conversation, record)
            logger.info("subagent_stopped", instance=instance, run=stopped, by=USER)
            record = self._runs.instance(conversation, instance) or record
            if record.status not in (INSTANCE_STOPPED, INSTANCE_CLOSED):
                # Idle, or its run ended as it went: stopped without a run.
                self._emit_instance(conversation, record, INSTANCE_STOPPED, STOPPED_BY_USER_ERROR)
            return True

    async def close_by_user(self, conversation: SessionRef | None, instance: str) -> list[str]:
        """The user's close of ``conversation``'s ``instance`` and its descendants.

        An idle or stopped one is closed at once (``close``, reason
        ``CLOSED_BY_USER``); a running one's run is cancelled by the user and
        waited for, its end closing it, then its open descendants. Its
        transcript stays readable; it takes no follow-up. ``conversation``
        as ``stop_by_user`` takes it.

        Returns:
            The ids closed, ``instance`` first, then its descendants in the
            order they started; none for an instance that is unknown or closed.
        """
        conversation = conversation if conversation is not None else get_active_turn()
        with _as_turn(conversation):
            record = self._runs.instance(conversation, instance)
            if record is None or record.status == INSTANCE_CLOSED:
                return []
            if record.status != INSTANCE_RUNNING:
                return self.close(conversation, instance, reason=CLOSED_BY_USER)
            closing = [
                one.instance for one in (record, *self._descendants(conversation, instance))
                if one.status != INSTANCE_CLOSED
            ]
            self._closing[(conversation, instance)] = CLOSED_BY_USER
            if await self._user_stops_run(conversation, record) is None:
                # No run of this manager's works for it: closed as an open one is.
                self._closing.pop((conversation, instance), None)
                for one in (record, *self._descendants(conversation, instance)):
                    if one.instance == instance or one.status in _OPEN:
                        self._close(conversation, one, CLOSED_BY_USER)
            return [
                one for one in closing
                if (found := self._runs.instance(conversation, one)) is not None
                and found.status == INSTANCE_CLOSED
            ]

    async def _user_stops_run(
        self, conversation: SessionRef | None, record: InstanceRecord
    ) -> str | None:
        """Cancel ``record``'s working run by the user and wait for its end; the run's id.

        None when no run of this manager works for it.
        """
        control = self._working_control(conversation, record=record)
        if control is None:
            return None
        run = control.ref.run or ""
        ended = self._ends[(conversation, run)]
        self.cancel(run, by=USER, error=STOPPED_BY_USER_ERROR, conversation=conversation)
        await ended.wait()
        return run

    def _emit_instance(
        self, conversation: SessionRef | None, instance: InstanceRecord, status: str, reason: str
    ) -> None:
        """Report and record ``instance``'s change to ``status`` (``AGENT_INSTANCE``),
        named by its last run."""
        last = self._runs.record(conversation, instance.instance)
        if last is not None:
            self._emit(WorkflowEvent.agent_instance(last.ref, status, reason=reason))

    def stop_cause(self, run: RunRef | None, *, conversation: SessionRef | None = None) -> str | None:
        """What stopped ``run``, for a call it left at its prompt or running
        (``interrupts.not_run``, ``interrupts.interrupted_by``).

        Its own control's cancel (``RunControl.stop_cause``: its parent, its
        time limit, its token limit), else ``its parent`` when a run above
        it was cancelled by its control; None for the user's stop, a run
        that was not stopped, the root and a run that does not work.
        """
        if run is None or run.run is None:
            return None
        conversation = conversation if conversation is not None else get_active_turn()
        control = self._controls.get((conversation, run.run))
        if control is not None and control.cancelled is not None:
            return control.stop_cause
        return PARENT_CAUSE if self._parent_stopped(conversation, run) else None

    def forget(self, conversation: SessionRef) -> None:
        """Forget ``conversation`` (it was deleted): its working slots, the messages
        kept unread and the closes waiting for its runs' ends, and that its
        store was read, so a conversation of its id asks its store again."""
        self._numbered.pop(conversation, None)
        self._numbering.pop(conversation, None)
        self._pools.pop(conversation, None)
        for held in (self._unread, self._closing):
            for key in [k for k in held if k[0] == conversation]:
                del held[key]

    def _yours(self, conversation: SessionRef | None, caller: RunRef | None) -> list[InstanceRecord]:
        """The instances ``caller`` started (the root's, for None), in start order."""
        sender = caller.instance if caller is not None else "root"
        return [i for i in self._runs.instances(conversation) if i.parent == sender]

    def _working_control(
        self,
        conversation: SessionRef | None,
        request: FollowUpRequest | None = None,
        *,
        record: InstanceRecord | None = None,
    ) -> RunControl | None:
        """The control of the run that works for an instance, else None.

        The instance is ``record``, or the one ``request`` names when it is
        the sender's (the user's: any); None for an instance that is not
        running.
        """
        if record is None:
            assert request is not None
            record = self._runs.instance(conversation, request.instance)
            if record is None:
                return None
            if request.started_by != USER and not any(
                i.instance == record.instance for i in self._yours(conversation, request.caller)
            ):
                return None
        if record.status != INSTANCE_RUNNING or not record.runs:
            return None
        return self._controls.get((conversation, record.runs[-1]))

    @staticmethod
    def _refused_to(
        instance: str, yours: Sequence[InstanceRecord], record: InstanceRecord | None
    ) -> str | None:
        """Why a parent may not stop ``instance``, else None."""
        if record is None:
            return _not_yours(instance, yours)
        if record.status == INSTANCE_CLOSED:
            return CLOSED_ERROR.format(instance=instance)
        if record.status == INSTANCE_STOPPED:
            return STOPPED_ERROR.format(instance=instance)
        return None

    def _checked(self, conversation: SessionRef | None, record: InstanceRecord) -> dict[str, Any]:
        """One row of ``check``."""
        runs = [r for r in self._runs.records(conversation) if r.ref.instance == record.instance]
        last = runs[-1] if runs else None
        spent = {"input": 0, "output": 0}
        for run in runs:
            usage = run.usage
            if run.status not in RUN_END_STATUSES and self._ledgers is not None:
                usage = self._ledgers.ledger(conversation).run_usage(run.ref.run or "")
            for totals in usage.values():
                spent["input"] += totals.get("input", 0)
                spent["output"] += totals.get("output", 0)
        return {
            "agent": record.instance,
            "type": record.agent,
            "status": record.status,
            "title": record.title,
            "run": last.ref.run if last is not None else None,
            "run_status": last.status if last is not None else None,
            "result": (
                "\n".join(last.result.splitlines()[:_CHECK_LINES]) if last is not None else ""
            ),
            "tokens": spent,
        }

    def _parent_stopped(self, conversation: SessionRef | None, ref: RunRef) -> bool:
        """Whether a run above ``ref`` was cancelled by its control: its parent's stop, a limit, the user's."""
        parent = ref.parent_run
        while parent:
            control = self._controls.get((conversation, parent))
            if control is None:
                return False
            if control.cancelled is not None:
                return True
            parent = control.ref.parent_run
        return False

    def _close_at_end(
        self, conversation: SessionRef | None, records: Sequence[InstanceRecord], reason: str
    ) -> None:
        """Close each of ``records`` that works as its run ends (``_end``)."""
        for record in records:
            if record.status == INSTANCE_RUNNING:
                self._closing[(conversation, record.instance)] = reason

    def _drop_unread(self, conversation: SessionRef | None, instance: str) -> None:
        """Forget the messages kept for the next run of ``instance``, which will not run."""
        dropped = self._unread.pop((conversation, instance), None)
        if dropped:
            logger.info("subagent_steering_dropped", instance=instance, messages=len(dropped))

    def _keep_unread(
        self, conversation: SessionRef | None, instance: str, unread: Sequence[str]
    ) -> None:
        """Keep the messages a run's transcript did not hold for the instance's next run."""
        if not unread:
            return
        record = self._runs.instance(conversation, instance)
        self._unread.setdefault((conversation, instance), []).extend(unread)
        if record is not None and record.status == INSTANCE_CLOSED:
            self._drop_unread(conversation, instance)

    def _past_runs_per_turn(self, agent: str) -> Delegated | None:
        """The error result of a call past the runs its turn may start, else None."""
        started = runs_this_turn()
        limit = self.policy.max_runs_per_turn
        if started is not None and started >= limit:
            logger.info("subagent_runs_per_turn_reached", agent=agent, limit=limit)
            return Delegated(
                result={"success": False, "error": RUNS_PER_TURN_ERROR.format(limit=limit)},
                state={}, run=None,
            )
        return None

    def _follow_up_refused(
        self,
        conversation: SessionRef | None,
        request: FollowUpRequest,
        *,
        running_ok: bool = False,
    ) -> str | None:
        """Why ``request`` cannot start a run, else None.

        ``running_ok``: a running instance is no reason (an interrupt asks
        whether the message could start a run once the instance's run ends).
        """
        message = request.message
        if not isinstance(message, str) or not message.strip():
            return NO_MESSAGE_ERROR
        if request.started_by == USER:
            yours = self._runs.instances(conversation)
        else:
            sender = request.caller.instance if request.caller is not None else "root"
            yours = [i for i in self._runs.instances(conversation) if i.parent == sender]
        found = next((i for i in yours if i.instance == request.instance), None)
        if found is None:
            return _not_yours(request.instance, yours)
        if found.status == INSTANCE_CLOSED or not found.continuable:
            return CLOSED_ERROR.format(instance=found.instance)
        if found.status == INSTANCE_STOPPED and request.started_by != USER:
            return STOPPED_ERROR.format(instance=found.instance)
        if found.status == INSTANCE_RUNNING and not running_ok:
            return RUNNING_ERROR.format(instance=found.instance)
        if found.agent not in self._types:
            return TYPE_GONE_ERROR.format(instance=found.instance, agent=found.agent)
        return None

    async def _run(
        self,
        ref: RunRef,
        config: AgentConfig,
        conversation: SessionRef | None,
        call_id: str | None,
        *,
        message: str,
        state: Mapping[str, Any],
        history: Callable[[], Any] | None,
        follow_up: bool = False,
    ) -> Delegated:
        """Run ``ref``'s run, started and reported already, to its end; its envelope.

        What a first run and a follow-up share: the run's control, its task,
        the caller's slot lent while it waits.
        """
        control = RunControl(
            ref, self._limits(config), side_tokens=self._side_tokens(conversation, ref)
        )
        # What an earlier run was steered and did not store: this run's, after its own message.
        for unread in self._unread.pop((conversation, ref.instance), ()):
            control.steer(unread)
        inner = asyncio.create_task(
            self._in_run(
                ref, config, conversation, call_id, control,
                message=message, state=state, history=history, follow_up=follow_up,
            ),
            name=f"agent run {ref.run}",
        )
        control.bind(inner)
        key = (conversation, ref.run or "")
        self._controls[key] = control
        ended = self._ends[key] = asyncio.Event()
        caller = _working.get()

        async def ended_run() -> Delegated:
            # As the run ends, before its caller waits for its slot back: a
            # follow-up then starts with the messages this run did not store.
            try:
                return await self._ended(ref, call_id, inner, control)
            finally:
                self._controls.pop(key, None)
                self._ends.pop(key, None)
                self._keep_unread(conversation, ref.instance, control.unstored())

        try:
            if caller is None:
                return await ended_run()
            # The caller waits: its slot works for another run meanwhile.
            caller.children.add(ref.run or "")
            try:
                async with caller.pool.lent(caller.slot):
                    return await ended_run()
            finally:
                caller.children.discard(ref.run or "")
        finally:
            ended.set()

    def control(self, run: RunRef | None) -> RunControl | None:
        """The control of ``run`` while it works, in the active turn's conversation; else None.

        None for the root's own work (no run), and for a run that has ended.
        """
        if run is None or run.run is None:
            return None
        return self._controls.get((get_active_turn(), run.run))

    def cancel(
        self, run_id: str, *, by: str, error: str, conversation: SessionRef | None = None
    ) -> bool:
        """Cancel the working run ``run_id`` of ``conversation`` (the active turn's, for None), alone.

        ``by`` its parent, the user or a limit; ``error`` says why, as its
        envelope will. Its siblings and its turn go on; its parent reads its
        envelope (``cancelled``, with what it had of its work).

        Returns:
            True when this cancelled it; False for no such working run, one
            cancelled already, or one whose work has ended
            (``RunControl.finish``).

        Raises:
            ValueError: ``by`` is not ``parent``, ``user`` or ``limit``.
        """
        if by not in CANCELLED_BY:
            raise ValueError(f"A run is cancelled by its parent, user or limit, not {by!r}.")
        key = (conversation if conversation is not None else get_active_turn(), run_id)
        control = self._controls.get(key)
        if control is None or control.cancelled is not None:
            return False
        control.cancel(by=by, error=error)
        return control.cancelled is not None

    def _limits(self, config: AgentConfig) -> AgentLimits:
        """A run of ``config``'s limits: the type's own over the policy's."""
        if config.limits is None:
            return self.policy.limits
        return config.limits.over(self.policy.limits)

    async def _ended(
        self, ref: RunRef, call_id: str | None, inner: asyncio.Task[RunAnswer], control: RunControl
    ) -> Delegated:
        """Wait for the run's task ``inner``, then end the run as it went; its envelope."""
        try:
            answer = await inner
        except asyncio.CancelledError:
            if _caller_cancelling():
                self._stopped_with_caller(ref, call_id, RunAnswer())
                raise
            if control.cancelled is not None:
                # Its control cancelled it and the hook let the cancel out:
                # nothing of its work came back.
                return self._cancelled(ref, call_id, control.cancelled, RunAnswer())
            self._end(ref, call_id, RUN_INTERRUPTED, error=INTERRUPTED_ERROR)
            # The run's own task was stopped, not by its control nor its
            # caller's: the caller reads that it ended.
            return Delegated(
                result=envelope(ref, RUN_INTERRUPTED, error=INTERRUPTED_ERROR), state={}, run=ref
            )
        except RunFailed as error:
            return self._failed(
                ref, call_id, str(error), warning=str(error), code=error.code, text=error.text
            )
        except Exception as error:  # noqa: BLE001 - a sub-agent's failure is its parent's to read
            reason = failure_reason(error)
            logger.debug("subagent_failed", run=ref.run, exc_info=error)
            return self._failed(
                ref, call_id, reason,
                warning=f"{ref.instance} failed: {reason}", code=type(error).__name__,
            )
        if _caller_cancelling():
            # The caller was stopped while the run's own cancel ended it (its
            # hook took that cancel and answered): the run is its caller's.
            self._stopped_with_caller(ref, call_id, answer)
            raise asyncio.CancelledError
        if control.cancelled is not None:
            return self._cancelled(ref, call_id, control.cancelled, answer)
        if answer.error is not None:
            return self._failed(
                ref, call_id, answer.error, warning=f"{ref.instance} failed: {answer.error}",
                code=answer.error_code, text=answer.text, state=answer.state,
                plan=answer.plan, tasks=answer.tasks,
            )
        if answer.outcome == REFUSED:
            self._end(
                ref, call_id, RUN_REFUSED, text=answer.text, error=REFUSAL_ERROR,
                plan=answer.plan, tasks=answer.tasks,
            )
            result = envelope(ref, RUN_REFUSED, result=answer.text, error=REFUSAL_ERROR)
        else:
            stepped = STEP_LIMIT_ERROR if control.at_step_limit else None
            partial = answer.outcome == CUT or stepped is not None
            end = self._end(
                ref, call_id, RUN_COMPLETED, text=answer.text, partial=partial, error=stepped,
                plan=answer.plan, tasks=answer.tasks,
            )
            if answer.outcome == CUT:
                warning = cut_off_warning(end)
                if warning is not None:
                    self._report(warning)
            result = envelope(
                ref, RUN_COMPLETED, result=answer.text, partial=partial, error=stepped,
                open_tasks=open_task_descriptions(answer.tasks) if partial else None,
            )
        return Delegated(result=result, state=dict(answer.state), run=ref)

    async def _number_past_stored(self, conversation: SessionRef | None) -> None:
        """Number ``conversation``'s instances past the ones its store keeps.

        Asked once per conversation per turn (``stored``), before the turn's
        first run is named; the turn's parallel first runs wait for that one
        ask. Each turn asks again: between two turns another process may
        have run the conversation (it holds it while its turn runs:
        ``ConversationLock``), and ``AgentRuns.number_from`` never lowers a
        count. A store that cannot be read is warned of in the log and asked
        again at the next run, and this one is numbered from what the
        manager knows (a stored instance it meets is its backend's to
        refuse); a cancel leaves it to be asked again too. With no turn's
        span open (``current_turn_scope``), every run asks.
        """
        if self._stored is None or conversation is None:
            return
        turn = current_turn_scope()
        if turn is not None and self._numbered.get(conversation) is turn:
            return
        async with self._numbering.setdefault(conversation, asyncio.Lock()):
            if turn is not None and self._numbered.get(conversation) is turn:
                return
            try:
                counts = await self._stored(conversation)
            except Exception as error:  # noqa: BLE001 - the run goes on: the next one asks again
                logger.warning("subagent_instances_unread", session_id=conversation.session_id,
                               error=truncate(str(error), _MAX_LOGGED_ERROR))
                logger.debug("subagent_instances_unread", exc_info=error)
                return
            self._runs.number_from(conversation, counts)
            if turn is not None:
                self._numbered[conversation] = turn

    async def _in_run(
        self,
        ref: RunRef,
        config: AgentConfig,
        conversation: SessionRef | None,
        call_id: str | None,
        control: RunControl,
        *,
        message: str,
        state: Mapping[str, Any],
        history: Callable[[], Any] | None,
        follow_up: bool,
    ) -> RunAnswer:
        """The run, in its own task: the active run, and no tool call active.

        The task's context is a copy of its caller's, so what this sets stays
        here. An inheriting type's history is read here, once. The hook runs
        in a working slot of the conversation's pool, the run ``queued``
        while it waits for one; the slot goes back at the run's end, to its
        caller when the caller waits on it alone (``SlotPool.hand_back``).
        The run's time limit runs from when its hook starts
        (``RunControl.timing``), not while it waits for its slot.
        """
        caller = _working.get()
        set_active_run(ref)
        set_active_tool_call(None)
        run = AgentRun(
            ref=ref,
            config=config,
            message=message,
            conversation=conversation,
            state=dict(state),
            history=history() if history is not None else None,
            control=control,
            follow_up=follow_up,
        )
        pool = self.slots(conversation)
        queued = pool.free == 0
        if queued:
            self._emit(WorkflowEvent.agent_run(ref, RUN_QUEUED, call_id=call_id))
        slot = await pool.acquire(NEW)
        try:
            if queued:
                self._emit(WorkflowEvent.agent_run(ref, RUN_RUNNING, call_id=call_id))
            _working.set(_Working(ref, call_id, pool, slot, emit=self._emit))
            with control.timing():
                return await self._hook(run)
        finally:
            if caller is None:
                pool.release(slot)
            else:
                last = caller.waits_on_only(ref.run or "")
                caller.children.discard(ref.run or "")
                if last:
                    # The caller's last wait: its slot goes straight back to it.
                    pool.hand_back(slot, to=caller.slot)
                else:
                    pool.release(slot)

    def _stopped_with_caller(self, ref: RunRef, call_id: str | None, answer: RunAnswer) -> None:
        """End ``ref``'s run, whose caller's task is being cancelled.

        A run above it cancelled by its control (its parent's stop, a limit,
        the user's stop of that run) is what stopped the caller: the run
        ends ``cancelled`` by its parent, saying so. Otherwise the turn is
        stopped (Ctrl+C), and the run ends ``interrupted``.
        """
        if self._parent_stopped(get_active_turn(), ref):
            self._cancelled(ref, call_id, (PARENT, WITH_ITS_PARENT_ERROR), answer)
        else:
            self._end(ref, call_id, RUN_INTERRUPTED, error=INTERRUPTED_ERROR)

    def _cancelled(
        self,
        ref: RunRef,
        call_id: str | None,
        cancelled: tuple[str, str],
        answer: RunAnswer,
    ) -> Delegated:
        """End ``ref``'s run cancelled (``cancelled``: by whom, why) with what ``answer`` had."""
        by, error = cancelled
        logger.info("subagent_cancelled", run=ref.run, agent=ref.agent, by=by, error=error)
        partial = bool(answer.text)
        self._end(
            ref, call_id, RUN_CANCELLED, text=answer.text, error=error, partial=partial,
            cancelled_by=by, plan=answer.plan, tasks=answer.tasks,
        )
        return Delegated(
            result=envelope(
                ref, RUN_CANCELLED, result=answer.text, partial=partial, error=error,
                cancelled_by=by, open_tasks=open_task_descriptions(answer.tasks),
            ),
            state=dict(answer.state), run=ref,
        )

    def _failed(
        self,
        ref: RunRef,
        call_id: str | None,
        error: str,
        *,
        warning: str,
        code: str | None,
        text: str = "",
        state: Mapping[str, Any] | None = None,
        plan: str = "",
        tasks: list[dict[str, Any]] | None = None,
    ) -> Delegated:
        """End ``ref``'s run failed with ``error``, warn the user, and its envelope."""
        logger.info("subagent_failed", run=ref.run, agent=ref.agent,
                    error=truncate(error, _MAX_LOGGED_ERROR))
        self._end(ref, call_id, RUN_FAILED, text=text, error=error, plan=plan, tasks=tasks)
        warned = WorkflowEvent.error(warning, error_code=code, recoverable=True)
        warned.metadata["agent"] = ref.agent
        warned.run = ref
        self._report(warned)
        return Delegated(
            result=envelope(
                ref, RUN_FAILED, result=text, error=error,
                open_tasks=open_task_descriptions(tasks or []),
            ),
            state=dict(state or {}), run=ref,
        )

    def _end(
        self,
        ref: RunRef,
        call_id: str | None,
        status: str,
        *,
        text: str = "",
        error: str | None = None,
        partial: bool = False,
        cancelled_by: str | None = None,
        plan: str = "",
        tasks: list[dict[str, Any]] | None = None,
    ) -> WorkflowEvent:
        """Report and record the ``AGENT_RUN`` that ends ``ref``'s run; that event.

        The end carries the run's last ``plan`` and ``tasks`` when it has any,
        and the status it leaves its instance in: ``closed`` for a type that
        is not continuable, an instance its parent closed while the run worked
        (``stop`` with ``close``: each open descendant is closed after this
        event) or one past ``max_open`` (the oldest idle instance is closed:
        another one after this event), ``stopped`` for the user's cancel,
        else ``idle``.
        """
        conversation = get_active_turn()
        instance = self._runs.instance(conversation, ref.instance)
        # A close its parent asked for while the run worked, applied now.
        closing_reason = self._closing.pop((conversation, ref.instance), None)
        if (instance is not None and not instance.continuable) or closing_reason is not None:
            left = INSTANCE_CLOSED
        elif status == RUN_CANCELLED and cancelled_by == USER:
            left = INSTANCE_STOPPED
        else:
            left = INSTANCE_IDLE
        closing = self._past_max_open(conversation, ref.instance, left)
        if ref.instance in closing:
            left = INSTANCE_CLOSED
        end = WorkflowEvent.agent_run(
            ref, status, call_id=call_id, result=result_head(text), error=error,
            partial=partial, usage=self._usage(ref), cancelled_by=cancelled_by,
            plan=plan or None, tasks=list(tasks) if tasks else None,
            instance_status=left,
        )
        self._emit(end)
        if left == INSTANCE_CLOSED:
            self._release_stop(conversation, ref.instance)
        if closing_reason is not None:
            for below in self._descendants(conversation, ref.instance):
                if below.status in _OPEN:
                    self._close(conversation, below, closing_reason)
        for other in closing:
            record = self._runs.instance(conversation, other)
            if other != ref.instance and record is not None and record.status in _OPEN:
                self._close(
                    conversation, record,
                    MAX_OPEN_REASON.format(limit=self.policy.max_open),
                )
        return end

    def _past_max_open(
        self, conversation: SessionRef | None, ending: str, left: str
    ) -> list[str]:
        """The instances to close for ``max_open``, when ``ending`` leaves in status ``left``.

        The open instances (``idle`` or ``stopped``) are counted, the ending
        one included when it is; past the limit, the oldest idle ones (by
        their last run's end, the ending one the newest) are the ones to
        close. A stopped instance counts and is never one of them.
        """
        instances = self._runs.instances(conversation)
        open_ = [i for i in instances if i.instance != ending and i.status in _OPEN]
        ending_open = left in _OPEN
        excess = len(open_) + (1 if ending_open else 0) - self.policy.max_open
        if excess <= 0:
            return []

        def ended(record: InstanceRecord) -> int:
            return self._runs.end_order(conversation, record.runs[-1]) if record.runs else 0

        idle = sorted(
            (i for i in open_ if i.status == INSTANCE_IDLE), key=ended
        )
        names = [i.instance for i in idle]
        if left == INSTANCE_IDLE:
            names.append(ending)
        return names[:excess]

    def _descendants(self, conversation: SessionRef | None, instance: str) -> list[InstanceRecord]:
        """The instances below ``instance`` (its children, theirs...), in the order they started."""
        below = {instance}
        found: list[InstanceRecord] = []
        for record in self._runs.instances(conversation):
            if record.parent in below:
                below.add(record.instance)
                found.append(record)
        return found

    def _release_stop(self, conversation: SessionRef | None, instance: str) -> None:
        """Take the stop recorded for ``instance``, closed: nothing will run to answer its calls."""
        if self._open_calls is not None and conversation is not None:
            self._open_calls.take_cancel(conversation, instance=instance)

    def _close(self, conversation: SessionRef | None, record: InstanceRecord, reason: str) -> None:
        """Report that ``record``'s instance is closed (``AGENT_INSTANCE``)."""
        last = self._runs.record(conversation, record.instance)
        if last is None:  # pragma: no cover - an instance has a run
            return
        logger.info("subagent_closed", instance=record.instance, reason=reason)
        self._release_stop(conversation, record.instance)
        self._drop_unread(conversation, record.instance)
        self._emit(WorkflowEvent.agent_instance(last.ref, INSTANCE_CLOSED, reason=reason))

    def _side_tokens(
        self, conversation: SessionRef | None, ref: RunRef
    ) -> Callable[[], int] | None:
        """What ``ref``'s run's side calls have used so far, as its conversation's ledger
        counts them: its tools' side calls and its summaries (``RunControl``'s
        ``side_tokens``; its own steps it counts itself). None without ledgers."""
        if self._ledgers is None:
            return None
        ledger = self._ledgers.ledger(conversation)

        def spent() -> int:
            side = ledger.by_purpose(run=ref.run or "")
            return sum(totals.input + totals.output for totals in side.values())

        return spent

    def _usage(self, ref: RunRef) -> dict[str, dict[str, int]] | None:
        """What ``ref``'s run used, by model, from its conversation's ledger."""
        if self._ledgers is None:
            return None
        return self._ledgers.ledger(get_active_turn()).run_usage(ref.run or "")

    def _emit(self, event: WorkflowEvent) -> None:
        """Record ``event`` (an ``AGENT_RUN``), then report it."""
        self._runs.observe(event)
        self._report(event)

    @staticmethod
    def _report(event: WorkflowEvent) -> None:
        """Report ``event`` to the turn's live events (none outside a turn)."""
        live_events.report(event)


@contextmanager
def _as_turn(conversation: SessionRef | None) -> Iterator[None]:
    """The block runs as a turn of ``conversation``, unless it is None or already the active turn."""
    if conversation is None or conversation == get_active_turn():
        yield
        return
    token = set_active_turn(conversation)
    try:
        yield
    finally:
        reset_active_turn(token)


def _caller_cancelling() -> bool:
    """Whether the task running ``delegate`` (the caller's: its turn's) is being cancelled."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0
