"""A delegation: the one place a sub-agent run starts, ends and is decided.

A parent's ``task`` call reaches ``Delegations.delegate`` from its backend's
task tool (ADK's ``adk/task_tool.py::TaskTool``, Deep Agents'
``deepagents/task_tool.py::build_task_tool``) as a ``DelegationRequest``.
``delegate``:

- reads the brief against the caller's own sub-agents (``brief.read_brief``):
  a brief that cannot start a run is the call's error result, and nothing
  else happens (no run, no record, no event);
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
engine) says ``input_required``, lends its slot for the wait, and says
``running`` once it has its slot back.

Each ``AGENT_RUN`` is recorded (``AgentRuns.observe``) as it is made and
reported to the turn's live events, so it reaches the turn's stream while
the run works. A run ends:

- ``failed`` when the hook raised (not a cancel) or answered an error: the
  parent's envelope carries the reason (the provider's, for a failed model
  call), and the user is warned once, a recoverable ``ERROR``:
  ``<instance> failed: <reason>``, or the hook's own words (``RunFailed``: a
  context overflow, ``context.subagent_overflow_message``);
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
  stopped it. A run's own task cancelled by anything but its control while
  its caller goes on ends ``interrupted`` too, and its caller reads that
  envelope.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers
from agentic_cli.workflow import live_events
from agentic_cli.workflow.agents.brief import BriefError, read_brief, task_message
from agentic_cli.workflow.agents.control import RunControl
from agentic_cli.workflow.agents.envelope import (
    CANCELLED_BY,
    INTERRUPTED_ERROR,
    STEP_LIMIT_ERROR,
    envelope,
)
from agentic_cli.workflow.agents.outcome import CUT, REFUSAL_ERROR, REFUSED, cut_off_warning
from agentic_cli.workflow.agents.policy import AgentLimits
from agentic_cli.workflow.agents.runs import (
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INPUT_REQUIRED,
    RUN_INTERRUPTED,
    RUN_QUEUED,
    RUN_REFUSED,
    RUN_RUNNING,
    RUN_STARTED,
    AgentRuns,
    result_head,
    run_title,
)
from agentic_cli.workflow.agents.slots import NEW, Slot, SlotPool
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    count_run_this_turn,
    current_turn_scope,
    get_active_turn,
    runs_this_turn,
    set_active_run,
    set_active_tool_call,
)

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.policy import SubAgentPolicy
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


def started_by(caller: RunRef | None) -> str:
    """Who starts a run its parent delegated: ``agent:<the caller's instance>``.

    ``agent:root`` for the root's own calls (no run, or ``root_run``).
    """
    return f"agent:{caller.instance if caller is not None else 'root'}"


@dataclass(frozen=True)
class DelegationRequest:
    """A ``task`` call, as its backend's task tool hands it over.

    Attributes:
        caller: The run that made the call (``get_active_run()`` at the
            call): None, or ``root_run``, for the root.
        caller_agent: The calling agent's type name.
        call_id: The caller's ``task`` call.
        arguments: ``task``'s arguments as the model wrote them.
        state: The shared state at the call (the plan, the tasks, the
            outside-content mark), which the run starts from.
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
class AgentRun:
    """What the backend hook is asked to run.

    Attributes:
        ref: The run.
        config: Its agent type.
        message: The task message (``brief.task_message``): the run's first
            user message.
        conversation: The conversation it runs in.
        state: The shared state it starts from.
        history: An inheriting run's parent conversation, as the backend
            reads it; None for a fresh one.
        control: Its control (``agents/control.py``): its limits, a cancel
            of the run alone, the messages steered to it. A run made
            without one gets one with no limits.
    """

    ref: RunRef
    config: AgentConfig
    message: str
    conversation: SessionRef | None
    state: Mapping[str, Any]
    history: Any = None
    control: RunControl = None  # type: ignore[assignment]  # filled below when None

    def __post_init__(self) -> None:
        if self.control is None:
            object.__setattr__(self, "control", RunControl(self.ref, AgentLimits()))


@dataclass(frozen=True)
class RunAnswer:
    """What the backend hook returns for a run it ran to its end.

    Attributes:
        text: Its last non-empty answer (``""`` for none).
        outcome: How that answer ended (``agents/outcome.py``: ``REFUSED``,
            ``CUT``), None when it ended as an answer should.
        error: Why it failed, when it did; None when it answered.
        state: The shared state keys it changed, with their new values.
    """

    text: str = ""
    outcome: str | None = None
    error: str | None = None
    state: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Delegated:
    """What the backend's task tool gets back for a ``task`` call.

    Attributes:
        result: The envelope, or the error result of a call that started no run.
        state: What the run changed of the shared state; empty when none started.
        run: The run, None when no run started.
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
    pool, the child runs it waits on (by run id), and its waits on the user."""

    ref: RunRef
    call_id: str
    pool: SlotPool
    slot: Slot
    emit: Callable[[WorkflowEvent], None]
    children: set[str] = field(default_factory=set)
    asking: int = 0

    def waits_on_only(self, child: str) -> bool:
        """Whether ``child`` is all the run waits on: its last child, the user not asked."""
        return self.asking == 0 and self.children <= {child}

    @asynccontextmanager
    async def waiting_on_user(self) -> AsyncIterator[None]:
        """``input_required``, its slot lent for the block, then ``running``; nested waits once."""
        self.asking += 1
        if self.asking == 1:
            self.emit(WorkflowEvent.agent_run(self.ref, RUN_INPUT_REQUIRED, call_id=self.call_id))
        stopped = False
        try:
            async with self.pool.lent(self.slot):
                yield
        except asyncio.CancelledError:
            stopped = True
            raise
        finally:
            self.asking -= 1
            if self.asking == 0 and not stopped:
                self.emit(WorkflowEvent.agent_run(self.ref, RUN_RUNNING, call_id=self.call_id))


#: The run at work in this task (``_in_run`` sets it in the run's own task,
#: whose copies its tool calls run in); None in the root's own work.
_working: ContextVar[_Working | None] = ContextVar("agentic_cli_working_run", default=None)


@asynccontextmanager
async def waiting_on_user() -> AsyncIterator[None]:
    """The block waits on the user (a question, a permission prompt): the run lends its slot.

    In a sub-agent run's own work, the run says ``input_required``, its slot
    works for another run meanwhile, and once the block ends it takes the
    slot back, ahead of the runs that have not started, and says
    ``running``; nested waits (a prompt waiting its turn behind another's,
    then the prompt) count once. A wait a cancel ends takes nothing back:
    the run is being stopped. Nothing in the root's own work, nor outside a
    run.
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
    envelope's ``error`` and the user's warning, as given, and ``code`` that
    warning's error code. Any other exception a hook lets out fails the run
    with its text (its type's name, for none), warned of as ``<instance>
    failed: <reason>`` with the type's name as its code.
    """

    def __init__(self, error: str, *, code: str) -> None:
        super().__init__(error)
        self.code = code


class Delegations:
    """A manager's delegations: the one place a sub-agent run starts, ends and is decided.

    Args:
        runs: The manager's runs, which name and record each run.
        policy: How the manager's sub-agents run.
        types: Every agent type, by name: a caller may start its own
            sub-agents (``AgentConfig.sub_agents``).
        hook: Runs one run (``BaseWorkflowManager._run_agent``).
        ledgers: The manager's usage ledgers: a run's end carries what it
            used, by model. None: an end carries no usage.
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
    ) -> None:
        self._runs = runs
        self.policy = policy
        self._types = dict(types)
        self._hook = hook
        self._ledgers = ledgers
        self._stored = stored
        # Each conversation's turn its stored instances were last read in
        # (its span, ``current_turn_scope``: held here, so no later turn's
        # can be the same object), and the lock a turn's parallel first runs
        # wait at while one reads them.
        self._numbered: dict[SessionRef, object] = {}
        self._numbering = asyncio.Lock()
        # The working runs' controls, by (conversation, run id).
        self._controls: dict[tuple[SessionRef | None, str], RunControl] = {}
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

    async def delegate(self, request: DelegationRequest) -> Delegated:
        """Run the sub-agent ``request`` asks for, to its end; its envelope.

        A brief that cannot start a run, or a call past the runs its turn
        may start, is the error result, and nothing else happens. A caller
        that is a run lends its slot while it waits for this one. A cancel
        of the caller's task (the turn's) ends the run ``interrupted`` once
        it has stopped, and goes on.
        """
        try:
            config, brief = read_brief(request.arguments, self.sub_agents(request.caller_agent))
        except BriefError as error:
            return Delegated(result={"success": False, "error": str(error)}, state={}, run=None)
        started = runs_this_turn()
        limit = self.policy.max_runs_per_turn
        if started is not None and started >= limit:
            logger.info("subagent_runs_per_turn_reached", agent=config.name, limit=limit)
            return Delegated(
                result={"success": False, "error": RUNS_PER_TURN_ERROR.format(limit=limit)},
                state={}, run=None,
            )
        count_run_this_turn()
        conversation = get_active_turn()
        await self._number_past_stored(conversation)
        ref = self._runs.run_for(conversation, request.caller, request.call_id, config.name)
        self._emit(WorkflowEvent.agent_run(
            ref, RUN_STARTED, call_id=request.call_id, title=run_title(brief.title),
            brief=brief.fields(), started_by=started_by(request.caller), mode=BLOCKING,
        ))
        control = RunControl(ref, self._limits(config))
        inner = asyncio.create_task(
            self._in_run(ref, config, brief, conversation, request, control),
            name=f"agent run {ref.run}",
        )
        control.bind(inner)
        key = (conversation, ref.run or "")
        self._controls[key] = control
        caller = _working.get()
        try:
            if caller is None:
                return await self._ended(ref, request.call_id, inner, control)
            # The caller waits: its slot works for another run meanwhile.
            caller.children.add(ref.run or "")
            try:
                async with caller.pool.lent(caller.slot):
                    return await self._ended(ref, request.call_id, inner, control)
            finally:
                caller.children.discard(ref.run or "")
        finally:
            self._controls.pop(key, None)

    def control(self, run: RunRef | None) -> RunControl | None:
        """The control of ``run`` while it works, in the active turn's conversation; else None.

        None for the root's own work (no run), and for a run that has ended.
        """
        if run is None or run.run is None:
            return None
        return self._controls.get((get_active_turn(), run.run))

    def cancel(self, run_id: str, *, by: str, error: str) -> bool:
        """Cancel the working run ``run_id`` of the active turn's conversation, alone.

        ``by`` its parent, the user or a limit; ``error`` says why, as its
        envelope will. Its siblings and its turn go on; its parent reads its
        envelope (``cancelled``, with what it had of its work).

        Returns:
            True when this cancelled it; False for no such working run, or
            one cancelled already.

        Raises:
            ValueError: ``by`` is not ``parent``, ``user`` or ``limit``.
        """
        if by not in CANCELLED_BY:
            raise ValueError(f"A run is cancelled by its parent, user or limit, not {by!r}.")
        control = self._controls.get((get_active_turn(), run_id))
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
        self, ref: RunRef, call_id: str, inner: asyncio.Task[RunAnswer], control: RunControl
    ) -> Delegated:
        """Wait for the run's task ``inner``, then end the run as it went; its envelope."""
        try:
            answer = await inner
        except asyncio.CancelledError:
            if not _caller_cancelling() and control.cancelled is not None:
                # Its control cancelled it and the hook let the cancel out:
                # nothing of its work came back.
                return self._cancelled(ref, call_id, control.cancelled, RunAnswer())
            self._end(ref, call_id, RUN_INTERRUPTED, error=INTERRUPTED_ERROR)
            if _caller_cancelling():
                raise
            # The run's own task was stopped, not by its control nor its
            # caller's: the caller reads that it ended.
            return Delegated(
                result=envelope(ref, RUN_INTERRUPTED, error=INTERRUPTED_ERROR), state={}, run=ref
            )
        except RunFailed as error:
            return self._failed(ref, call_id, str(error), warning=str(error), code=error.code)
        except Exception as error:  # noqa: BLE001 - a sub-agent's failure is its parent's to read
            reason = str(error) or type(error).__name__
            logger.debug("subagent_failed", run=ref.run, exc_info=error)
            return self._failed(
                ref, call_id, reason,
                warning=f"{ref.instance} failed: {reason}", code=type(error).__name__,
            )
        if _caller_cancelling():
            # The turn was stopped while the run's own cancel ended it (its
            # hook took that cancel and answered): the run is the turn's.
            self._end(ref, call_id, RUN_INTERRUPTED, error=INTERRUPTED_ERROR)
            raise asyncio.CancelledError
        if control.cancelled is not None:
            return self._cancelled(ref, call_id, control.cancelled, answer)
        if answer.error is not None:
            return self._failed(
                ref, call_id, answer.error, warning=f"{ref.instance} failed: {answer.error}",
                code=None, text=answer.text, state=answer.state,
            )
        if answer.outcome == REFUSED:
            self._end(ref, call_id, RUN_REFUSED, text=answer.text, error=REFUSAL_ERROR)
            result = envelope(ref, RUN_REFUSED, result=answer.text, error=REFUSAL_ERROR)
        else:
            stepped = STEP_LIMIT_ERROR if control.at_step_limit else None
            partial = answer.outcome == CUT or stepped is not None
            end = self._end(
                ref, call_id, RUN_COMPLETED, text=answer.text, partial=partial, error=stepped
            )
            if answer.outcome == CUT:
                warning = cut_off_warning(end)
                if warning is not None:
                    self._report(warning)
            result = envelope(
                ref, RUN_COMPLETED, result=answer.text, partial=partial, error=stepped
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
        async with self._numbering:
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
        brief: Any,
        conversation: SessionRef | None,
        request: DelegationRequest,
        control: RunControl,
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
        history = (
            request.history()
            if config.inherit_context and request.history is not None else None
        )
        run = AgentRun(
            ref=ref,
            config=config,
            message=task_message(brief, inherit=config.inherit_context),
            conversation=conversation,
            state=dict(request.state),
            history=history,
            control=control,
        )
        pool = self.slots(conversation)
        queued = pool.free == 0
        if queued:
            self._emit(WorkflowEvent.agent_run(ref, RUN_QUEUED, call_id=request.call_id))
        slot = await pool.acquire(NEW)
        try:
            if queued:
                self._emit(WorkflowEvent.agent_run(ref, RUN_RUNNING, call_id=request.call_id))
            _working.set(_Working(ref, request.call_id, pool, slot, emit=self._emit))
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

    def _cancelled(
        self,
        ref: RunRef,
        call_id: str,
        cancelled: tuple[str, str],
        answer: RunAnswer,
    ) -> Delegated:
        """End ``ref``'s run cancelled (``cancelled``: by whom, why) with what ``answer`` had."""
        by, error = cancelled
        logger.info("subagent_cancelled", run=ref.run, agent=ref.agent, by=by, error=error)
        partial = bool(answer.text)
        self._end(
            ref, call_id, RUN_CANCELLED, text=answer.text, error=error, partial=partial,
            cancelled_by=by,
        )
        return Delegated(
            result=envelope(
                ref, RUN_CANCELLED, result=answer.text, partial=partial, error=error,
                cancelled_by=by,
            ),
            state=dict(answer.state), run=ref,
        )

    def _failed(
        self,
        ref: RunRef,
        call_id: str,
        error: str,
        *,
        warning: str,
        code: str | None,
        text: str = "",
        state: Mapping[str, Any] | None = None,
    ) -> Delegated:
        """End ``ref``'s run failed with ``error``, warn the user, and its envelope."""
        logger.info("subagent_failed", run=ref.run, agent=ref.agent,
                    error=truncate(error, _MAX_LOGGED_ERROR))
        self._end(ref, call_id, RUN_FAILED, text=text, error=error)
        warned = WorkflowEvent.error(warning, error_code=code, recoverable=True)
        warned.metadata["agent"] = ref.agent
        warned.run = ref
        self._report(warned)
        return Delegated(
            result=envelope(ref, RUN_FAILED, result=text, error=error),
            state=dict(state or {}), run=ref,
        )

    def _end(
        self,
        ref: RunRef,
        call_id: str,
        status: str,
        *,
        text: str = "",
        error: str | None = None,
        partial: bool = False,
        cancelled_by: str | None = None,
    ) -> WorkflowEvent:
        """Report and record the ``AGENT_RUN`` that ends ``ref``'s run; that event."""
        end = WorkflowEvent.agent_run(
            ref, status, call_id=call_id, result=result_head(text), error=error,
            partial=partial, usage=self._usage(ref), cancelled_by=cancelled_by,
        )
        self._emit(end)
        return end

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


def _caller_cancelling() -> bool:
    """Whether the task running ``delegate`` (the caller's: its turn's) is being cancelled."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0
