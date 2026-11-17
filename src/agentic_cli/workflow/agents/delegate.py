"""A delegation: the one place a sub-agent run starts, ends and is decided.

A parent's ``task`` call reaches ``Delegations.delegate`` from its backend's
task tool (ADK's ``adk/task_tool.py::TaskTool``, Deep Agents'
``deepagents/task_tool.py::build_task_tool``) as a ``DelegationRequest``.
``delegate``:

- reads the brief against the caller's own sub-agents (``brief.read_brief``):
  a brief that cannot start a run is the call's error result, and nothing
  else happens (no run, no record, no event);
- names the run (``AgentRuns.run_for``: its instance ``<type>-<n>`` and its run
  ``<instance>#1``, under the caller's run) and reports its ``AGENT_RUN``
  ``started``: the title, the call, the brief, who started it, ``blocking``;
- runs it through the backend's hook (``BaseWorkflowManager._run_agent``) in
  an asyncio task of its own, in which the run is the active run
  (``get_active_run``) and no tool call is (``get_active_tool_call``): the
  run's model calls, tool calls, permission checks and questions are its
  own, never its caller's ``task`` call's;
- ends the run as the hook's answer says, reports the end (and a cut run's
  or a failed run's one warning after it), and returns the result envelope
  (``envelope.envelope``) with the shared state the run changed.

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
  limit, followed by the cut's one warning (``outcome.cut_off_warning``);
- ``interrupted`` when its caller's task is cancelled (Ctrl+C stops the
  turn): the cancel goes on once the run has stopped. A run's own task
  cancelled while its caller goes on ends ``interrupted`` too, and its
  caller reads that envelope.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers
from agentic_cli.workflow import live_events
from agentic_cli.workflow.agents.brief import BriefError, read_brief, task_message
from agentic_cli.workflow.agents.envelope import INTERRUPTED_ERROR, envelope
from agentic_cli.workflow.agents.outcome import CUT, REFUSAL_ERROR, REFUSED, cut_off_warning
from agentic_cli.workflow.agents.runs import (
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INTERRUPTED,
    RUN_REFUSED,
    RUN_STARTED,
    AgentRuns,
    result_head,
    run_title,
)
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    get_active_turn,
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
    """

    ref: RunRef
    config: AgentConfig
    message: str
    conversation: SessionRef | None
    state: Mapping[str, Any]
    history: Any = None


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
    """

    def __init__(
        self,
        *,
        runs: AgentRuns,
        policy: SubAgentPolicy,
        types: Mapping[str, AgentConfig],
        hook: RunHook,
        ledgers: UsageLedgers | None = None,
    ) -> None:
        self._runs = runs
        self.policy = policy
        self._types = dict(types)
        self._hook = hook
        self._ledgers = ledgers

    def sub_agents(self, agent: str) -> dict[str, AgentConfig]:
        """The types ``agent`` may start, by name, in its ``sub_agents``' order."""
        caller = self._types.get(agent)
        names = caller.sub_agents if caller is not None else ()
        return {name: self._types[name] for name in names if name in self._types}

    async def delegate(self, request: DelegationRequest) -> Delegated:
        """Run the sub-agent ``request`` asks for, to its end; its envelope.

        A brief that cannot start a run is the error result, and nothing
        else happens. A cancel of the caller's task (the turn's) ends the run
        ``interrupted`` once it has stopped, and goes on.
        """
        try:
            config, brief = read_brief(request.arguments, self.sub_agents(request.caller_agent))
        except BriefError as error:
            return Delegated(result={"success": False, "error": str(error)}, state={}, run=None)
        conversation = get_active_turn()
        ref = self._runs.run_for(conversation, request.caller, request.call_id, config.name)
        self._emit(WorkflowEvent.agent_run(
            ref, RUN_STARTED, call_id=request.call_id, title=run_title(brief.title),
            brief=brief.fields(), started_by=started_by(request.caller), mode=BLOCKING,
        ))
        inner = asyncio.create_task(
            self._in_run(ref, config, brief, conversation, request), name=f"agent run {ref.run}"
        )
        try:
            answer = await inner
        except asyncio.CancelledError:
            self._end(ref, request.call_id, RUN_INTERRUPTED, error=INTERRUPTED_ERROR)
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            # The run's own task was stopped, not its caller's: the caller
            # reads that it ended.
            return Delegated(
                result=envelope(ref, RUN_INTERRUPTED, error=INTERRUPTED_ERROR), state={}, run=ref
            )
        except RunFailed as error:
            return self._failed(ref, request.call_id, str(error), warning=str(error), code=error.code)
        except Exception as error:  # noqa: BLE001 - a sub-agent's failure is its parent's to read
            reason = str(error) or type(error).__name__
            logger.debug("subagent_failed", run=ref.run, exc_info=error)
            return self._failed(
                ref, request.call_id, reason,
                warning=f"{ref.instance} failed: {reason}", code=type(error).__name__,
            )
        if answer.error is not None:
            return self._failed(
                ref, request.call_id, answer.error, warning=f"{ref.instance} failed: {answer.error}",
                code=None, text=answer.text, state=answer.state,
            )
        if answer.outcome == REFUSED:
            self._end(ref, request.call_id, RUN_REFUSED, text=answer.text, error=REFUSAL_ERROR)
            result = envelope(ref, RUN_REFUSED, result=answer.text, error=REFUSAL_ERROR)
        else:
            partial = answer.outcome == CUT
            end = self._end(ref, request.call_id, RUN_COMPLETED, text=answer.text, partial=partial)
            warning = cut_off_warning(end)
            if warning is not None:
                self._report(warning)
            result = envelope(ref, RUN_COMPLETED, result=answer.text, partial=partial)
        return Delegated(result=result, state=dict(answer.state), run=ref)

    async def _in_run(
        self,
        ref: RunRef,
        config: AgentConfig,
        brief: Any,
        conversation: SessionRef | None,
        request: DelegationRequest,
    ) -> RunAnswer:
        """The run, in its own task: the active run, and no tool call active.

        The task's context is a copy of its caller's, so what this sets stays
        here. An inheriting type's history is read here, once.
        """
        set_active_run(ref)
        set_active_tool_call(None)
        history = (
            request.history()
            if config.inherit_context and request.history is not None else None
        )
        return await self._hook(AgentRun(
            ref=ref,
            config=config,
            message=task_message(brief, inherit=config.inherit_context),
            conversation=conversation,
            state=dict(request.state),
            history=history,
        ))

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
    ) -> WorkflowEvent:
        """Report and record the ``AGENT_RUN`` that ends ``ref``'s run; that event."""
        end = WorkflowEvent.agent_run(
            ref, status, call_id=call_id, result=result_head(text), error=error,
            partial=partial, usage=self._usage(ref),
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
