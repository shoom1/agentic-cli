"""Abstract base class for workflow managers.

This module defines the interface that all workflow orchestration backends
must implement, enabling pluggable orchestrators (ADK, LangGraph, etc.).

The base class provides auto-detection of required managers (memory, planning,
HITL) based on tool requirements, creating them lazily when needed.

It also provides shared implementations for:
- The lifecycle: initialization, ``cleanup``, ``reinitialize`` and the one
  release they run, which the backends join through two hooks
- The turn: one entry (``_turn``) for every turn, and the templates of a
  job's resume, ``can_resume`` and ``compact`` over the backends' hooks
- User input handling (callback-based)
- Models: the model catalog, and every agent's model resolved at startup
- Context: each agent's context numbers (``context_budget``), for a UI
- The memory index the root agent's requests carry, one snapshot per
  conversation (``memory_index.py``)
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import warnings
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from types import MappingProxyType
from typing import Any, AsyncGenerator, Awaitable, Callable, Iterator, Mapping, TYPE_CHECKING

from agentic_cli.workflow.events import (
    EventType,
    UserInputRequest,
    UserInputUnavailable,
    WorkflowEvent,
)
from agentic_cli.models import (
    CatalogView,
    ModelCatalog,
    ModelRef,
    ModelResolutionError,
    ModelSpecError,
    Provider,
    ProviderKind,
    ResolvedModel,
    RoleMap,
    resolve_model,
)
from agentic_cli.workflow import live_events
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.agents.control import configured_tool_names
from agentic_cli.workflow.agents.delegate import (
    STOPPED_BY_USER_ERROR,
    USER,
    AgentRun,
    Delegated,
    Delegations,
    FollowUpRequest,
    RunAnswer,
    waiting_on_user,
)
from agentic_cli.workflow.agents.policy import SubAgentPolicy, as_policy
from agentic_cli.workflow.agents.records import RecordsWriter
from agentic_cli.workflow.agents.roster import Rosters
from agentic_cli.workflow.agents.runs import (
    INSTANCE_CLOSED,
    INSTANCE_RUNNING,
    RUN_END_STATUSES,
    RUN_STARTED,
    AgentRuns,
)
from agentic_cli.workflow.agents.spawn import GENERAL_PURPOSE, can_have_children
from agentic_cli.workflow.usage import (
    PURPOSE_MEMORY_EXTRACTION,
    PURPOSE_SIDE_CALL,
    PURPOSE_TOOL,
    UsageLedger,
    UsageLedgers,
    current_side_call,
    side_call,
)
from agentic_cli.workflow.interrupts import OpenCalls
# The two limits are still importable from here.
from agentic_cli.workflow.memory_index import (  # noqa: F401
    MEMORY_INDEX_BUDGET_SHARE,
    MEMORY_INDEX_MIN_BYTES,
    MemoryIndex,
)
from agentic_cli.workflow.model_settings import ModelSettings, ThinkingSettings
from agentic_cli.workflow.outside_content import OutsideContentMarks, outside_content_note
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionInfo,
    SessionRef,
    close_turn_scope,
    get_active_run,
    get_active_tool_call,
    get_active_turn,
    open_turn_scope,
    reset_active_turn,
    set_active_turn,
    turn_has_ended,
)
from agentic_cli.workflow.tool_output import cap_tool_result, tool_output_dir
from agentic_cli.workflow.service_registry import (
    get_service,
    set_service_registry,
    JOB_MANAGER,
    KNOWLEDGE_BASE,
    LLM_SUMMARIZER,
    MEMORY_STORE,
    SANDBOX_MANAGER,
    WORKFLOW,
)
from agentic_cli.workflow.services import (
    SERVICE_CLOSERS,
    ServiceLoan,
    ServiceSet,
    close_services,
    rebind_services,
)
from agentic_cli.logging import Loggers, bind_context

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.config import AgentGraph
    from agentic_cli.workflow.context import CompactResult, ContextBudget
    from agentic_cli.memory import MemoryStore, Proposal
    from agentic_cli.memory.kb import KnowledgeBase
    from agentic_cli.tools.sandbox.manager import SandboxManager

logger = Loggers.workflow()

#: Where the backends' tool classes live.
_TOOL_CLASS_MODULES = ("google.adk.tools", "langchain_core.tools")

#: One thing a release closes: a name for the log, and the call that closes it
#: (a coroutine function, or a plain function). The detach hooks return them
#: (see ``BaseWorkflowManager._release_resources``).
Closer = tuple[str, Callable[[], Any]]

#: Releases still closing after a cancel stopped their wait: held so none is
#: garbage collected halfway (``BaseWorkflowManager._release_resources``).
_CLOSING: set[asyncio.Future] = set()

#: How long a release waits for the sub-agent records still being written, in
#: seconds: a store that hangs holds the rest of the release no longer.
RECORDS_RELEASE_WAIT = 10.0


class _UserRun:
    """A run the user started (``run_instance``, ``start_instance``), while its
    turn lasts: its task once it starts, and whether it is interrupted
    (``interrupt_instance``), which makes a stop of its stream a turn's
    cancel, not the user's stop."""

    def __init__(self) -> None:
        self.work: asyncio.Task[Delegated] | None = None
        self.interrupted = False


class _UsersRunEvents(live_events.LiveEvents):
    """The live events of a run the user starts (``_users_run``).

    The first run whose start they report is that run (the runs it starts
    are reported later), noted as its start is reported, before anything
    else of it, so a stop of the stream finds it even before the stream's
    consumer read that start. ``Delegations`` reports a start on the loop.
    """

    def __init__(self) -> None:
        super().__init__()
        #: The run the user started, once its start is reported.
        self.started: RunRef | None = None
        self._waiting: asyncio.Future[None] | None = None

    def emit(self, event: WorkflowEvent) -> None:
        if (
            self.started is None
            and event.type is EventType.AGENT_RUN
            and event.metadata.get("status") == RUN_STARTED
            and event.run is not None
        ):
            self.started = event.run
            self._wake()
        super().emit(event)

    def _wake(self) -> None:
        waiting = self._waiting
        if waiting is not None and not waiting.done():
            waiting.set_result(None)

    async def until_started(self, work: "asyncio.Task[Any]") -> None:
        """Wait until the run has started, or ``work`` (what starts it) has ended.

        One future, resolved by the report of the start or by the end of
        ``work`` and awaited directly: the caller then runs in the step after
        the report, ahead of the run's own first step, so a stop it makes
        cancels the run before its hook is entered.
        """
        if self.started is not None or work.done():
            return
        self._waiting = asyncio.get_running_loop().create_future()
        work.add_done_callback(lambda _: self._wake())
        await self._waiting


def _module_of(klass: type) -> str:
    module = getattr(klass, "__module__", None)
    return module if isinstance(module, str) else ""


def _wrapped_callable(tool: Any) -> Callable | None:
    """The function a backend's function-tool object runs, unwrapped as the
    permission gates unwrap it (exact ADK ``FunctionTool`` and
    ``LongRunningFunctionTool``, exact LangChain ``StructuredTool``), or None."""
    kind = type(tool)
    try:
        from google.adk.tools import FunctionTool, LongRunningFunctionTool
    except ImportError:
        pass
    else:
        if kind is FunctionTool or kind is LongRunningFunctionTool:
            return tool.func
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        pass
    else:
        if kind is StructuredTool:
            return tool.coroutine or tool.func
    return None


def _tool_name(tool: Any) -> str:
    """The name a model would see for ``tool``."""
    try:
        inner = tool.func if isinstance(tool, functools.partial) else tool
        for attribute in ("name", "__name__"):
            value = getattr(inner, attribute, None)
            if isinstance(value, str) and value:
                return value
    except Exception:  # a __getattr__ that raises
        pass
    return type(tool).__name__


def _unregistered_name(tool: Any) -> str | None:
    """The name of ``tool``, which the registry did not issue, when the gate
    will deny every call to it; None when the gate never sees it.

    A function-tool object is judged by the function it runs. ADK's own tool
    objects are left to ADK: a built-in such as ``GoogleSearchTool`` runs
    inside the model and never reaches the gate. Anything else callable, and an
    app's own tool class (either backend's), is denied.
    """
    from agentic_cli.tools.registry import identify_tool

    if isinstance(tool, (str, type)):
        return None
    inner = _wrapped_callable(tool)
    if inner is not None:
        return None if identify_tool(inner) is not None else _tool_name(tool)
    kind = type(tool)
    if _module_of(kind).startswith("google.adk."):
        return None
    is_tool_object = any(
        _module_of(klass).startswith(_TOOL_CLASS_MODULES) for klass in kind.__mro__
    )
    return _tool_name(tool) if callable(tool) or is_tool_object else None


class BaseWorkflowManager(ABC):
    """Abstract base class for workflow managers.

    Defines the interface that all workflow orchestration backends must implement.
    This enables pluggable orchestrators like Google ADK, LangGraph, etc.

    Implementations must handle:
    - Agent orchestration based on AgentConfig definitions
    - Session management
    - Event streaming
    - User input request/response flow

    Concurrency contract:
        A manager runs **one turn at a time**. ``process()``,
        ``resume_with_job_result()``, ``compact()``, ``run_instance()``
        (the user's follow-up of a sub-agent) and ``tell_root()`` enter through
        ``_turn`` and serialize on an internal turn lock, so overlapping
        callers queue rather than interleave — the backend's per-invocation
        event buffers are manager-scoped, and interleaving would let one
        invocation drain another's events. Run separate managers for genuine
        parallelism. (The HITL input callback is context-local, so it does
        not depend on that serialization; see ``set_input_callback``.)

        ``reinitialize`` and ``cleanup`` take the lifecycle lock and then the
        turn lock, so the backend is never torn down under a running
        generator. ``initialize_services`` takes only the lifecycle lock: a
        turn initializes before it takes the turn lock (``_turn_admission``),
        which is what keeps the two from deadlocking. Both locks are released
        on cancellation. The active session/user identity remains a
        ContextVar, so it stays correct for nested and task-spawned work.

    Lifecycle contract:
        The base owns the lifecycle: ``initialize_services`` (transactional:
        a failed attempt is rolled back), ``cleanup``, ``reinitialize`` (a
        model switch, which keeps the session store and the services with
        ``_keeping``) and the one release they all run,
        ``_release_resources``. A backend builds in ``_do_initialize`` and
        takes part in the release through two hooks, ``_detach_backend``
        (what it built for this initialization) and ``_detach_store`` (the
        store its conversations live in), each synchronous, detaching its
        handles and returning their closers; the base closes them under a
        shield, so cancelling the caller never leaves one open (only an event
        loop shutting down, which cancels the closing task itself, skips the
        closers after the one running). A backend that overrides ``cleanup``
        or ``reinitialize`` keeps its own (the deprecated LangGraph backend
        does).

        A run's services (the permission engine and its session grants, the
        sandbox kernels, the job manager and its in-process jobs, the
        knowledge bases and the embedding model, the memory store) can move
        to another manager, of either backend, instead of being built
        again::

            loan = old.lend_services()
            await new.initialize_services(services=loan)  # builds only what is missing
            await old.give_services(loan)  # new owns them now
            await old.cleanup()  # closes none of them

        Until ``give_services`` the old manager owns them: a new manager
        that fails to initialize, is cancelled while it does, or is cleaned
        up before they are given closes none of them, and the old one keeps
        working with them (nothing has to hand them back). ``give_services``
        waits for the old manager's running turn (inside one of its turns it
        raises ``RuntimeError`` rather than wait for itself), makes the new
        manager their owner and moves what they hold of their owner to it:
        the permission engine's prompts and per-call say, the knowledge
        base's summarizer. Each manager keeps its own ``WORKFLOW`` and
        summarizer entries and builds its own service-bound tools. A manager
        that holds services it does not own runs no turn
        (``_turn_admission`` raises ``RuntimeError``), since the engine's
        per-call check reads its owner's outside-content marks: the new
        manager until the give, the old one after it, until it is released.

        The base also computes each agent's context numbers (``_budgets``),
        once the models are resolved and before any service is built, so a
        model too small to work in fails the start with nothing allocated: a
        backend that summarizes and trims by them says so
        (``_manages_context``) and reads ``_budgets`` in ``_do_initialize``,
        and computes none itself.

    Turn contract:
        Every turn starts with ``async with self._turn(session_id, user_id)
        as ref:``: the conversation is resolved (``session_ref``) and named in
        the log context, the manager initialized when it is not, the turn
        lock taken with a live backend behind it (``_turn_admission``), and
        the settings, the services and the active turn set for the block
        (``_workflow_context``). A backend's ``process`` runs its agents
        inside it, and a turn that fails yields one ``ERROR`` (``process``
        says what it holds). The rest of the turn protocol is the base's, over hooks:
        ``resume_with_job_result`` (the record's checks, the turn, the job's
        result read on a worker thread, the notice) runs the backend's
        ``_deliver_job_notice`` in the conversation ``_conversation_exists``
        finds, and ``can_resume`` makes the same checks without the turn;
        ``compact`` runs the backend's ``_compact_turn`` as a turn. A backend
        implements the hooks it can, and ``supports_job_resume`` and
        ``supports_compact`` are derived from them; one that overrides a
        template keeps its own.

    Memory index:
        The root agent's requests carry an index of the saved memories, one
        snapshot per conversation, which the manager holds as a
        ``MemoryIndex`` (``memory_index.py``). A backend that carries it asks
        ``_memory_index_wanted(root)`` (the setting is on and the root has a
        memory tool) and gives the root ``_memory_index_now`` (the snapshot,
        taken the first time; it reads files, so off the event loop),
        ``_memory_index_cached`` (never reads) and ``_forget_memory_index``
        (a summary written, ``/resume``, ``/compact``). Each is the manager's
        own method, which forwards to the index, so a subclass may override
        one. The release clears the snapshots. A root that continues its
        sub-agents (``_roster_wanted``: it has ``message_agent``) also gets
        their roster after the index, a snapshot per conversation too
        (``agents/roster.py``): ``_roster_now`` takes it, and
        ``_forget_memory_index`` drops it with the index's.

    Example:
        A minimal backend: it builds its engine in ``_do_initialize`` and
        hands it to the release in ``_detach_backend``; everything else of
        the lifecycle is the base's. ``initialize_services`` calls the
        backend's ``_validate_agent_graph`` (the base's validates nothing and
        leaves ``root_agent_name`` None), resolves the models, computes the
        context budgets (none: ``_manages_context`` says False), builds the
        services the tools declare and then calls ``_do_initialize``. A turn
        enters through ``_turn``, which initializes the manager when it is
        not and re-checks ``_backend_ready`` under the turn lock, and passes
        each event through ``_apply_event_hook``, which is the constructor's
        ``on_event``. A backend with a session store also returns its
        closers from ``_detach_store``, which a model switch skips, and
        builds on the store it still holds. This one has no job resume and
        no ``compact``: a backend that does implements their hooks (the turn
        contract above)::

            from functools import partial
            from typing import Callable

            from agentic_cli import BaseWorkflowManager, WorkflowEvent
            from agentic_cli.workflow.config import AgentGraph, validate_agent_graph

            class CustomWorkflowManager(BaseWorkflowManager):
                _engine = None  # what a turn runs on

                @property
                def backend_type(self) -> str:
                    return "custom"

                def _get_state_tools(self) -> list[Callable]:
                    return []  # the backend's plan and task tools

                def _validate_agent_graph(self) -> AgentGraph:
                    return validate_agent_graph(
                        self._agent_configs, backend=self.backend_type,
                        policy=self.subagent_policy,
                    )

                async def _do_initialize(self) -> None:
                    self._engine = object()  # build the backend's agents or graph

                def _backend_ready(self) -> bool:
                    return self._initialized and self._engine is not None

                def _detach_backend(self):
                    engine, self._engine = self._engine, None
                    return [("engine", partial(self._aclose_owned, engine, "engine"))]

                async def process(self, message, user_id, session_id=None):
                    async with self._turn(session_id, user_id) as ref:
                        event = WorkflowEvent.text(f"echo: {message}", ref.session_id)
                        event = self._apply_event_hook(event)  # None: dropped
                        if event is not None:
                            yield event
    """

    #: The conversation a call without a session id belongs to
    #: (``session_ref``): ``load_session`` adopts the one a resume names (on
    #: the instance), and the CLI's /status shows it. A class attribute, so a
    #: backend that provides its own (set before or after the base constructor,
    #: a class default or a property) is never overwritten by the base.
    session_id: str = "default_session"

    #: True while a model switch keeps the session store (``_keeping``): a
    #: release then leaves it attached and open (``_detach_store`` is not
    #: called). A class default, so it holds for a manager whatever built it.
    _keeping_store: bool = False

    #: The session store it was built for (``session_store``); None: its
    #: settings'. A class default, so it holds whatever built the manager.
    _built_for_store: str | None = None

    #: Inside a release, the closes of the services it released, started on
    #: worker threads, for the release to wait for (``_cleanup_managers``);
    #: None outside one. A class default, as ``_keeping_store`` is.
    _services_closing: "list[asyncio.Future[None]] | None" = None

    #: How this manager's sub-agents run (``subagent_policy``): the one
    #: ``__init__`` was given. A class default, so a manager made without
    #: ``__init__`` (a test harness's ``__new__``) has the default policy.
    _subagent_policy: SubAgentPolicy = SubAgentPolicy()

    def __init__(
        self,
        agent_configs: list[AgentConfig],
        settings: "BaseSettings | None" = None,
        app_name: str | None = None,
        model: str | None = None,
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
        *,
        subagents: SubAgentPolicy | None = None,
    ) -> None:
        """Initialize the workflow manager base.

        Args:
            agent_configs: List of agent configurations defining the workflow.
            settings: Application settings (resolved via get_settings() if None).
            app_name: The name conversations are kept under: ``app_name``
                returns it and ``session_ref`` puts it in every conversation's
                identity (the backends' session stores and checkpoints key by
                it), and an OpenRouter provider is told it (its HTTP-Referer
                and X-Title headers). Defaults to ``settings.app_name``. What
                the app keeps on disk and builds for the tools (the jobs
                folder, tool outputs, the permission context) is named by
                ``settings.app_name``, whatever this is.
            model: This manager's default model, as a ``provider:model``
                reference or a model id (never a role). It takes the
                ``default`` role for this manager only; without it, the
                default is ``settings.default_model``, else the newest
                default-tier model in ``settings.models``, else of the first
                provider with a key.
            on_event: Optional hook to transform/filter events before yielding.
            subagents: How this manager's sub-agents run
                (``subagent_policy``); None is ``SubAgentPolicy()``.

        Raises:
            TypeError: ``subagents`` is neither a ``SubAgentPolicy`` nor None.
        """
        from agentic_cli.config import get_settings

        self._subagent_policy = as_policy(subagents)
        self._agent_configs = agent_configs
        self._settings = settings or get_settings()
        self._app_name = app_name or self._settings.app_name
        self._initialized = False
        # --- Concurrency contract (see the class docstring) ---
        # _lifecycle_lock serializes initialize/reinitialize/cleanup: background
        # init and a first user message may call them concurrently, and the
        # _initialized guard alone is check-then-act.
        # _turn_lock serializes turns, and is taken by reinitialize and cleanup
        # (after the lifecycle lock) so they cannot tear the backend down under
        # a running generator.
        # Lock order is lifecycle → turn: a turn releases the lifecycle lock
        # (inside _ensure_initialized) *before* taking the turn lock, so the
        # two can never deadlock.
        self._lifecycle_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._on_event = on_event

        # Models. The override takes the default role for this manager; the
        # catalog and the resolved models are built by each initialization
        # (see _load_models) and the catalog is released with the backend.
        self._model_override: str | None = model
        self._catalog: ModelCatalog | None = None
        self._default_model: ResolvedModel | None = None
        self._resolved_models: dict[str, ResolvedModel] = {}

        # Context. The root agent's name comes from the validated agent graph;
        # each agent's numbers are computed by the base when the models are
        # resolved (_compute_budgets), and stay empty on a backend that manages
        # no context.
        self._root_agent_name: str | None = None
        self._budgets: dict[str, "ContextBudget"] = {}
        # Each agent's model choices, resolved with its model and given their
        # own numbers, by (agent, choice as written).
        self._choice_models: dict[tuple[str, str], ResolvedModel] = {}
        self._choice_budgets: dict[tuple[str, str], "ContextBudget"] = {}

        # User input handling (callback-only), context-local — see
        # set_input_callback(). One ContextVar per manager; there are only ever
        # a handful of managers in a process.
        self._user_input_callback: ContextVar[
            Callable[[UserInputRequest], Awaitable[str]] | None
        ] = ContextVar(f"agentic_cli_input_callback_{id(self):x}", default=None)

        # Resolve string / dotted-path tool refs to callables BEFORE manager
        # detection and tool assembly (both key on ``tool.__name__``).
        self._resolve_config_tool_refs()

        # Auto-detect required managers from tools
        self._required_managers = self._detect_required_managers()

        # The services the tools declared (workflow/services.py): published
        # for them as ``_services``; which of them a reinitialize carries over
        # is ``_carried_service_keys`` (see _keeping()). Plan/task state lives
        # in native backend state (ToolContext.state for ADK, graph state for
        # LangGraph), not here.
        self._services_held = ServiceSet()
        # Which conversations have read outside content (workflow/outside_content.py).
        self._outside_marks = OutsideContentMarks()
        # What became of the turns' tool calls: a cancelled turn answers the
        # ones it leaves open from it (workflow/interrupts.py). A question a
        # tool asks marks its call as waiting at the prompt (request_user_input).
        # A backend that made its own before this constructor (or provides one
        # on its class) keeps it.
        if "_open_calls" not in vars(self) and not hasattr(type(self), "_open_calls"):
            self._open_calls = OpenCalls()
        # The memory index each conversation's main agent gets (spec §11.1):
        # taken once, so the prompt prefix stays cached (workflow/memory_index.py).
        self._memory_index_held = self._new_memory_index()
        # The (agent, tool) pairs already named as unregistered: once each.
        self._unregistered_named: set[tuple[str, str]] = set()

    def set_input_callback(
        self, callback: Callable[[UserInputRequest], Awaitable[str]]
    ) -> "Token | None":
        """Register a callback for handling user input requests from tools.

        **Context-local**, not manager-global. The turn lock serialises
        ``process()``, but callbacks are installed *before* it: with a single
        manager attribute, a second consumer that installed its callback while
        a turn was already running would answer that turn's prompt, and the
        first consumer's ``clear_input_callback()`` would then unregister the
        second's. A task started after this call inherits the value (the
        context is copied at ``create_task`` time), which is exactly the
        consumer → turn relationship.

        Returns:
            The ContextVar token, for an exact ``clear_input_callback(token)``.
            Callers may ignore it.
        """
        return self._user_input_callback.set(callback)

    def clear_input_callback(self, token: "Token | None" = None) -> None:
        """Remove the user input callback *this context* registered.

        Args:
            token: The token ``set_input_callback()`` returned. Passing it
                restores whatever was installed before; without it the value is
                cleared for this context only. Either way another consumer's
                callback is untouched.
        """
        if token is not None:
            try:
                self._user_input_callback.reset(token)
                return
            except ValueError:
                # Token from a different context (the caller crossed tasks);
                # fall through and clear this context's value instead.
                pass
        self._user_input_callback.set(None)

    def can_request_user_input(self) -> bool:
        """Whether ``request_user_input`` could reach the user from this context now.

        False when no input callback is set for this context, or the turn
        this context comes from has ended: ``request_user_input`` then raises
        ``UserInputUnavailable`` at once. The permission engine asks this
        before a sub-agent run lends its working slot for a prompt: a prompt
        no one can be shown is denied without waiting, and the run keeps its
        slot. A subclass that overrides ``request_user_input`` to ask
        through another channel overrides this too, saying when that
        channel can answer: else the engine takes its prompts for ones no
        one can be shown, and a sub-agent run keeps its slot while it waits
        for the answer.
        """
        return self._user_input_callback.get() is not None and not turn_has_ended()

    @property
    def agent_configs(self) -> list[AgentConfig]:
        """Get the agent configurations."""
        return self._agent_configs

    @property
    def subagent_policy(self) -> SubAgentPolicy:
        """How this manager's sub-agents run (provisional): the ``subagents=``
        it was built with, ``SubAgentPolicy()`` without one.

        Read-only: a manager keeps its policy for its life, a model switch
        (``reinitialize``) included; another policy is another manager's.
        """
        return self._subagent_policy

    @property
    def agent_runs(self) -> AgentRuns:
        """This manager's sub-agent runs, kept by conversation (provisional).

        Named by the backends' event processors and recorded from their
        ``AGENT_RUN`` and ``AGENT_INSTANCE`` events
        (``workflow/agents/runs.py``): ``records(ref)`` lists a
        conversation's runs in the order they started and ``instances(ref)``
        its sub-agent instances, each with its status; a turn ends the runs
        it leaves open (``_turn``). Kept in memory for this manager's life
        (a model switch keeps it), and beside each conversation in its store
        (the backend's store hooks): written as they change, loaded with the
        conversation (``load_agents``), so another manager (after a restart,
        or a backend's) reads them; made on first use.
        """
        runs = self.__dict__.get("_agent_runs")
        if runs is None:
            runs = self.__dict__["_agent_runs"] = AgentRuns(
                on_change=self._agent_records_changed
            )
        return runs

    @property
    def usage(self) -> UsageLedger:
        """The open conversation's token usage: every model call made in it (provisional).

        A row per call, the root's and its sub-agents' steps and the side
        calls (summaries, ``generate_simple``), each counted alike on every
        backend (``workflow/usage.py``): totals by run, agent, model, purpose
        and turn. In memory, for this manager's life (a backend switch builds
        a new manager, and the rows before it do not follow; each run's
        totals by model are kept with its record, beside the conversation).
        ``usage_of`` reads another conversation's.
        """
        return self.usage_of(self.session_ref())

    def usage_of(self, conversation: SessionRef) -> UsageLedger:
        """``conversation``'s token usage, as ``usage`` is the open one's (provisional)."""
        return self._usage_ledgers.ledger(conversation)

    @property
    def _usage_ledgers(self) -> UsageLedgers:
        """Every conversation's ledger, made on first use (as ``agent_runs``)."""
        ledgers = self.__dict__.get("_usage_ledgers")
        if ledgers is None:
            ledgers = self.__dict__["_usage_ledgers"] = UsageLedgers()
        return ledgers

    @property
    def _delegations(self) -> Delegations:
        """This manager's delegations (``workflow/agents/delegate.py``), made on first use.

        From the agent configs, the policy (``subagent_policy``) and the
        backend's hooks (``_run_agent``, ``_stored_instances``): each
        ``task`` call of a delegating agent hands its brief to ``delegate``,
        which names, runs, ends and records the run.
        """
        delegations = self.__dict__.get("_delegations_made")
        if delegations is None:
            delegations = self.__dict__["_delegations_made"] = Delegations(
                runs=self.agent_runs,
                policy=self.subagent_policy,
                types={config.name: config for config in self._agent_configs},
                hook=self._run_agent,
                ledgers=self._usage_ledgers,
                stored=self._stored_instances,
                open_calls=getattr(self, "_open_calls", None),
                tool_names=self._agent_tool_names,
            )
        return delegations

    async def _stored_instances(self, conversation: SessionRef) -> Mapping[str, int]:
        """How many instances of each agent type ``conversation``'s store keeps (a backend hook).

        The highest stored instance number of each type, by type: what a
        store holds of another manager's runs (before a restart or a switch
        of backend or store, or another process's between two turns), which
        this manager's numbering goes past (``Delegations``, once per
        conversation per turn). A sub-agent instance's transcript is ADK's
        child session and Deep Agents' checkpointer thread of its own; the
        base keeps none.
        """
        return {}

    # ---- Sub-agent records beside the conversation ----
    #
    # A conversation's records (``agent_runs``: its instances, its runs with
    # their briefs, envelopes and usage, its id counters) are kept in its
    # store through the backend's hooks: written at each run event that
    # changes them (``RecordsWriter``: one write at a time per conversation,
    # the latest records each time), all written before a turn returns,
    # loaded once per conversation (``load_agents``: ``load_session`` and each
    # turn's start), and deleted with the conversation by the backend's
    # session code, after the transcripts and before the conversation
    # (``_deleting`` around the backend's delete). The base keeps none: its
    # records are this manager's, in memory.

    async def _load_agent_records(self, conversation: SessionRef) -> Mapping[str, Any] | None:
        """``conversation``'s stored records (an ``AgentRuns.export``), or None (a backend hook).

        None when the store keeps none. A store that cannot be read raises.
        The base keeps none.
        """
        return None

    async def _save_agent_records(
        self, conversation: SessionRef, data: Mapping[str, Any]
    ) -> None:
        """Keep ``data`` (``AgentRuns.export``) as ``conversation``'s records (a backend hook).

        Replaces what the store kept. Called one at a time per conversation
        (``RecordsWriter``); a failure is warned of once per conversation and
        fails nothing. The base keeps none.
        """
        return None

    async def load_agents(self, conversation: SessionRef) -> None:
        """Load ``conversation``'s sub-agent records from its store, once (provisional).

        ``load_session`` and each turn's start call it; a call after the
        first that loaded does nothing (a conversation is marked loaded once
        its records were read into it, or ignored as unreadable: one whose
        restore failed some other way is read again), and so does one before the manager
        is initialized (its store is not open: the turn's start loads). The
        stored records replace what this manager holds of the conversation
        (``AgentRuns.restore``): a run found working is ended
        ``interrupted`` and its instance made idle, which is written back.
        Records of another version, or that do not read, are ignored with one
        warning: the conversation starts with none, numbered past the
        transcripts its store keeps (``_stored_instances``), and what the
        store keeps is never written over (this manager's runs of it are not
        saved) until they are read again (``load_session``).

        Raises:
            Exception: The store's, when its records cannot be read; nothing
                is loaded, and the next call reads it again.
        """
        loaded = self._agents_loaded
        if conversation in loaded or not self._initialized:
            return
        lock = self.__dict__.get("_agents_loading")
        if lock is None:
            lock = self.__dict__["_agents_loading"] = asyncio.Lock()
        async with lock:
            if conversation in loaded:
                return
            data = await self._load_agent_records(conversation)
            if data is None:
                loaded.add(conversation)
                return
            try:
                interrupted = self.agent_runs.restore(conversation, data)
            except ValueError as error:
                # Never written over: what cannot be read here may be read by
                # another version, and a run of this manager is not saved.
                loaded.add(conversation)
                self._agents_unwritten.add(conversation)
                logger.warning(
                    "subagent_records_ignored", session_id=conversation.session_id,
                    error=str(error), kept=True,
                )
                return
            self._agents_unwritten.discard(conversation)
            loaded.add(conversation)
            logger.info(
                "subagent_records_loaded", session_id=conversation.session_id,
                runs=len(self.agent_runs.records(conversation)), interrupted=len(interrupted),
            )
            if interrupted:
                # Written back before the load returns: a release that follows
                # at once finds the store open.
                self._agent_records.changed(conversation)
                await self._agent_records.flush(conversation)

    @property
    def _agents_loaded(self) -> set[SessionRef]:
        """The conversations whose records ``load_agents`` loaded (made on first use)."""
        loaded = self.__dict__.get("_agents_loaded_set")
        if loaded is None:
            loaded = self.__dict__["_agents_loaded_set"] = set()
        return loaded

    @property
    def _agents_unwritten(self) -> set[SessionRef]:
        """The conversations whose stored records could not be read: never written (made on first use)."""
        held = self.__dict__.get("_agents_unwritten_set")
        if held is None:
            held = self.__dict__["_agents_unwritten_set"] = set()
        return held

    @property
    def _agent_records(self) -> RecordsWriter:
        """What writes the conversations' records to their store (made on first use)."""
        writer = self.__dict__.get("_records_writer")
        if writer is None:
            writer = self.__dict__["_records_writer"] = RecordsWriter(
                export=self.agent_runs.export,
                # Looked up at each write: a test or a backend may replace it.
                save=lambda conversation, data: self._save_agent_records(conversation, data),
            )
        return writer

    def _agent_records_changed(self, conversation: SessionRef | None) -> None:
        """``conversation``'s records changed (``AgentRuns``' ``on_change``): write them.

        Only a conversation whose records were loaded is written: a write
        replaces what its store keeps.
        """
        if (
            conversation is not None and conversation in self._agents_loaded
            and conversation not in self._agents_unwritten
        ):
            self._agent_records.changed(conversation)

    def _holds_unsaved_agents(self, conversation: SessionRef) -> bool:
        """Whether this manager holds what the store lacks of ``conversation``'s records.

        Its latest write failed, or one of its runs still works (its end is
        not in any store yet): reading the store again would replace them.
        """
        writer = self.__dict__.get("_records_writer")
        if writer is not None and writer.unsaved(conversation):
            return True
        return any(
            record.status not in RUN_END_STATUSES
            for record in self.agent_runs.records(conversation)
        )

    async def _agent_records_written(self, conversation: SessionRef) -> None:
        """Wait until ``conversation``'s records are written. Never raises."""
        writer = self.__dict__.get("_records_writer")
        if writer is not None:
            await writer.flush(conversation)

    @contextlib.asynccontextmanager
    async def _deleting(self, conversation: SessionRef) -> AsyncGenerator[None, None]:
        """The block deletes ``conversation`` from the store; this manager then forgets it.

        Before the block, nothing more of its records is written (a write
        under way is waited for, so none lands after the delete, and a change
        made during it, by a run still going, writes nothing: the
        conversation is unmarked until the block ends; a block that raises
        marks it again). After it,
        unless it raised: its sub-agent records and their loading
        (``agent_runs``), the calls it recorded (``OpenCalls``: a failed
        turn's too), its usage ledger, its main agent's roster, the latest
        result of each instance the user ran (``instance_result``) and its
        numbering (``Delegations``), so
        a conversation of the same id starts as a new one.
        """
        writer = self.__dict__.get("_records_writer")
        was_loaded = conversation in self._agents_loaded
        self._agents_loaded.discard(conversation)
        try:
            if writer is not None:
                await writer.discard(conversation)
            yield
        except BaseException:
            if was_loaded:
                self._agents_loaded.add(conversation)
            raise
        self._agents_unwritten.discard(conversation)
        self.agent_runs.forget(conversation)
        open_calls = getattr(self, "_open_calls", None)  # a manager made without __init__ has none
        if open_calls is not None:
            open_calls.forget(conversation)
        self._usage_ledgers.forget(conversation)
        self._rosters.forget(conversation)
        for key in [k for k in self._instance_results if k[0] == conversation]:
            del self._instance_results[key]
        delegations = self.__dict__.get("_delegations_made")
        if delegations is not None:
            delegations.forget(conversation)

    # ---- The user's control of the open conversation's sub-agents ----
    #
    # What the CLI's ``/agent`` commands and an instance's view call
    # (provisional). Each acts on the open conversation (``session_ref()``),
    # or on the one its ``session_id`` and ``user_id`` name, as ``process``
    # takes them, on any of its instances, through ``Delegations``.

    @property
    def _user_runs(self) -> dict[tuple[SessionRef, str], _UserRun]:
        """The runs the user started that are working, by conversation and instance."""
        held = self.__dict__.get("_user_runs_held")
        if held is None:
            held = self.__dict__["_user_runs_held"] = {}
        return held

    @property
    def _instance_results(self) -> dict[tuple[SessionRef, str], dict[str, Any]]:
        """The latest envelope of each instance the user ran, by conversation and
        instance, as the cap left it (``_user_ran``)."""
        held = self.__dict__.get("_instance_results_held")
        if held is None:
            held = self.__dict__["_instance_results_held"] = {}
        return held

    async def run_instance(
        self,
        instance: str,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run the user's follow-up of a sub-agent ``instance``, as a turn of its own (provisional).

        The instance's next run, on its transcript, ``started_by`` the user:
        a turn of the conversation (``_turn``: the root does not run), whose
        events are the run's (its ``AGENT_RUN`` start and end, its text, its
        tool calls, its own sub-agents' runs). It is counted toward no
        turn's runs (``max_runs_per_turn``) and starts from the
        conversation's outside-content mark; a mark it brought marks the
        conversation here. A stopped instance runs (only the user may); one
        that is unknown, closed or running, or a blank message, is one
        ``ERROR`` event and nothing runs. The run's envelope is kept for
        ``instance_result``, through the tool result cap.

        Stopping the stream (Ctrl+C cancels the consumer, or it closes the
        generator) is the user's stop of the run: it ends ``cancelled`` by
        the user and the instance is ``stopped``, inside the turn. A run
        ``interrupt_instance`` interrupts ends as a turn's cancel ends its
        runs instead: ``interrupted``, the instance ``idle``, the stream
        raising ``asyncio.CancelledError``.

        Yields:
            The run's ``WorkflowEvent``s.

        Raises:
            Exception: The store's, when the conversation's sub-agent records
                cannot be read (``load_agents``): nothing runs.
        """
        async with self._turn(session_id, user_id) as ref:
            request = FollowUpRequest(
                instance=instance, message=message, started_by=USER, conversation=ref,
                state=await self._conversation_state(ref),
            )
            async with contextlib.aclosing(self._users_run(
                ref, lambda: self._delegations.follow_up(request),
                instance=instance, name=f"user run of {instance}",
            )) as events:
                async for event in events:
                    yield event

    async def start_instance(
        self,
        agent_type: str,
        objective: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Start a sub-agent of ``agent_type`` for the user, as a turn of its own (provisional).

        What ``/agent new <type> <objective>`` runs: the instance's first
        run, ``started_by`` the user, its parent the root (depth 1, so
        ``max_spawn_depth`` never refuses it), its brief the objective,
        titled with its first line. A turn of the conversation
        (``_turn``: the root does not run), whose first event is the run's
        ``AGENT_RUN`` start, which names the instance; then the run's events,
        as ``run_instance``'s. It is counted toward no turn's runs
        (``max_runs_per_turn``), takes a working slot while it works, and
        starts from the conversation's outside-content mark; an inheriting
        type reads the conversation as the root's ``task`` would hand it
        (``_inherited_history``). A type the user may not start (the
        policy's ``spawn_sources`` and the type's ``spawnable_by`` must both
        name ``"user"``) or a blank objective is one ``ERROR`` event, and
        nothing runs. The run's envelope is kept for ``instance_result``.

        Stopping the stream (Ctrl+C) is the user's stop of the run, as in
        ``run_instance``: it ends ``cancelled`` by the user and the instance
        is ``stopped``; a stop before the run started stops the run as it
        starts. ``interrupt_instance`` ends it as a turn's cancel does.

        A conversation that is not stored yet is stored right before the run
        starts (after every read that can fail, so a start that raised
        stores nothing), titled from the objective with no turns of the
        user's with the main agent (``_keep_conversation``), so what the user
        did here is found by ``/resume``, ``--continue`` and ``--resume``.

        Yields:
            The run's ``WorkflowEvent``s, its start first.

        Raises:
            Exception: The store's, when the conversation's sub-agent records,
                or an inheriting type's conversation, cannot be read or the
                conversation cannot be stored: nothing runs.
        """
        async with self._turn(session_id, user_id) as ref:
            refused = self._delegations.user_start_refused(agent_type, objective)
            if refused is not None:
                yield WorkflowEvent.error(refused)
                return
            config = next((c for c in self._agent_configs if c.name == agent_type), None)
            history = None
            if config is not None and config.inherit_context:
                copy = await self._inherited_history(ref)
                history = lambda: copy  # noqa: E731 - read once, before the run
            state = await self._conversation_state(ref)
            # Stored after every read that can fail, right before the run
            # starts: a start that raised lists no empty conversation.
            await self._keep_conversation(ref, objective)
            async with contextlib.aclosing(self._users_run(
                ref,
                lambda: self._delegations.start_by_user(
                    ref, agent_type, objective, state=state, history=history,
                ),
                name=f"user start of {agent_type}",
            )) as events:
                async for event in events:
                    yield event

    async def _users_run(
        self,
        ref: SessionRef,
        work: Callable[[], Awaitable[Delegated]],
        *,
        instance: str | None = None,
        name: str,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """A run the user starts in ``ref``'s turn, streamed (``run_instance``, ``start_instance``).

        ``work()`` starts it through ``Delegations`` and returns its
        envelope; it runs in a task of its own, so a stop of the stream
        cancels the run, not its caller: the run ends ``cancelled`` by the
        user. A run that has not started when the stream stops is stopped
        as it starts. An interrupted one (``interrupt_instance``, held under
        ``instance``, or the new instance's id once its start is reported)
        ends as a turn's cancel ends its runs. A refusal (no run) is one
        ``ERROR``. The envelope of a run that ended is kept (``_user_ran``).
        """
        live = _UsersRunEvents()
        held = _UserRun()
        key = (ref, instance) if instance is not None else None
        if key is not None:
            self._user_runs[key] = held

        def stop_it() -> None:
            # Its consumer stopped (Ctrl+C): the user's stop of the run
            # alone, unless the run is interrupted (a turn's cancel).
            if held.interrupted or live.started is None or live.started.run is None:
                return
            self._delegations.cancel(
                live.started.run, by=USER, error=STOPPED_BY_USER_ERROR, conversation=ref
            )

        async def source() -> AsyncGenerator[WorkflowEvent, None]:
            if held.interrupted:  # before its run started: none starts
                raise asyncio.CancelledError
            # The run's own task: a stop cancels the run, not its caller,
            # so it ends cancelled by the user, not interrupted.
            task = held.work = asyncio.create_task(work(), name=name)
            try:
                delegated = await asyncio.shield(task)
            except asyncio.CancelledError:
                if held.interrupted:
                    task.cancel()  # the run's caller is cancelled: it ends interrupted
                else:
                    stop_it()
                    if live.started is None:
                        # Stopped before its run started (its instance being
                        # numbered): the run that starts is stopped as it starts.
                        await live.until_started(task)
                        stop_it()
                await asyncio.wait([task])
                if not task.cancelled() and task.exception() is None:
                    ended = task.result()
                    if ended.run is not None:
                        self._user_ran(ref, ended.run.instance, ended)
                raise
            if delegated.run is None:
                yield WorkflowEvent.error(str(delegated.result.get("error", "")))
                return
            self._user_ran(ref, delegated.run.instance, delegated)

        try:
            with live_events.reporting(live):
                async with contextlib.aclosing(
                    live_events.merged(source(), live, on_stop=stop_it)
                ) as items:
                    async for item in items:
                        if key is None and live.started is not None:
                            # The new instance is named: interrupt_instance finds it.
                            key = (ref, live.started.instance)
                            self._user_runs[key] = held
                        event = self._apply_event_hook(item)
                        if event is not None:
                            yield event
        finally:
            if key is not None and self._user_runs.get(key) is held:
                del self._user_runs[key]

    def interrupt_instance(
        self, instance: str, *, session_id: str | None = None, user_id: str | None = None
    ) -> bool:
        """Interrupt the run of ``instance`` the user started, as a turn's cancel (provisional).

        What a process exit does to a run: not the user's stop. The run of
        that ``run_instance`` (or ``start_instance``, once its start named
        the instance) ends ``interrupted`` and its instance stays
        ``idle``, so its parent may continue it; its stream then raises
        ``asyncio.CancelledError``, and its turn records the run and writes
        the records as it ends (a stop of that stream from now on is no
        stop of the user's either). Only ``run_instance``'s and
        ``start_instance``'s run: a run of the root's turn ends with that turn.

        Returns:
            Whether such a run was working (or about to start); False,
            doing nothing, otherwise.
        """
        held = self._user_runs.get((self.session_ref(session_id, user_id), instance))
        if held is None or held.interrupted or (held.work is not None and held.work.done()):
            return False
        held.interrupted = True
        if held.work is not None:
            held.work.cancel()
        return True

    def _user_ran(self, ref: SessionRef, instance: str, delegated: Delegated) -> None:
        """Take what a run the user started left: its mark on the conversation, its envelope.

        The envelope is kept as the root reads it (``view_notice``): through
        the tool result cap, as every envelope a model reads is (a ``task``
        result's too), so one over ``tool_result_max_chars`` is cut as a
        dict, its note under ``truncated`` naming its full output, saved
        once, as the run ends, in the tool-output folder under the
        instance's name.
        """
        for value in delegated.state.values():
            if isinstance(value, str) and value:
                self._outside_marks.mark(ref, value)
        self._instance_results[(ref, instance)] = self._capped(instance, dict(delegated.result))

    def _capped(self, instance: str, result: dict[str, Any]) -> dict[str, Any]:
        """``result`` through the tool result cap, under ``instance``'s name; uncut if the cap fails."""
        try:
            capped = cap_tool_result(
                result,
                tool_name=instance,
                max_chars=self._settings.tool_result_max_chars,
                folder=tool_output_dir(self._settings.app_name),
            )
        except Exception as error:  # noqa: BLE001 - the cap never raises by design
            # A bug in the cap must not lose the envelope: it goes uncut, as
            # a tool result does when the backends' cap fails.
            logger.warning(
                "tool_result_cap_failed", tool=instance,
                error_type=type(error).__name__, error=str(error),
            )
            logger.debug("tool_result_cap_failed", tool=instance, exc_info=error)
            return result
        return capped if isinstance(capped, dict) else result

    def instance_result(
        self, instance: str, *, session_id: str | None = None, user_id: str | None = None
    ) -> dict[str, Any] | None:
        """The envelope of the latest run of ``instance`` that the user started (provisional).

        Kept in memory for the conversation (the records keep only the head
        of a result), as the tool result cap left it: one over
        ``tool_result_max_chars`` is cut, its note under ``truncated`` naming
        its full output. None when no run of the user's ended there, or
        after the conversation was deleted.
        """
        return self._instance_results.get((self.session_ref(session_id, user_id), instance))

    async def stop_instance(
        self, instance: str, *, session_id: str | None = None, user_id: str | None = None
    ) -> bool:
        """Stop the sub-agent ``instance`` for the user (provisional).

        A running instance's run (it works inside some turn, the root's or a
        ``run_instance``'s) is cancelled from outside, ``cancelled_by`` the
        user, and waited for: the instance is ``stopped`` and its records
        written when this returns, which takes no turn lock. An idle one is
        stopped at once. Only the user may start a stopped instance's next
        run (``run_instance``); its parent's ``message_agent`` is refused.

        Returns:
            False, doing nothing, for an instance that is unknown, closed or
            stopped already.

        Raises:
            Exception: The store's, when the conversation's sub-agent records
                cannot be read: nothing is stopped.
        """
        ref = self.session_ref(session_id, user_id)
        await self.load_agents(ref)
        stopped = await self._delegations.stop_by_user(ref, instance)
        await self._agent_records_written(ref)
        return stopped

    async def close_instance(
        self, instance: str, *, session_id: str | None = None, user_id: str | None = None
    ) -> list[str]:
        """Close the sub-agent ``instance`` and its own sub-agents for the user (provisional).

        A closed instance takes no follow-up; its work stays readable. A
        running one's run is stopped by the user first and the instance
        closes as it ends, which this waits for; the records are written when
        it returns, which takes no turn lock.

        Returns:
            The ids closed, ``instance`` first; none for an instance that is
            unknown or closed already.

        Raises:
            Exception: The store's, when the conversation's sub-agent records
                cannot be read: nothing is closed.
        """
        ref = self.session_ref(session_id, user_id)
        await self.load_agents(ref)
        closed = await self._delegations.close_by_user(ref, instance)
        await self._agent_records_written(ref)
        return closed

    def steer_instance(
        self,
        instance: str,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Give the user's ``message`` to ``instance``'s running run, before its next model step (provisional).

        Returns:
            False when the instance has no running run (unknown, idle,
            stopped, closed, or one that takes no more steps).

        Raises:
            ValueError: ``message`` is not text, or blank.
        """
        ref = self.session_ref(session_id, user_id)
        record = self.agent_runs.instance(ref, instance)
        if record is None or record.status != INSTANCE_RUNNING or not record.runs:
            return False
        steered = self._delegations.steer(ref, record.runs[-1], message)
        if steered:
            logger.info("subagent_steered", run=record.runs[-1], started_by=USER)
        return steered

    async def _conversation_state(self, ref: SessionRef) -> dict[str, Any]:
        """What a run the user starts in ``ref``'s conversation starts from (a backend hook).

        Its outside-content mark, under the key its backend's runs share it
        by (the state a ``task`` call hands its run); the marks are seeded
        from the conversation's own state first. The base has none.
        """
        return {}

    async def _keep_conversation(self, ref: SessionRef, objective: str) -> None:
        """Store ``ref``'s conversation when it is not yet, as a turn's start does (a backend hook).

        What a conversation whose only work is a run the user started
        (``start_instance``) needs to be one the session hooks find
        (``session_exists``, ``list_sessions``): its title is the objective's
        (``session_title``) and its turns none, since the user typed none to
        the main agent. A stored conversation keeps its title and turns. The
        base stores nothing.
        """

    async def _inherited_history(self, ref: SessionRef) -> list[Any]:
        """``ref``'s conversation as an inheriting run the user starts reads it (a backend hook).

        What the root's ``task`` would hand such a run (its ``history``),
        read from the store between turns: ADK the conversation's
        root-visible events (``adk/task_tool.py::inherited_events``), Deep
        Agents the root's messages with its summary applied
        (``deepagents/task_tool.py::inherited_messages``). Empty for a
        conversation with none yet. The base has none.
        """
        return []

    async def tell_root(
        self, text: str, *, session_id: str | None = None, user_id: str | None = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run the root agent on ``text``, a notice the user did not type (provisional).

        A turn of the conversation whose input is ``text`` as one user-role
        message, marked as a job's notice is: ``recent_messages`` (and so
        fact extraction) and the conversation's title and turns leave it out.
        The root reads it and may answer; the outside-content mark this
        manager holds for the conversation (what a run the user started
        brought) is written with it. A turn that fails yields one ``ERROR``.
        The notice after the user worked with a sub-agent is
        ``agents/envelope.py::view_notice``.

        Yields:
            The turn's ``WorkflowEvent``s.

        Raises:
            NotImplementedError: The backend cannot (it has no
                ``_deliver_notice``), raised as iteration starts, before any
                turn.
        """
        if type(self)._deliver_notice is BaseWorkflowManager._deliver_notice:
            raise NotImplementedError(
                f"{type(self).__name__} cannot tell its root agent a notice (tell_root)."
            )
        async with self._turn(session_id, user_id) as ref:
            async with contextlib.aclosing(self._deliver_notice(text, ref)) as events:
                async for event in events:
                    yield event

    async def _deliver_notice(
        self, text: str, ref: SessionRef
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run the root agent on ``text`` as one user-role message marked as a notice (a backend hook).

        The base calls it inside the turn of ``ref``'s conversation (the
        active turn, the services and the settings set). The message is
        marked so ``recent_messages`` and the conversation's row leave it
        out; the mark this manager holds for the conversation is written
        with it when the conversation's state lacks one; each event goes
        through ``_apply_event_hook``. Closing the generator stops the run.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot tell its root agent a notice.")
        if False:  # pragma: no cover - an async generator
            yield  # type: ignore[misc]

    async def _run_agent(self, run: AgentRun) -> RunAnswer:
        """Run one sub-agent run to its end: the backend's hook (``Delegations``'s).

        Called in the run's own task, where the run is the active run and no
        tool call is (``agents/delegate.py``). It runs ``run.config``'s agent
        on ``run.message`` (after ``run.history`` for an inheriting one), on
        its instance's own transcript: a first run starts it, a follow-up
        (``run.follow_up``) continues it, and the calls an earlier run left
        open there are answered before its message. The run starts from the
        shared state ``run.state`` (the outside-content mark: its
        plan and tasks start empty for an instance's first run; a follow-up
        continues its own), reports the run's events to the turn's
        live events as they are made, and returns its last answer, its
        outcome, the shared state it changed and its own last plan and tasks
        (``RunAnswer.plan``, ``RunAnswer.tasks``). A failed model call is
        answered (``RunAnswer.error``: the provider's reason,
        ``failure_reason``; ``RunAnswer.error_code``: its error's type, the
        warning's code) and a failure in the backend's own words raised
        (``RunFailed``), each with the last text the run wrote, which its
        envelope carries; any other exception it lets out fails the run with
        no result. A cancel stops the run; the
        run's own (``run.control.own_cancel()``: a limit's, its parent's,
        the user's) returns what it has, its last text and the shared state
        it changed, and any other (its turn's) raises. Once the run's work
        has ended, before anything it awaits to read what the run did, it
        calls ``run.control.finish()``, so a cancel of the run alone then
        loses nothing. ``run.control`` keeps its limits at its model calls
        (each backend's in its sub-agents' stacks). A run narrowed to some of
        its type's tools (``run.control.tools``) is offered only those and the
        tools every run keeps (``KEPT_TOOLS``), and a call of another is
        answered as one of a tool the agent does not have
        (``unknown_tool_result``). A run of the built-in ``general_purpose``
        type (``run.based_on``: the type it is made for) runs on the
        general-purpose agent built for that type (``Delegations.general_purpose``),
        on that type's own model (``_general_model``).

        Raises:
            NotImplementedError: This backend runs no sub-agent.
        """
        raise NotImplementedError(f"{type(self).__name__} runs no sub-agent")

    def _run_models(
        self, run: AgentRun
    ) -> "tuple[Mapping[str, ResolvedModel], Mapping[str, ContextBudget]]":
        """The agents' models and context numbers a sub-agent ``run`` reads, by agent
        name.

        Those of the manager's, except that a run on a model choice
        (``run.model``) has its type's entries the choice's: its backend's
        event processing, usage and overflow messages then name and count
        the model the run went to, under the type's name. A run of the
        built-in ``general_purpose`` type (``run.based_on``) has its entry
        the model and numbers of the type it is made for.

        Raises:
            KeyError: ``run.model`` is not one of its type's model choices.
        """
        if run.model is None and run.based_on is None:
            return self._resolved_models, self._budgets
        models = dict(self._resolved_models)
        budgets = dict(self._budgets)
        if run.based_on is not None:
            models[run.ref.agent] = self._general_model(run.based_on)
            if run.based_on in self._budgets:
                budgets[run.ref.agent] = self._budgets[run.based_on]
        if run.model is not None:
            key = (run.ref.agent, run.model)
            models[run.ref.agent] = self._choice_models[key]
            if key in self._choice_budgets:
                budgets[run.ref.agent] = self._choice_budgets[key]
        return models, budgets

    def _general_model(self, spawner: str) -> "ResolvedModel":
        """The model of the general-purpose agent made for the type ``spawner``.

        ``spawner``'s own resolved model (never a run's model choice), named
        for ``general_purpose``: a backend keys an agent's own reasoning and
        its usage by that name.
        """
        return dataclasses.replace(self._resolved_models[spawner], agent=GENERAL_PURPOSE)

    async def _agent_tool_names(self, agent: str) -> frozenset[str]:
        """The names a ``task`` call's ``tools`` may give for ``agent`` (a backend
        hook).

        Asked for a call that names tools and for a narrowed instance's
        follow-up: the agent's own tools by the names its model sees (a
        backend adds its skill and MCP tools, ADK its built-ins), the state
        and delegation tools (``KEPT_TOOLS``) left out. The base's are the
        agent's configured tools as ``_build_tools`` assembles them, each by
        its callable's name; an agent that is not configured has none.
        """
        config = next((c for c in self._agent_configs if c.name == agent), None)
        if config is None:
            return frozenset()
        return configured_tool_names(self._build_tools(config, {}))

    @property
    def settings(self) -> "BaseSettings":
        """Get the settings instance."""
        return self._settings

    @property
    def app_name(self) -> str:
        """Get the application name."""
        return self._app_name

    @property
    def is_initialized(self) -> bool:
        """Check if services have been initialized."""
        return self._initialized

    @property
    def catalog(self) -> ModelCatalog | None:
        """The model catalog this manager loaded; None until it initializes."""
        return self._catalog

    @property
    def required_managers(self) -> set[str]:
        """Get the set of required manager types detected from tools."""
        return self._required_managers

    @property
    def services(self) -> dict[str, Any]:
        """Get the service registry dict."""
        return self._services

    @property
    def _service_set(self) -> ServiceSet:
        """The services this manager holds (``workflow/services.py``).

        Made on first use too, so a manager built without the constructor
        (a test harness's ``__new__``) has one.
        """
        held = vars(self).get("_services_held")
        if held is None:
            held = self._services_held = ServiceSet()
        return held

    @property
    def _services(self) -> dict[str, Any]:
        """The published services: the registry a turn's tools read (``get_service``)."""
        return self._service_set.services

    @_services.setter
    def _services(self, services: dict[str, Any]) -> None:
        self._service_set.reset(services)

    @property
    def _carried_service_keys(self) -> frozenset[str]:
        """The keys of the services a reinitialize is carrying over (:meth:`_keeping`)."""
        return self._service_set.carried

    @_carried_service_keys.setter
    def _carried_service_keys(self, keys: frozenset[str]) -> None:
        self._service_set.carried = frozenset(keys)

    @property
    def _memory_index(self) -> MemoryIndex:
        """The memory index each conversation's main agent gets (``workflow/memory_index.py``).

        Made on first use too, so a manager built without the constructor (a
        test harness's ``__new__``) has one.
        """
        held = vars(self).get("_memory_index_held")
        if held is None:
            held = self._memory_index_held = self._new_memory_index()
        return held

    @property
    def _index_snapshots(self) -> dict[SessionRef, str]:
        """The memory index each conversation took, by conversation."""
        return self._memory_index.snapshots

    @_index_snapshots.setter
    def _index_snapshots(self, snapshots: dict[SessionRef, str]) -> None:
        self._memory_index.snapshots = snapshots

    @property
    def knowledge_base(self) -> "KnowledgeBase | None":
        """The knowledge base, over the project and user scopes (if required by tools)."""
        return self._services.get(KNOWLEDGE_BASE)

    @property
    def memory_store(self) -> "MemoryStore | None":
        """The memory store (if required by tools)."""
        return self._services.get(MEMORY_STORE)

    @property
    def llm_summarizer(self) -> Any | None:
        """Get the LLM summarizer (if required by tools)."""
        return self._services.get(LLM_SUMMARIZER)

    @property
    def sandbox_manager(self) -> "SandboxManager | None":
        """Get the sandbox manager (if required by tools)."""
        return self._services.get(SANDBOX_MANAGER)

    @property
    def job_manager(self):
        """Get the long-running-job manager (if required by tools)."""
        return self._services.get(JOB_MANAGER)

    # ------------------------------------------------------------------
    # Tool assembly
    # ------------------------------------------------------------------

    def _build_tools(
        self, config: "AgentConfig", service_map: dict[str, Callable] | None = None,
    ) -> list[Callable]:
        """Build the tool list for an agent config.

        Replaces service tools with closure-bound factory versions and
        auto-injects backend-specific state tools when requested.

        Entries are matched by *registry identity* — the exact object the
        default registry issued — never by ``__name__``. A plain callable an
        application happens to name ``kb_search`` is not the framework's tool:
        substituting the service-bound variant for it would silently run
        different code. It is therefore passed through untouched, as is a tool
        registered into some other ``ToolRegistry``. A callable passed through
        is denied at permission time, and named once in a
        ``tool_not_registered`` warning; an ADK built-in tool (such as
        ``GoogleSearchTool``) runs inside the model and never reaches the
        permission check at all.

        Conversely ``register(func, name=...)`` leaves the caller holding a
        callable whose ``__name__`` is the private implementation name; that
        one *is* bound, so it resolves to its service-bound variant, or is
        replaced by the canonical callable so the model sees the registered
        name.
        """
        from agentic_cli.tools.registry import get_registry, identify_tool

        if service_map is None:
            service_map = self._get_service_tool_map()

        registry = get_registry()
        result = []
        for tool in config.tools or []:
            definition = identify_tool(tool)
            if definition is None:
                self._name_unregistered(config.name, tool)
                result.append(tool)
                continue
            variant = service_map.get(definition.name)
            if variant is not None and identify_tool(variant) is definition:
                # The variant must *be* this tool, not merely share its name:
                # an application that took the name over (register(...,
                # replace=True)) would otherwise have the framework's
                # implementation run in place of its own.
                result.append(variant)
                continue
            # A renamed tool's (or a declared variant's) original callable:
            # hand the backend the canonical one so the model-visible name is
            # the tool's identity. Never None — see ``canonical_for``.
            result.append(registry.canonical_for(tool))

        if config.include_state_tools:
            result.extend(self._injectable_state_tools(result))

        return result

    def _name_unregistered(self, agent: str, tool: Any) -> None:
        """Warn that the gate will deny every call to ``tool``, which the
        registry did not issue (``_unregistered_name``). Once per agent and
        tool, however often the agent is built (a model switch builds it
        again); an object no check can read is let through unnamed."""
        try:
            name = _unregistered_name(tool)
        except Exception as e:
            logger.debug(
                "tool_registration_unchecked", agent=agent, error=f"{type(e).__name__}: {e}"
            )
            return
        if name is None or (agent, name) in self._unregistered_named:
            return
        self._unregistered_named.add((agent, name))
        logger.warning(
            "tool_not_registered",
            agent=agent,
            tool=name,
            hint="register it with @register_tool; every call to it is denied",
        )

    def _injectable_state_tools(self, assembled: list) -> list[Callable]:
        """State tools worth auto-injecting, given what the agent already has.

        A state tool the application has taken over (``replace=True``) leaves
        the backend's variant *retired* — no identity, no capabilities, and the
        replacement is what the agent should call. Injecting it anyway would
        hand the model two tools with the same name, one of them denied.

        An unregistered state tool that collides with nothing is left alone:
        a backend may legitimately supply its own.
        """
        from agentic_cli.tools.registry import get_registry

        registry = get_registry()
        present = {getattr(tool, "__name__", "") for tool in assembled}
        injectable = []
        for tool in self._get_state_tools():
            name = getattr(tool, "__name__", "")
            if name in present:
                logger.debug("state_tool_already_present", tool=name)
                continue
            if registry.is_retired(tool):
                logger.debug("state_tool_retired", tool=name)
                continue
            injectable.append(tool)
        return injectable

    def _get_service_tool_map(self) -> dict[str, Callable]:
        """Create service tools via factories, returning name→function map.

        Only creates tools for services that have been initialized, from
        the table (``ServiceSet.tool_map``); the interaction tools always.
        A backend that adds a tool of its own overrides this, and the
        backends build their agents from what it returns.
        """
        return self._service_set.tool_map(self)

    @abstractmethod
    def _get_state_tools(self) -> list[Callable]:
        """Return backend-specific state tools (plan/task management).

        Subclasses must override to return ADK or LangGraph native tools.
        """
        ...

    def _resolve_config_tool_refs(self) -> None:
        """Resolve string/dotted-path tool refs in configs to callables.

        Each ``config.tools`` entry may be a callable, a registered tool name,
        or a dotted import path. This rewrites every config's ``tools`` list in
        place so subsequent service-detection and tool assembly (which key on
        ``tool.__name__``) operate on real callables. Callables pass through
        unchanged, so the call is idempotent.
        """
        from agentic_cli.tools.tool_resolver import resolve_tools

        for config in self._agent_configs:
            if config.tools:
                config.tools = resolve_tools(config.tools)

    def _detect_required_managers(self) -> set[str]:
        """Detect which services the configured tools declared they need.

        Each tool declares its own dependencies via
        ``register_tool(..., requires=...)``, so an extension can ship a
        service-backed tool without editing the framework. Tools that are not
        registered (plain callables) declare nothing and need nothing.

        Resolution is by registry identity only: a renamed tool's original
        callable still declares its services, but a callable the default
        registry never issued declares nothing, however it is named. Building a
        knowledge base because an application named a function ``kb_search``
        would be work done for a tool that is denied at permission time.

        Returns:
            Set of required service keys (e.g. ``{"knowledge_base", "memory_store"}``).
        """
        from agentic_cli.tools.registry import identify_tool

        required: set[str] = set()
        for config in self._agent_configs:
            for tool in config.tools or []:
                definition = identify_tool(tool)
                if definition is not None:
                    required.update(definition.requires)
        return required

    def _ensure_managers_initialized(self) -> None:
        """Create and publish the services detected from tool metadata.

        A helper for tests and scripts that need the services without
        initializing: no initialization path calls it, so stubbing it changes
        nothing there (``initialize_services`` builds them through
        :meth:`_construct_services`, off the event loop and transactionally).
        It is synchronous and blocks on the constructors, so it must not be
        called on the event loop.
        """
        self._publish_services(self._build_services(frozenset(self._services)))

    def _publish_services(self, built: dict[str, Any]) -> None:
        """Adopt constructed services without displacing anything already live."""
        self._service_set.publish(built)

    def _build_services(
        self, existing: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Construct the required services into a **fresh** dict.

        Pure construction: it never writes to ``self._services``. Constructors
        here can load heavy dependencies (the sentence-transformers model inside
        ``EmbeddingService``), so this runs on a worker thread — and cancelling
        the coroutine that awaits it does not stop that thread. Writing results
        straight into the manager therefore published services *after* a
        rolled-back or cleaned-up initialization, leaking whatever the thread
        had built. The caller publishes, and only while it still owns the
        attempt (see :meth:`_construct_services`).

        Transactional in itself: if a later constructor raises, everything this
        call already built is released before the error propagates. Nothing has
        been published at that point, so nobody else could ever close it — an
        abandoned SandboxManager or JobManager would keep its pool alive for
        the life of the process.

        Args:
            existing: Service keys already published; those are not rebuilt.

        Returns:
            The newly constructed services, keyed by service key.

        Raises:
            Exception: Whatever a service constructor raised, after rollback.
        """
        s: dict[str, Any] = {}
        try:
            self._build_services_into(s, existing)
        except BaseException:
            self._close_services(s)
            raise
        return s

    def _build_services_into(
        self, s: dict[str, Any], existing: frozenset[str]
    ) -> None:
        """Construct the required services into ``s``. See :meth:`_build_services`.

        The table's services (``ServiceSet.build_into``): those the tools
        declared, and those every manager has. A backend that builds a
        service of its own overrides this, calls it, and adds its own to
        ``s`` (and its closer to ``_SYNC_SERVICE_CLOSERS``).
        """
        self._service_set.build_into(self, s, existing)

    async def _construct_services(self) -> None:
        """Build services off the event loop and publish them transactionally.

        The build runs on a worker thread that cancellation cannot interrupt,
        so the result is published only if this attempt is still the one that
        owns initialization. If the await is cancelled, whatever the thread
        goes on to build is *released* rather than published — otherwise a
        rolled-back initialization would leave a live sandbox or job manager
        behind that nothing would ever close.
        """
        build = asyncio.ensure_future(
            asyncio.to_thread(self._build_services, frozenset(self._services))
        )
        try:
            built = await asyncio.shield(build)
        except BaseException:
            build.add_done_callback(self._discard_built_services)
            raise
        self._publish_services(built)

    def _discard_built_services(self, build: "asyncio.Future[dict[str, Any]]") -> None:
        """Release services constructed for an attempt that no longer owns init.

        A done callback, on the event loop: the closes run on a worker thread
        (:meth:`_start_closing_services`).
        """
        if build.cancelled() or build.exception() is not None:
            return
        built = build.result()
        if not built:
            return
        logger.warning("services_discarded_after_rollback", services=sorted(built))
        self._start_closing_services(built)

    async def summarize(self, content: str, prompt: str) -> str:
        """Summarize content using the configured LLM.

        A prompt longer than the default model takes is cut in the middle
        (``_fit_prompt``): its callers cap what they send (web_fetch's page
        text, the knowledge base's content), and this is the backstop.

        Args:
            content: The text being summarized. Not used here: ``prompt``
                already holds it (every caller builds the prompt around it),
                so sending both would send the text twice. It is part of the
                summarizer protocol (the knowledge base's ``Summarizer``,
                web_fetch's ``LLMSummarizer``) for implementations that need
                the text apart from the prompt.
            prompt: The full summarization prompt, which is what is sent.
        Returns:
            Summarized text response.
        """
        with side_call(PURPOSE_TOOL):
            return await self.generate_simple(
                self._fit_prompt(prompt, 12000), max_tokens=12000
            )

    def _fit_prompt(self, prompt: str, max_tokens: int) -> str:
        """``prompt`` as a side call on the default model may send it.

        A side call that carries outside text (``summarize``: a fetched page,
        a knowledge base document; fact extraction: the session's last
        messages) could be longer than the model's window: a context
        overflow, or a very expensive call. One over the room the model
        leaves (``context.prompt_room``, with the ``max_tokens`` the call is
        resolved to; the 128k floor before initialization) keeps its start
        and its end, where the instructions are, and leaves out the middle.
        A room under ``MIN_PROMPT_ROOM`` holds no useful prompt: the prompt
        goes as it is, and the provider answers.
        """
        from agentic_cli.workflow.context import (
            MIN_PROMPT_ROOM,
            cut_middle,
            prompt_room,
        )

        model = None
        if self._default_model is not None:
            model = self._simple_model(max_tokens)
        room = prompt_room(model, max_tokens)
        if len(prompt) <= room or room < MIN_PROMPT_ROOM:
            return prompt
        logger.debug("side_call_prompt_cut", chars=len(prompt), room=room)
        return cut_middle(prompt, room)

    async def on_session_end(
        self,
        messages: list[dict] | None = None,
        *,
        session: "SessionRef | None" = None,
    ) -> list["Proposal"]:
        """Propose memories from a conversation that ends, for review in ``/memory inbox``.

        Runs with ``auto_extract_session_facts`` on and a memory store. It
        writes no memory: each candidate the model returns goes to
        ``MemoryStore.propose``, which files it in the inbox or drops it.
        Never raises. Override in downstream apps for custom session-end
        behavior.

        Args:
            messages: Recent messages from the session (optional).
            session: Which conversation to read when ``messages`` is omitted.
                Defaults to the turn still in context, else this manager's
                current session under ``settings.default_user`` — so a session
                belonging to another user is read as *that* user rather than
                silently coming back empty.

        Returns:
            The proposals filed (empty if disabled, without a memory store or
            messages, or when the call fails).
        """
        if not getattr(self._settings, "auto_extract_session_facts", False):
            return []
        store = self._services.get(MEMORY_STORE)
        if store is None:
            return []
        ref = session or get_active_turn() or self.session_ref()
        # When the caller doesn't supply messages, pull them from the live
        # backend session (same source/sid save_session uses) so the CLI can
        # invoke this with no arguments on exit.
        if messages is None:
            try:
                if self._is_default_user(ref.user_id):
                    # Compatible call for backends that predate the user_id
                    # parameter (see _user_scoped_kwargs).
                    messages = await self.recent_messages(ref.session_id)
                else:
                    messages = await self.recent_messages(ref.session_id, user_id=ref.user_id)
            except Exception:
                logger.debug("session_fact_extraction_extract_failed", exc_info=True)
                return []
        if not messages:
            return []

        from agentic_cli.memory import ALL_MEMORY_TOOLS, extraction_prompt, parse_candidates

        try:
            index = await asyncio.to_thread(store.index, tools=ALL_MEMORY_TOOLS)
            discarded = await asyncio.to_thread(store.discarded)
            prompt = extraction_prompt(messages[-20:], index, discarded)
            with side_call(PURPOSE_MEMORY_EXTRACTION, ref):
                answer = await self.generate_simple(
                    self._fit_prompt(prompt, 2000), max_tokens=2000
                )
        except Exception:
            logger.debug("session_fact_extraction_failed", exc_info=True)
            return []
        mark = self._outside_content_mark(ref)
        filed: list["Proposal"] = []
        for candidate in parse_candidates(answer):
            try:
                proposal = await asyncio.to_thread(
                    store.propose, candidate, conversation=ref.session_id, outside_content=mark
                )
            except Exception:
                logger.debug("memory_proposal_failed", exc_info=True)
                continue
            if proposal is not None:
                filed.append(proposal)
        return filed

    def _root_memory_tools(self, root: str | None = None) -> tuple[str, ...]:
        """The memory tools the main agent ``root`` has, by name, from its config."""
        from agentic_cli.memory import ALL_MEMORY_TOOLS
        from agentic_cli.tools.registry import identify_tool

        root = root or self._root_agent_name
        config = next((c for c in self._agent_configs if c.name == root), None)
        if config is None:
            return ()
        names = [
            definition.name
            for definition in (identify_tool(tool) for tool in config.tools or [])
            if definition is not None and definition.name in ALL_MEMORY_TOOLS
        ]
        return tuple(dict.fromkeys(names))

    def _memory_index_wanted(self, root: str) -> bool:
        """Whether ``root``'s requests carry the memory index (setting on, a memory tool)."""
        return bool(getattr(self._settings, "memory_index_enabled", True)) and bool(
            self._root_memory_tools(root)
        )

    def _new_memory_index(self) -> MemoryIndex:
        """A memory index over what this manager holds, read as it is when a snapshot is taken."""

        def room() -> tuple[int, ProviderKind] | None:
            budget = self._budgets.get(self._root_agent_name)
            model = self._resolved_models.get(self._root_agent_name)
            if budget is None or model is None:
                return None
            return budget.budget, model.record.kind

        return MemoryIndex(
            settings=lambda: self._settings,
            store=lambda: self._services.get(MEMORY_STORE),
            tools=lambda: self._root_memory_tools(),
            room=room,
            # Asked through the forward, so a subclass's cap is the one a snapshot takes.
            limit=lambda: self._memory_index_cap(),
        )

    def _memory_index_now(self) -> str:
        """The index the turn's conversation carries, taken the first time it is
        asked for; ``""`` for none. It reads the memory files: call it off the
        event loop."""
        return self._memory_index.now()

    def _memory_index_cached(self) -> str:
        """The turn's conversation's snapshot if one was taken, else ``""``; reads nothing."""
        return self._memory_index.cached()

    def _memory_index_cap(self) -> int:
        """How many bytes the memory index may take: ``memory_index_max_bytes``, or,
        on a root whose budget it would crowd, a tenth of that budget."""
        return self._memory_index.cap()

    def _forget_memory_index(self, ref: SessionRef | None = None) -> None:
        """Drop a conversation's memory index and roster snapshots (the turn's by default).

        Both open the main agent's requests, so they change only where the
        prompt prefix changes anyway: the next request takes new ones.
        """
        self._memory_index.forget(ref)
        self._rosters.forget(ref)

    @property
    def _rosters(self) -> Rosters:
        """The roster each conversation's main agent gets (``workflow/agents/roster.py``).

        Made on first use, so a manager built without its constructor has it.
        """
        held = vars(self).get("_rosters_held")
        if held is None:
            held = self._rosters_held = Rosters(lambda: self.agent_runs)
        return held

    def _roster_wanted(self, root: str) -> bool:
        """Whether ``root``'s requests carry the roster of its sub-agents.

        Only a root that can continue them does (it has ``message_agent``: it
        can have sub-agents, its own or types started at run time, and the
        policy's ``follow_ups``); without the tool it could do nothing with an
        open instance.
        """
        config = next((c for c in self._agent_configs if c.name == root), None)
        return (
            config is not None
            and self.subagent_policy.follow_ups
            and can_have_children(config, self._agent_configs, self.subagent_policy, root=True)
        )

    def _roster_now(self) -> str:
        """The roster the turn's conversation carries, taken the first time it is
        asked for; ``""`` for none. It reads only the records in memory."""
        return self._rosters.now()

    def _outside_content_mark(self, ref: SessionRef) -> str | None:
        """The tool that brought outside content into ``ref``'s conversation, or None."""
        return self._outside_marks.get(ref)

    def _per_call_capabilities(self, settings: "BaseSettings") -> dict[str, str]:
        """``memory.write`` with its note while the turn's conversation is marked."""
        tool = self._outside_marks.get(get_active_turn())
        return {"memory.write": outside_content_note(tool)} if tool else {}

    @property
    def active_turn(self) -> SessionRef | None:
        """Identity of the turn running in this context, or None when idle.

        Context-local, not manager-local: concurrent ``process()`` calls on one
        manager (possible for framework consumers — the CLI serializes turns)
        each see their own value.
        """
        return get_active_turn()

    @property
    def active_session_id(self) -> str | None:
        """Session id of the in-flight ``process()`` call, or None when idle."""
        ref = get_active_turn()
        return ref.session_id if ref else None

    @property
    def active_user_id(self) -> str | None:
        """User id of the in-flight ``process()`` call, or None when idle."""
        ref = get_active_turn()
        return ref.user_id if ref else None

    # ---- Job resume: the base's template over the backend's hooks ----

    @property
    def supports_job_resume(self) -> bool:
        """Whether this backend can tell a conversation that a job of it finished.

        Derived, as ``supports_compact`` is, so it cannot drift from the
        code: True when the subclass implements :meth:`_deliver_job_notice`
        (the base's ``resume_with_job_result`` runs it) or overrides
        ``resume_with_job_result`` itself, as a backend written before the
        template does.
        """
        cls = type(self)
        return (
            cls._deliver_job_notice is not BaseWorkflowManager._deliver_job_notice
            or cls.resume_with_job_result is not BaseWorkflowManager.resume_with_job_result
        )

    @property
    def _answers_job_calls(self) -> bool:
        """Whether a finished job answers its pending call, so its record must name the call.

        False: the root agent is told in a message of its own, which needs
        only the job's conversation. A backend whose job answers the call
        instead says True, and a record without ``call_id`` cannot be
        resumed there.
        """
        return False

    def _record_resumable(self, record: Any) -> bool:
        """Whether ``record`` names what a resume needs: its conversation
        (session and user), and its call where a job answers it
        (:attr:`_answers_job_calls`)."""
        if not (record.session_id and record.user_id):
            return False
        return bool(getattr(record, "call_id", None)) or not self._answers_job_calls

    async def can_resume(self, record: Any) -> bool:
        """Whether a finished job can be resumed into its conversation now.

        The checks ``resume_with_job_result`` makes, without its turn: the
        backend can resume (``supports_job_resume``), the record names its
        conversation (session and user, and its call where a job answers
        it), and the conversation is in the backend's
        store (:meth:`_conversation_exists`). The CLI asks it to resume a
        job or to say "finished while its conversation was unavailable", and
        whether a job's conversation is gone.

        Sessions are durable by default (``session_store='sqlite'``), so a
        conversation normally survives a CLI restart and stays resumable. It is
        unavailable when, for example, it was deleted, the record is missing the
        identifiers the backend needs, or the run used the explicitly
        ephemeral ``session_store='memory'`` and the process restarted.
        """
        if not self.supports_job_resume or not self._record_resumable(record):
            return False
        return await self._conversation_exists(
            self.session_ref(record.session_id, record.user_id)
        )

    async def resume_with_job_result(
        self, record: Any, result: Any = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Tell the root agent a long-running job finished, in a turn of its own.

        One template for every backend. The record must name its
        conversation (its session and user, and its call where the job
        answers it); then, as one turn of that conversation (:meth:`_turn`:
        after any turn running, never against a released backend), the
        conversation must be there (:meth:`_conversation_exists`: telling one
        that is not would start it, and it never saw the job). The job's
        result, when not given, is read from this manager's job manager on a
        worker thread: it takes the jobs directory's cross-process lock,
        which on the event loop froze the app while another process held
        it. The notice is ``tools/jobs/resume.py::job_notice`` (the header,
        what the agent is asked, the payload), and the backend tells it
        (:meth:`_deliver_job_notice`), whose events this yields; a turn that
        fails yields one ``ERROR``, as ``process`` documents. A record that
        names too little, or a conversation that is not there, runs nothing
        and yields nothing (logged).

        Args:
            record: The finished ``JobRecord``.
            result: The job's result; read from the job manager if omitted.

        Yields:
            The turn's ``WorkflowEvent``s.

        Raises:
            NotImplementedError: The backend cannot (``supports_job_resume``),
                raised as iteration starts, before any turn.
        """
        if not self.supports_job_resume:
            raise NotImplementedError(
                f"{type(self).__name__} cannot tell a conversation that its job "
                "finished (resume_with_job_result). Check ``supports_job_resume`` "
                "before calling."
            )
        if not self._record_resumable(record):
            logger.warning(
                "job_resume_missing_ids",
                job_id=getattr(record, "job_id", None),
                have_session=bool(record.session_id),
                have_user=bool(record.user_id),
                have_call_id=bool(getattr(record, "call_id", None)),
            )
            return
        from agentic_cli.tools.jobs.resume import job_notice

        async with self._turn(record.session_id, record.user_id) as ref:
            if not await self._conversation_exists(ref):
                logger.warning(
                    "job_resume_session_missing",
                    job_id=record.job_id,
                    session_id=ref.session_id,
                )
                return
            if result is None:
                jobs = self._services.get(JOB_MANAGER)
                if jobs is not None:
                    result = await asyncio.to_thread(jobs.result, record.job_id)
            logger.info(
                "job_resume_started",
                job_id=record.job_id,
                call_id=getattr(record, "call_id", None),
                state=record.state.value,
            )
            # aclosing: a consumer that closes this generator closes the
            # backend's run now, inside the turn.
            async with contextlib.aclosing(self._deliver_job_notice(
                job_notice(record, result), record, result, ref
            )) as events:
                async for event in events:
                    yield event

    async def _conversation_exists(self, ref: SessionRef) -> bool:
        """Whether ``ref``'s conversation is in this backend's store now.

        A backend hook of the job resume (``resume_with_job_result`` asks it
        inside the turn, before it tells the conversation anything). It
        answers False before initialization and after a release, and never
        creates the conversation. The base knows of no conversation, so it
        answers False: a backend that implements :meth:`_deliver_job_notice`
        implements this too.
        """
        return False

    async def _deliver_job_notice(
        self, notice: str, record: Any, result: Any, ref: SessionRef
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Tell the root agent ``notice`` as one user-role message, and yield the turn's events.

        A backend hook of the job resume: the base calls it inside the turn
        of ``ref``'s conversation (the active turn, the services and the
        settings set, the conversation there), and implementing it is what
        makes ``supports_job_resume`` True. The backend marks the
        conversation as having read outside content (a job's output is), runs
        its root agent on the notice and yields what it yields, each event
        through ``_apply_event_hook``. Closing the generator stops the run.

        Args:
            notice: What the root agent is told
                (``tools/jobs/resume.py::job_notice``).
            record: The finished ``JobRecord`` (its ids, for a backend that
                marks the message with them).
            result: The job's result as read, or None: what a backend whose
                job answers its call instead sends
                (``job_result_payload(record, result)``).
            ref: The conversation, the turn's.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot tell a conversation that its job finished."
        )
        if False:  # pragma: no cover - an async generator
            yield  # type: ignore[misc]

    @contextlib.contextmanager
    def _workflow_context(
        self, session_id: str | None = None, user_id: str | None = None
    ) -> Iterator[None]:
        """Context manager that exposes the service registry to tools.

        Sets ContextVars (settings, the service registry, the active turn and
        the turn's span) so tools can call ``get_service(key)`` during
        execution and the JobManager can associate a launched job with the
        conversation that started it.

        All four are restored from tokens on exit, so a nested context
        restores the outer turn rather than clearing it, and concurrent turns
        on one manager never see each other's identity. The span ends as the
        block does (``workflow/sessions.py::close_turn_scope``): a task the
        turn started that outlives it can ask no question
        (:meth:`request_user_input`).
        """
        from agentic_cli.config import set_context_settings

        settings_token = set_context_settings(self._settings)
        registry_token = set_service_registry(self._services)
        turn_token = set_active_turn(self.session_ref(session_id, user_id))
        scope_token = open_turn_scope()
        try:
            yield
        finally:
            close_turn_scope(scope_token)
            reset_active_turn(turn_token)
            registry_token.var.reset(registry_token)
            settings_token.var.reset(settings_token)

    async def _aclose_owned(self, resource: Any, label: str) -> None:
        """Close one resource this manager owns; awaits an async close.

        The close contract is duck-typed on ``aclose()``/``close()`` (ADK's
        ``DatabaseSessionService`` exposes an async ``close()``; the in-memory
        one exposes none) and must be idempotent: callers null out their
        reference first, so a second cleanup passes ``None`` and does nothing.
        Never raises — a failing close must not block shutdown.

        Args:
            resource: The owned resource, or None.
            label: Name used in the failure log.
        """
        if resource is None:
            return
        closer = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if closer is None:
            return
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail
            logger.warning("resource_close_failed", resource=label, error=str(exc))

    # Services that own OS resources, and the sync method that releases
    # them: the table's (workflow/services.py). A backend that builds a
    # service of its own adds its closer here.
    _SYNC_SERVICE_CLOSERS: tuple[tuple[str, str], ...] = SERVICE_CLOSERS

    @classmethod
    def _close_services(cls, services: dict[str, Any]) -> None:
        """Release the owned resources in a service mapping. Never raises.

        Each closer is isolated — one raising must not leave the rest open.
        Used both for the live registry and for services a rolled-back
        initialization constructed but never published.
        """
        close_services(services, cls._SYNC_SERVICE_CLOSERS)

    def _cleanup_managers(self) -> None:
        """Release the synchronous resources this manager owns.

        Only services this manager created are released (see
        ``_build_services``). Idempotent: the registry is emptied, so a second
        call finds nothing. The registry is cleared regardless of failures.

        Services a reinitialize is carrying over (:meth:`_keeping`) stay
        published and open. The release's services hook
        (:meth:`_release_resources`): inside a release their closes start on
        a worker thread, and the release waits for them last; outside one
        (LangGraph's ``cleanup``, a script) they close here and now. A backend
        that releases something of its own overrides it and calls it.
        """
        services = self._service_set.release()
        closing = self._services_closing
        if closing is None:
            self._close_services(services)
            return
        started = self._start_closing_services(services)
        if started is not None:
            closing.append(started)

    def _start_closing_services(
        self, services: dict[str, Any]
    ) -> "asyncio.Future[None] | None":
        """Start closing ``services`` on a worker thread now; what to await, or None.

        Their closers block (a container removed, a kernel shut down, the
        jobs directory's lock waited for), so they never run on the event
        loop, where they froze the app. Started at once, they run to their
        end whatever becomes of the caller: a thread cannot be cancelled, and
        the loop's default executor is joined when the loop shuts down. A
        loop whose executor is already shut down closes them here.
        """
        if not services:
            return None
        close = functools.partial(
            contextvars.copy_context().run, self._close_services, services
        )
        try:
            return asyncio.get_running_loop().run_in_executor(None, close)
        except RuntimeError:  # no loop, or its executor is shut down
            close()
            return None

    @contextlib.contextmanager
    def _keeping(self, preserve: bool) -> Iterator[None]:
        """Keep the conversation's resources through a reinitialize when ``preserve`` is set.

        The session store (the backend's ``_detach_store``), session grants
        (the permission engine), sandbox kernels, the job manager and loaded
        embedding models belong to the conversation, not to the model, so a
        model switch that preserves sessions must not release them. Inside
        this block a release, including the rollback of a failed attempt,
        leaves the store attached and the live services published, and
        initialization reuses them: the backend builds on the store it holds,
        and only the services that are missing are built. After a failed
        attempt they stay with the manager, for a retry to reuse or
        ``cleanup()`` to release. The one mechanism :meth:`reinitialize` keeps
        them by: nothing has to put them back afterwards.
        """
        self._carried_service_keys = (
            frozenset(self._services) if preserve else frozenset()
        )
        self._keeping_store = preserve
        try:
            yield
        finally:
            self._carried_service_keys = frozenset()
            self._keeping_store = False

    def _carrying_services(self, carry: bool) -> contextlib.AbstractContextManager[None]:
        """:meth:`_keeping` under its earlier name (LangGraph's ``reinitialize`` uses it)."""
        return self._keeping(carry)

    def lend_services(self) -> ServiceLoan:
        """Lend this manager's services to another manager, of either backend.

        The services it owns, but its own ``WORKFLOW`` and summarizer
        entries, which are the manager itself. The other manager borrows
        them as it initializes (``initialize_services(services=loan)``) and
        builds only what they lack; :meth:`give_services` then makes it
        their owner. Until then this manager owns them, keeps working with
        them and closes them when it releases: a borrower that fails, is
        cancelled or is cleaned up closes none of them, and runs no turn.
        """
        return ServiceLoan(self, self._service_set.lendable(self))

    async def give_services(self, loan: ServiceLoan) -> None:
        """Give the services this manager lent to the manager that borrowed them.

        Takes this manager's lifecycle lock and then its turn lock (so it
        waits for a running turn), then the borrower's lifecycle lock (so no
        initialization of it runs meanwhile). The borrower owns them from
        here on and closes them when it releases; this manager's release
        drops them unclosed. What each holds of its owner moves to the
        borrower (``ServiceSpec.rebind``): the permission engine asks through
        it and takes its per-call say, the knowledge base summarizes through
        it. A service the borrower already had its own of stays this
        manager's. This manager runs no turn after it (``_turn_admission``
        refuses, a turn queued on its turn lock too) until it is released.

        Raises:
            ValueError: This manager did not lend them.
            RuntimeError: It was called inside a turn of this manager (it
                would wait for that turn, forever); they were given already;
                no manager borrowed them (none initialized with them, or its
                initialization failed or was cancelled, or it was released
                since); or this manager released them while they were lent.
        """
        if loan.lender is not self:
            raise ValueError(f"{type(self).__name__} did not lend these services.")
        if get_service(WORKFLOW) is self:
            # The turn's registry is this manager's: the give would wait for
            # the turn lock the calling turn holds.
            raise RuntimeError(
                f"give_services was called inside a turn of {type(self).__name__}: "
                "it waits for this manager's running turn. Call it between turns."
            )
        async with self._lifecycle_lock:
            async with self._turn_lock:
                if loan.given:
                    raise RuntimeError("These services were given already.")
                borrower = loan.borrower
                if borrower is None:
                    raise RuntimeError(
                        "No manager has borrowed these services: initialize one "
                        "with initialize_services(services=loan) first."
                    )
                async with borrower._lifecycle_lock:
                    moving = borrower._service_set.borrowed(loan.services)
                    if loan.services and not moving:
                        raise RuntimeError(
                            f"{type(borrower).__name__} holds none of the services it "
                            "borrowed: its initialization failed or was cancelled, or "
                            "it was released."
                        )
                    lost = sorted(
                        key for key, s in moving.items() if not self._service_set.owns(key, s)
                    )
                    if lost:
                        raise RuntimeError(
                            f"{type(self).__name__} released {', '.join(lost)} while "
                            "they were lent: they are closed."
                        )
                    self._service_set.give(moving)
                    borrower._service_set.adopt(moving)
                    rebind_services(moving, self, borrower)
                    loan.given = True
        logger.info(
            "services_given", services=sorted(moving), to=borrower.backend_type
        )

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return the backend type identifier (e.g. 'adk', 'langgraph')."""
        ...

    def _apply_event_hook(self, event: WorkflowEvent) -> WorkflowEvent | None:
        """Apply the optional on_event transformation hook.

        Returns the (possibly transformed) event, or None if suppressed.
        """
        if self._on_event:
            return self._on_event(event)
        return event

    @property
    def model(self) -> str:
        """The default model, as a ``provider:model`` reference.

        After initialization, the model the ``default`` role resolved to.
        Before it, the model the settings (or this manager's override) name,
        computed without the network when no setting names one.

        Raises:
            ModelResolutionError: Before initialization, if no setting names a
                default model and no provider with a key can supply one.
            ModelSpecError: If the override or a role setting is malformed.
        """
        if self._default_model is not None:
            return str(self._default_model.ref)
        return str(self._default_ref_without_listings())

    @property
    def agent_models(self) -> Mapping[str, ResolvedModel]:
        """Each agent's model, by agent name; empty before initialization."""
        return MappingProxyType(self._resolved_models)

    @property
    def choice_models(self) -> Mapping[tuple[str, str], ResolvedModel]:
        """Each agent's model choices' models, by ``(agent, choice)``, the choice
        as ``AgentConfig.model_choices`` writes it; empty before initialization."""
        return MappingProxyType(self._choice_models)

    @property
    def root_agent_name(self) -> str | None:
        """The root agent's name, from the validated agent graph.

        None before initialization, and on a manager that validates no graph
        of its own (LangGraph).
        """
        return self._root_agent_name if self._initialized else None

    def context_budget(
        self, agent_name: str | None = None, *, model: str | None = None
    ) -> "ContextBudget":
        """An agent's context numbers, as its backend summarizes and trims by them.

        Its window (or the 128k floor), input budget, summary trigger and the
        tail a summary keeps (``workflow/context.py``), computed when the
        models were resolved, before any service was built: the numbers
        ``/status`` shows.

        Args:
            agent_name: The agent; None for the root agent (``root_agent_name``).
            model: One of the agent's model choices, as ``AgentConfig.model_choices``
                writes it: its numbers instead of the agent's own model's.

        Raises:
            RuntimeError: Before initialization, or after cleanup.
            KeyError: ``agent_name`` names no agent whose context this manager
                manages: one whose model did not resolve (a native ADK YAML
                app's), or a ``model`` that is none of its choices.
                LangGraph manages none: every name is unknown there, the
                root's included.
        """
        if not self._initialized:
            raise RuntimeError(
                f"{type(self).__name__} is not initialized: its agents' context "
                "budgets are computed when it initializes."
            )
        name = self.root_agent_name if agent_name is None else agent_name
        if model is not None:
            if (name, model) not in self._choice_budgets:
                raise KeyError(
                    f"No context budget for agent {name!r}'s model choice {model!r}: "
                    "this manager does not manage it."
                )
            return self._choice_budgets[(name, model)]
        if name is None or name not in self._budgets:
            who = "the root agent" if name is None else f"agent {name!r}"
            raise KeyError(f"No context budget for {who}: this manager does not manage it.")
        return self._budgets[name]

    @property
    def supports_compact(self) -> bool:
        """Whether this backend can summarize a conversation on demand (``compact``).

        Derived, as ``supports_sessions`` is, so it cannot drift from the
        code: True when the subclass implements :meth:`_compact_turn` (the
        base's ``compact`` runs it) or overrides ``compact`` itself. A backend
        that cannot in some configuration narrows it (ADK, for a main agent
        whose model has no context budget).
        """
        cls = type(self)
        return (
            cls._compact_turn is not BaseWorkflowManager._compact_turn
            or cls.compact is not BaseWorkflowManager.compact
        )

    async def compact(
        self, session: SessionRef, focus: str | None = None
    ) -> "CompactResult | None":
        """Summarize the main agent's conversation in ``session`` now (``/compact``).

        A template: it runs as one turn of ``session`` (:meth:`_turn`: under
        turn admission, so it never interleaves with a turn or a job's
        resume, and inside the conversation's workflow context), in which the
        backend's :meth:`_compact_turn` summarizes as its conversations are
        summarized when they grow, but whatever the prompt's size, keeping
        the same tail word for word; ``focus`` tells the summary what to keep
        above all. What it reports goes to the caller's live events
        (``workflow/live_events.py``), when there are any:
        ``CONTEXT_SUMMARIZING``, the summary's ``LLM_USAGE``
        (``purpose="summarization"``) and ``CONTEXT_TRIMMED``
        (``source="compact"``). A cancel writes nothing.

        Args:
            session: The conversation.
            focus: What the summary should keep above all, if anything.

        Returns:
            What the summary did; None when there is nothing to summarize (no
            conversation yet, or none before the tail kept word for word).

        Raises:
            NotImplementedError: The backend cannot (``supports_compact``),
                raised before any turn.
            Exception: Why the summary could not be written (the provider's
                error).
        """
        if not self.supports_compact:
            raise self._no_compact_support()
        async with self._turn(session.session_id, session.user_id):
            return await self._compact_turn(session, focus)

    async def _compact_turn(
        self, session: SessionRef, focus: str | None
    ) -> "CompactResult | None":
        """Summarize ``session``'s main-agent conversation now, inside its turn.

        The backend hook of ``compact``: the base calls it inside the
        conversation's turn, and implementing it is what makes
        ``supports_compact`` True. It returns and raises what ``compact``
        documents, and writes nothing until the summary is written.

        Args:
            session: The conversation, as the caller named it.
            focus: What the summary should keep above all, if anything.
        """
        raise self._no_compact_support()

    def _no_compact_support(self) -> NotImplementedError:
        """The error for ``compact`` on a backend that cannot (``supports_compact``)."""
        return NotImplementedError(
            f"{type(self).__name__} cannot summarize a conversation on demand "
            "(compact). Check ``supports_compact`` before calling."
        )

    async def initialize_services(
        self, validate: bool = True, *, services: ServiceLoan | None = None
    ) -> None:
        """Initialize backend services asynchronously.

        Concurrency-safe, idempotent and transactional: concurrent callers
        serialize on the lifecycle lock, late arrivals see ``_initialized`` and
        return, and a failed attempt releases whatever it had allocated instead
        of leaving the manager half-built.

        Args:
            validate: If True, validate settings before initialization.
            services: Another manager's services to start with
                (:meth:`lend_services`): published for the tools as that
                manager's, so only what they lack is built, and closed by no
                release of this one (a failed or cancelled attempt's rollback
                included) until the lender gives them over
                (:meth:`give_services`).
        Raises:
            SettingsValidationError: If settings validation fails.
            ValueError: ``services`` are this manager's own.
            RuntimeError: ``services`` were given already, or this manager
                is initialized (it borrows only as it builds its services).
        """
        if services is not None:
            if services.lender is self:
                raise ValueError(f"{type(self).__name__} cannot borrow its own services.")
            if services.given:
                raise RuntimeError("These services were given already: lend them again.")
        async with self._lifecycle_lock:
            if services is None:
                await self._initialize_locked(validate=validate)
                return
            if self._initialized:
                raise RuntimeError(
                    f"{type(self).__name__} is initialized: it borrows services "
                    "only as it builds its own (initialize_services(services=...) "
                    "before anything else initializes it)."
                )
            await self._initialize_locked(validate=validate, services=services)

    async def _initialize_locked(
        self, validate: bool = True, services: ServiceLoan | None = None
    ) -> None:
        """Initialization body. The caller must hold ``_lifecycle_lock``.

        Args:
            validate: Check the model settings before anything is allocated
                (``config.validate_settings``). Model resolution itself always
                runs: the backend builds its agents from it.
            services: Another manager's services to borrow before the
                services are built (:meth:`initialize_services`).
        """
        if self._initialized:
            return

        # Validate the declared agent graph, and the model settings, before
        # anything is allocated or any network call is made: a static
        # configuration error should not cost a model listing or an
        # embedding model.
        graph = self._validate_agent_graph()
        self._root_agent_name = graph.root_name if graph is not None else None
        if validate:
            self._validate_model_settings()

        self._settings.export_api_keys_to_env()

        try:
            # Load the catalog, then resolve every agent's model against it,
            # before services are built: a model that cannot run fails startup
            # with nothing else allocated.
            await self._load_models()
            # Each agent's context numbers need only those models: a model too
            # small to work in (ContextBudgetError) fails here too, before a
            # service or a session store is built.
            self._budgets = self._compute_budgets()
            self._choice_budgets = self._compute_choice_budgets()
            if services is not None:
                # Another manager's, published as its: only what they lack
                # is built, and the rollback below drops them unclosed.
                self._service_set.borrow(services.services)
                services.borrower = self
            # Create services BEFORE backend init so _build_tools() can
            # produce factory-bound tools during agent/graph creation.
            # Construction is offloaded to a worker thread (heavy constructors)
            # and published transactionally — see _construct_services.
            await self._construct_services()
            await self._do_initialize()
        except BaseException:
            # Roll back: the catalog, services (and any backend resource the
            # partial _do_initialize created) must not outlive the attempt.
            await self._release_resources()
            raise
        self._initialized = True
        # The providers in use the start did not list (nothing cached, or an
        # older or failed listing) are listed now, in the background: that is
        # how the models the settings name are checked. The catalog swaps each
        # result in as it arrives, and warns of a model a provider does not serve.
        self._catalog.start_background_refresh()

    # --- models ------------------------------------------------------------

    def _manages_context(self) -> bool:
        """Whether this backend summarizes and trims its agents' conversations by context budgets.

        A backend hook of the initialization (:meth:`_compute_budgets`). True
        gives every agent a budget (``context_budget``) once the models are
        resolved, and a model whose window is too small to work in fails the
        start before any service is built. The base says no: a backend that
        keeps its own history (LangGraph) has none.
        """
        return False

    def _compute_budgets(self) -> "dict[str, ContextBudget]":
        """Each agent's context numbers, by name; none for a backend that manages none.

        Called by ``_initialize_locked`` with the models resolved and nothing
        else built, so the failure it raises leaves nothing to release but the
        catalog. A backend that computes its own overrides this rather than
        assigning ``_budgets`` in ``_do_initialize``. ADK and Deep Agents take
        every number they summarize and trim by from ``_budgets``, so an
        override of it (in a subclass of either) sets them all.

        Raises:
            ContextBudgetError: An agent's model leaves too small an input
                budget to work in; the message names the agent and the model.
        """
        if not self._manages_context():
            return {}
        from agentic_cli.workflow.context import context_budget

        return {
            name: context_budget(model, self._settings)
            for name, model in self._resolved_models.items()
        }

    def _compute_choice_budgets(self) -> "dict[tuple[str, str], ContextBudget]":
        """Each model choice's context numbers, by ``(agent, choice)``.

        None for a backend that manages none.

        Called right after ``_compute_budgets``, with the same failure: a
        choice whose model leaves too small an input budget to work in.

        Raises:
            ContextBudgetError: Naming the agent and the model.
        """
        if not self._manages_context():
            return {}
        from agentic_cli.workflow.context import context_budget

        return {
            key: context_budget(model, self._settings)
            for key, model in self._choice_models.items()
        }

    def _validate_model_settings(self) -> None:
        """The static model checks (``validate_settings``), with this manager's override.

        Raises:
            SettingsValidationError: Listing every problem.
        """
        from agentic_cli.config import SettingsValidationError, _model_settings_problems

        errors = _model_settings_problems(
            self._settings, self._agent_configs, manager_model=self._model_override
        )
        if errors:
            raise SettingsValidationError("\n".join(errors))

    async def _load_models(self) -> None:
        """Build and load the model catalog, then resolve every model.

        The catalog is owned from here on: ``_release_resources`` closes it.
        Loading waits only on the listings that fill a gap the settings leave
        (no entry can run, a computed role has none of its tier, or nothing
        describes a model they name); every model the settings name is then
        resolved as they name it, and the background refresh
        (``_initialize_locked``) checks the rest, warning of a model a
        provider does not serve.
        """
        from agentic_cli.workflow import model_services

        providers = model_services.providers_from_settings(self._settings)
        self._catalog = model_services.build_catalog(
            self._settings,
            providers,
            agent_configs=self._agent_configs,
            manager_model=self._model_override,
        )
        await self._catalog.load()
        self._resolve_models(providers)

    def _resolve_models(self, providers: Mapping[str, Provider]) -> None:
        """Resolve the default model and every agent's, all or nothing.

        Every agent is resolved before any backend object is built, and every
        problem is reported at once.

        Raises:
            SettingsValidationError: Listing each agent (or the default model)
                that cannot run, with the model and the allowed values; a
                problem several share (a role nothing can fill) once.
        """
        from agentic_cli.config import SettingsValidationError
        from agentic_cli.workflow import model_services

        try:
            roles = model_services.role_map_from_settings(self._settings, self._model_override)
        except ModelSpecError as exc:
            raise SettingsValidationError(str(exc)) from None

        errors: list[str] = []
        resolved: dict[str, ResolvedModel] = {}
        default = None
        for config in (None, *self._agent_configs):
            try:
                model = self._resolve_for(config, roles, self._catalog, providers)
            except ModelResolutionError as exc:
                errors.append(str(exc))
                continue
            if config is None:
                default = model
            else:
                resolved[config.name] = model
        # A choice runs as the agent's model would: its name, its settings.
        choices: dict[tuple[str, str], ResolvedModel] = {}
        for config in self._agent_configs:
            for choice in config.model_choices:
                try:
                    choices[(config.name, choice)] = self._resolve_for(
                        dataclasses.replace(config, model=choice),
                        roles, self._catalog, providers
                    )
                except ModelResolutionError as exc:
                    errors.append(str(exc))
        if errors:
            raise SettingsValidationError("\n".join(dict.fromkeys(errors)))
        resolved.update(self._resolve_extra_models(roles, providers))

        self._default_model, self._resolved_models = default, resolved
        self._choice_models = choices
        logger.info(
            "models_resolved",
            default=str(default.ref),
            agents={name: str(model.ref) for name, model in resolved.items()},
            choices={
                f"{name}/{choice}": str(model.ref)
                for (name, choice), model in choices.items()
            },
        )

    def _resolve_extra_models(
        self, roles: RoleMap, providers: Mapping[str, Provider]
    ) -> dict[str, ResolvedModel]:
        """The models of agents a backend built without configs, by name (none here).

        A backend hook of ``_resolve_models``, after every config's model
        resolved: ADK's for a native ADK YAML app's agents. Leniently: an
        agent left out has no model and no context budget. ``_catalog`` is
        loaded by then.
        """
        return {}

    def _resolve_for(
        self,
        config: AgentConfig | None,
        roles: RoleMap,
        catalog: CatalogView,
        providers: Mapping[str, Provider],
    ) -> ResolvedModel:
        """One agent's model (None: the manager's default model)."""
        return resolve_model(
            agent=config.name if config is not None else None,
            spec=config.model if config is not None else None,
            generation=config.model_settings if config is not None else None,
            thinking_effort=self._settings.thinking_effort,
            roles=roles,
            catalog=catalog,
            providers=providers,
        )

    def _simple_model(self, max_tokens: int, agent: str | None = None) -> ResolvedModel:
        """The default model, or ``agent``'s, as a short side call runs it.

        ``generate_simple`` runs on the default model; a summary of an
        agent's conversation runs on that agent's own model. Resolved again
        from the model's reference, without the agent's generation settings
        or the global thinking effort: thinking is ``none`` (off wherever the
        model allows it) and ``max_tokens`` is the caller's. Only after
        initialization.

        Raises:
            KeyError: ``agent`` is not one of this manager's agents.
        """
        base = self._default_model if agent is None else self._resolved_models[agent]
        return resolve_model(
            agent=agent,
            spec=str(base.ref),
            generation=ModelSettings(
                max_tokens=max_tokens, thinking=ThinkingSettings(mode="none")
            ),
            thinking_effort="none",
            roles=RoleMap(default_model=None, model_roles={}),  # a reference needs none
            catalog=self._catalog,
            providers={base.provider.name: base.provider},
        )

    def _default_ref_without_listings(self) -> ModelRef:
        """The default role's model, from settings and the shipped table only."""
        from agentic_cli.workflow import model_services

        providers = model_services.providers_from_settings(self._settings)
        roles = model_services.role_map_from_settings(self._settings, self._model_override)
        # Never loaded: the entries and the shipped table.
        catalog = model_services.build_catalog(
            self._settings,
            providers,
            agent_configs=self._agent_configs,
            manager_model=self._model_override,
        )
        return roles.resolve("default", catalog=catalog, providers=providers)

    def _validate_agent_graph(self) -> "AgentGraph | None":
        """Validate the declared agent graph. Backends may narrow this.

        Runs before model listings and service creation so a static configuration
        error surfaces immediately and costs nothing.

        Returns:
            The validated graph, whose root is ``root_agent_name``; None when
            the backend validates none of its own (the base, LangGraph).
        """
        return None

    async def _ensure_initialized(self) -> None:
        """Initialize on demand. Backends override to add readiness checks."""
        if not self._initialized:
            await self.initialize_services()

    def _backend_ready(self) -> bool:
        """Whether the backend resources a turn needs are live right now.

        Backends override to check their own handles (ADK: runner, session
        service, root agent). Used by :meth:`_turn_admission` to detect a
        cleanup that landed between a turn's initialization and its admission.
        """
        return self._initialized

    @contextlib.asynccontextmanager
    async def _turn_admission(self) -> "AsyncGenerator[None, None]":
        """Hold the turn lock with a *live* backend behind it.

        Initialization takes the lifecycle lock, so a turn must initialize
        **before** taking the turn lock — that ordering is what stops
        cleanup (lifecycle → turn) from deadlocking against a running turn.
        It also leaves a window: a cleanup already queued on the turn lock runs
        first and releases everything the turn just initialized, and the turn
        was then admitted to a torn-down backend (a ``None`` runner, surfacing
        as an ``AttributeError`` deep inside ADK).

        Readiness is therefore re-checked *while holding the turn lock*. If the
        backend was released underneath, the lock is dropped and initialization
        retried once — anything worse fails cleanly rather than running against
        released resources.

        No turn runs while the manager holds services another manager owns
        (a hand-over: borrowed and not yet given, or given and not yet
        released), checked under the turn lock too, so a turn that queued
        behind ``give_services`` is refused.

        Raises:
            RuntimeError: If the backend cannot be made ready, or the
                manager holds services it does not own.
        """
        for attempt in (1, 2):
            await self._ensure_initialized()
            await self._turn_lock.acquire()
            if self._service_set.not_owned:
                self._turn_lock.release()
                raise RuntimeError(
                    f"{type(self).__name__} runs no turn while it holds services "
                    "it does not own: borrowed and not given to it yet, or given "
                    "to another manager and not released. Their permission "
                    "checks answer for their owner's conversations."
                )
            if self._backend_ready():
                try:
                    yield
                finally:
                    self._turn_lock.release()
                return
            self._turn_lock.release()
            logger.info("turn_admission_retry", attempt=attempt)
        raise RuntimeError(
            f"{type(self).__name__} was released while this turn waited for "
            "admission and could not be reinitialized. Retry the request."
        )

    @contextlib.asynccontextmanager
    async def _turn(
        self, session_id: str | None = None, user_id: str | None = None
    ) -> "AsyncGenerator[SessionRef, None]":
        """Run the block as one turn of a conversation, and yield the conversation.

        How every turn-shaped entry point starts (``process``,
        ``resume_with_job_result``, ``compact``, ``run_instance``,
        ``start_instance``, ``tell_root``): the conversation is
        resolved as ``session_ref`` resolves it (the manager's current
        session and the default user for what is omitted) and named in the
        log context; then turn admission (:meth:`_turn_admission`: the
        manager initialized outside the turn lock, the lock taken, the
        backend re-checked under it) and the workflow context
        (:meth:`_workflow_context`: the settings, the services and the
        active turn, reset as the block ends) are entered, in that order.
        The conversation's sub-agent records are loaded first, once
        (``load_agents``). As the block ends, however it ends, the sub-agent
        runs it left open are recorded interrupted (``agent_runs``), with
        what they used, and the turn waits until the records are written to
        the store; each turn is counted in its conversation's usage ledger
        (``usage_of``).

        Yields:
            The turn's conversation.

        Raises:
            RuntimeError: The backend cannot be made ready (``_turn_admission``).
            Exception: The store's, when the conversation's sub-agent records
                cannot be read (``load_agents``): the turn does not start.
        """
        ref = self.session_ref(session_id, user_id)
        bind_context(session_id=ref.session_id, user_id=ref.user_id)
        async with self._turn_admission():
            with self._workflow_context(session_id=ref.session_id, user_id=ref.user_id):
                await self.load_agents(ref)
                ledger = self.usage_of(ref)
                ledger.begin_turn()
                try:
                    yield ref
                finally:
                    # A sub-agent run the turn leaves open (a cancel, a
                    # failure) ends with it: no event can say so any more.
                    for stopped in self.agent_runs.interrupt_open(ref, usage=ledger.run_usage):
                        # An instance left closed (a one-shot type) never runs
                        # again to answer its calls: free them.
                        known = self.agent_runs.instance(ref, stopped.instance)
                        open_calls = getattr(self, "_open_calls", None)
                        closed = known is not None and known.status == INSTANCE_CLOSED
                        if closed and open_calls is not None:
                            open_calls.take_cancel(ref, instance=stopped.instance)
                    await self._agent_records_written(ref)

    async def _release_resources(
        self, keep_session_service: bool = False, keep_checkpointer: bool = False
    ) -> None:
        """Release everything this manager owns. Idempotent.

        The one release protocol: ``cleanup()``, ``reinitialize()`` and the
        rollback of a failed initialization (``_initialize_locked``) all run
        it, and a backend takes part through two hooks rather than an
        override. In order:

        0. A release that detaches the store first waits for the sub-agent
           records still being written, up to ``RECORDS_RELEASE_WAIT``, and
           abandons what the store does not answer (``RecordsWriter``). A
           cancel in that wait detaches and starts the closers all the same,
           then raises at once, the closers finishing in their task.
        1. The base's state, at once: ``is_initialized`` False, the context
           budgets and the memory index snapshots dropped, the catalog
           detached.
        2. What the backend holds: :meth:`_detach_backend` (the agents, the
           runner, the MCP sessions), then :meth:`_detach_store` (the session
           store), which is not called while a model switch keeps the store
           (:meth:`_keeping`). Each detaches its handles and returns their
           closers.
        3. The services, detached now by :meth:`_cleanup_managers`, except
           those a model switch carries and those another manager owns (lent
           to this one, or given away: dropped, never closed here), and their
           closes started on a worker thread
           (:meth:`_start_closing_services`): they block, and on the event
           loop they froze the app.
        4. The closers, in one task of its own, in a fixed order: the
           backend's, the store's and the catalog's, while the services
           close on their thread, then the services' closes waited for,
           last and under a shield of their own, so a service close that
           hangs holds none of the others, and a cancel never cancels one
           that has not started. Each is isolated (one that raises is
           logged, and the rest still run), and the task is awaited under a
           shield: a cancel of the caller (Ctrl+C during a ``/settings``
           model switch, a cancelled ``async with manager``) stops only the
           wait, and everything still closes, once. Only an event loop
           shutting down, which cancels the closing task itself, skips the
           closers after the one running; the services' closes still run,
           on a thread the loop joins as it closes.

        It raises nothing a close raises. It raises ``TypeError`` for a hook
        that is not synchronous (returns an awaitable), after releasing
        everything else: that hook detached nothing, so its handles are still
        open.

        A subclass that overrides this method (as 0.6's ADK backend did, with
        ``keep_session_service``) and calls it is called without the keyword
        during a model switch, since :meth:`reinitialize` keeps the store
        through :meth:`_keeping`: the override reads ``self._keeping_store``
        to know whether the store is being kept.

        Args:
            keep_session_service: Deprecated, removed in 0.8.0: keeps the
                session store for this release, as :meth:`_keeping` does.
            keep_checkpointer: Deprecated: the same, accepted for code written
                against the 0.7 development line (Deep Agents' own release
                took it).

        Raises:
            TypeError: A detach hook returned an awaitable.
        """
        logger.debug("cleaning_up_workflow_manager")
        if keep_session_service or keep_checkpointer:
            warnings.warn(
                "_release_resources(keep_session_service=..., keep_checkpointer=...) "
                "is deprecated and will be removed in 0.8.0; release inside "
                "`with self._keeping(True):`, which keeps the session store.",
                DeprecationWarning,
                stacklevel=2,
            )
        keep_store = self._keeping_store or keep_session_service or keep_checkpointer

        # A store about to be detached first takes the sub-agent records still
        # being written (a write looks it up as it starts, so one queued behind
        # another would find it detached): waited for, bounded, before the
        # release's synchronous part, so nothing is detached behind an await.
        # A cancel here stops only the wait: the writes still running are
        # abandoned and the release goes on, then re-raises it.
        cancelled: asyncio.CancelledError | None = None
        writer = self.__dict__.get("_records_writer")
        if writer is not None and not keep_store:
            flushing = asyncio.ensure_future(
                writer.flush_all(timeout=RECORDS_RELEASE_WAIT)
            )
            try:
                await asyncio.shield(flushing)
            except asyncio.CancelledError as error:
                cancelled = error
                flushing.cancel()
                writer.abandon()

        self._initialized = False
        self._budgets = {}
        self._choice_budgets = {}
        self._memory_index.clear()
        self._rosters.clear()
        catalog, self._catalog = self._catalog, None

        not_synchronous: list[TypeError] = []
        closers = self._detached(self._detach_backend, not_synchronous)
        if not keep_store:
            closers += self._detached(self._detach_store, not_synchronous)
        closers.append((
            "model_catalog",
            functools.partial(self._aclose_owned, catalog, "model_catalog"),
        ))

        # The services' hook: their closes start on worker threads now.
        services: list[asyncio.Future[None]] = []
        self._services_closing = services
        try:
            self._cleanup_managers()
        finally:
            self._services_closing = None
        if services:
            closers.append(
                ("services", lambda: asyncio.shield(asyncio.gather(*services)))
            )

        closing = asyncio.ensure_future(self._close_in_order(closers))
        _CLOSING.add(closing)
        closing.add_done_callback(_CLOSING.discard)
        if cancelled is not None:
            # The cancel stops only the wait: everything is detached, and the
            # closers finish in their own task (held in ``_CLOSING``).
            raise cancelled
        await asyncio.shield(closing)
        logger.info("workflow_manager_cleaned_up")
        if not_synchronous:
            raise not_synchronous[0]

    def _detach_backend(self) -> list[Closer]:
        """Detach what the backend built for this initialization, and say how to close it.

        A backend hook of the release (:meth:`_release_resources`): set every
        handle a turn runs on to None (so ``_backend_ready`` is False and a
        repeat release finds nothing) and return the closers of what they
        held, in the order they must run. Called on every release, a model
        switch's included. It must be synchronous (``def``, never
        ``async def``) and must not raise. The services may still be
        closing when its closers run (the base starts their closes first,
        on a worker thread, and waits for them last); that is safe because
        no turn is in flight (a release holds the turn lock, or rolls back
        an attempt no turn ran on) and the services never reach the
        backend's MCP servers or its store. The base holds nothing here.
        """
        return []

    def _detach_store(self) -> list[Closer]:
        """Detach the session store, and say how to close it.

        A backend hook of the release (:meth:`_release_resources`), as
        :meth:`_detach_backend` is, for the store the conversations live in
        (a session service, a checkpointer and its connections): synchronous,
        never raising, and its closers run while the services may still be
        closing, which is safe for the same reasons (no turn in flight; no
        service reaches the store). Not called while a model switch keeps the store
        (:meth:`_keeping`): the backend then finds its store still attached
        when it builds again, and must reuse it rather than open another. The
        base holds nothing here.
        """
        return []

    @staticmethod
    def _detached(
        detach: Callable[[], list[Closer]], not_synchronous: list[TypeError]
    ) -> list[Closer]:
        """What a detach hook returns; nothing, logged, when it raises.

        A hook that returns an awaitable (written ``async def`` by mistake)
        detached nothing, since its body never ran: the awaitable is closed,
        so no "never awaited" warning follows, and a ``TypeError`` naming the
        hook is added to ``not_synchronous`` for the release to raise once
        everything else is released.
        """
        name = getattr(detach, "__qualname__", repr(detach))
        try:
            closers = detach()
            if inspect.isawaitable(closers):
                if inspect.iscoroutine(closers):
                    closers.close()
                not_synchronous.append(TypeError(
                    f"{name} must be synchronous (def, not async def): it "
                    "returned an awaitable, so it detached nothing and what it "
                    "holds is still open. Set its handles to None and return "
                    "their closers; the closers may be coroutine functions."
                ))
                return []
            return list(closers)
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail
            logger.warning("resource_detach_failed", hook=name, error=str(exc))
            return []

    @staticmethod
    async def _close_in_order(closers: list[Closer]) -> None:
        """Run each closer in turn; one that raises is logged and the rest still run."""
        for label, close in closers:
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - shutdown must not fail
                logger.warning("resource_close_failed", resource=label, error=str(exc))

    async def _close_catalog(self) -> None:
        """Release the model catalog (its background refresh). Idempotent."""
        catalog, self._catalog = self._catalog, None
        await self._aclose_owned(catalog, "model_catalog")

    @abstractmethod
    async def _do_initialize(self) -> None:
        """Backend-specific initialization (create agents/graph).

        Subclasses implement this instead of ``initialize_services()``.
        """
        ...

    def _reset_model(self, model: str | None) -> None:
        """Set this manager's model override for the next initialization.

        ``None`` drops the override: the default comes from settings again.
        The models the last initialization resolved are forgotten.
        """
        self._model_override = model
        self._default_model = None
        self._resolved_models = {}
        self._choice_models = {}

    @abstractmethod
    async def process(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Process user input through the agentic workflow.

        This is the main entry point for message processing. Implementations
        should:
        1. Ensure services are initialized
        2. Set up appropriate context for tools
        3. Process the message through the backend
        4. Yield WorkflowEvent objects for each step

        A turn that fails yields one ``ERROR`` event and returns, raising
        nothing: whatever its run raised (a provider's error, a rate limit
        included), a context overflow nothing recovered, or, on ADK, a main
        agent's reply that is no answer. ``WorkflowEvent.turn_failed`` builds
        it, from the exception or, where none ended the turn, its code: not
        recoverable, the error's text as ``content``, ``rate_limited`` (a
        provider's 429, ``retry.is_rate_limit_error``: the CLI says to send
        the request again later, since a turn is never replayed) and
        ``error_type`` (the exception's type, else the code, for logs) in its
        ``metadata``.
        An ``ERROR`` the turn already yielded that is not recoverable (the
        overflow handler's ``CONTEXT_OVERFLOW``, say) is its one ``ERROR``: a
        failure after it is logged, not reported again. Recoverable
        ``ERROR``s are warnings, and the turn goes on. A cancel still raises
        ``asyncio.CancelledError``, and a turn that cannot start raises
        before any event: the backend cannot be initialized or admitted
        (``_turn``), or its conversation cannot be read or prepared before
        the run (the store's error; on ADK ``ConversationChangedError`` for
        a session another process changed). ``resume_with_job_result``
        fails the same way.

        Args:
            message: User message to process.
            user_id: User identifier.
            session_id: Optional session identifier.

        Yields:
            WorkflowEvent objects representing workflow output.
        """
        # This is needed to make the method an async generator
        if False:  # pragma: no cover
            yield  # type: ignore[misc]

    async def reinitialize(
        self,
        model: str | None = None,
        preserve_sessions: bool = True,
    ) -> None:
        """Rebuild the backend with a new configuration (a model switch).

        Takes the lifecycle lock and then the turn lock, so it waits for a
        running turn; releases (:meth:`_release_resources`), sets the model
        (:meth:`_reset_model`) and initializes again.

        Transactional: on failure the manager is left *uninitialized*
        (``is_initialized`` False) with nothing half-built, rather than
        holding a backend that would answer as if it were ready; the caller
        (``WorkflowController``) turns that into a FAILED state and keeps the
        manager for a retry. What a preserving switch keeps survives both
        outcomes (:meth:`_keeping`), so a retry continues the same
        conversations.

        Args:
            model: The new default model; None re-resolves it from settings.
            preserve_sessions: Keep the session store and the services
                (permission engine, sandbox, jobs, knowledge bases) and
                rebuild only the backend. False releases them and builds
                fresh ones.

        Raises:
            Exception: Whatever initialization raised, after the rollback.
        """
        logger.info(
            "reinitializing_workflow_manager",
            new_model=model,
            preserve_sessions=preserve_sessions,
        )
        async with self._lifecycle_lock:
            async with self._turn_lock:
                with self._keeping(preserve_sessions):
                    await self._release_resources()
                    self._reset_model(model)
                    await self._initialize_locked()
        logger.info(
            "workflow_manager_reinitialized",
            model=self.model,
            sessions_preserved=preserve_sessions,
        )

    async def cleanup(self) -> None:
        """Release everything this manager owns (:meth:`_release_resources`).

        Call it before shutting down. Idempotent: references are detached
        before they are closed, so a repeat call finds nothing. Takes the
        lifecycle lock and then the turn lock, so it waits for a running
        turn. A turn after it initializes the manager again.
        """
        async with self._lifecycle_lock:
            async with self._turn_lock:
                await self._release_resources()

    # User input handling — callback-only

    async def request_user_input(self, request: UserInputRequest) -> str:
        """Request user input from the CLI via callback.

        Called by tools that need user interaction. Requires
        ``set_input_callback()`` to be set by the consumer (e.g.
        MessageProcessor) before any tool invokes this method. The callback is
        resolved from the *current context*, so a tool always reaches the
        consumer that started its turn.

        The callback answers only inside its turn. A task a turn started
        (``asyncio.create_task`` copies the context, the callback with it)
        that asks after the turn ended reaches no one: the consumer has moved
        on, and its dialog would open outside any turn. It gets
        ``UserInputUnavailable``, as when no approver is attached, so a
        question has no answer and a permission prompt is denied (the
        engine's no-approver rule).

        A request that names no run is given the sub-agent run asking
        (``get_active_run()``; None in the root's own calls), so the consumer
        can say which agent asks, whoever built the request.

        A sub-agent run waits for the answer with its working slot lent
        (``agents/delegate.py::waiting_on_user``): it is ``input_required``
        meanwhile, another run may work in its slot, and it takes a slot
        back, ahead of the runs that have not started, once the user has
        answered. A request no one can answer (``can_request_user_input``)
        raises before the run lends anything. A callback that answers
        without waiting still lends the slot: a run queued for one may then
        work in it before the asking run takes it back.

        Args:
            request: The user input request.

        Returns:
            User's response string.

        Raises:
            UserInputUnavailable: If no callback is registered, or the turn
                this context comes from has ended (a RuntimeError).
        """
        if request.run is None:
            request.run = get_active_run()
        logger.debug(
            "user_input_requested",
            request_id=request.request_id,
            tool_name=request.tool_name,
        )

        callback = self._user_input_callback.get()
        if callback is None:
            raise UserInputUnavailable(
                "No user input callback registered. "
                "Call set_input_callback() before invoking tools that require user input."
            )
        if turn_has_ended():
            logger.info(
                "user_input_after_its_turn",
                request_id=request.request_id,
                tool_name=request.tool_name,
            )
            raise UserInputUnavailable(
                "The turn this request comes from has ended; no one is there to answer it."
            )

        # A call cancelled at its question was not run (workflow/interrupts.py).
        call = get_active_tool_call()
        async with waiting_on_user():
            if call is None:
                return await callback(request)
            with self._open_calls.asking(call.call_id):
                return await callback(request)

    # Async context manager support

    async def __aenter__(self) -> "BaseWorkflowManager":
        """Async context manager entry - initialize services."""
        await self.initialize_services()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup resources."""
        await self.cleanup()

    # Optional methods with default implementations

    def _report_side_call(self, event: WorkflowEvent) -> None:
        """Account for a side call (``generate_simple``), and report it in a turn.

        ``event`` is the call's ``LLM_USAGE``, its counts normalized
        (``workflow/usage.py``). Its purpose is the one its caller named
        (``usage.side_call``: ``tool`` for ``summarize``,
        ``memory_extraction`` for ``on_session_end``), else ``side_call``; it
        is the run in progress (``get_active_run``: a sub-agent's tool's call
        is that run's), and names that run's agent, else the main agent. It is
        recorded in the conversation its caller named, else the turn's, else
        the open one, and, in a turn, reported through the turn's live
        events, so the stream carries it as it carries the agents' own. A
        task that outlived its turn (``turn_has_ended``) records its call in
        that turn's conversation, with no turn, and reports it to no one:
        the turn's stream has ended.
        """
        side = current_side_call()
        run = get_active_run()
        event.run = run
        event.metadata["purpose"] = side.purpose if side is not None else PURPOSE_SIDE_CALL
        agent = run.agent if run is not None else self.root_agent_name
        if agent:
            event.metadata["agent"] = agent
        turn = get_active_turn()
        if side is not None and side.conversation is not None:
            conversation = side.conversation
        else:
            conversation = turn or self.session_ref()
        self.usage_of(conversation).record(event)
        if turn is not None and not turn_has_ended():
            from agentic_cli.workflow import live_events

            live_events.report(event)

    async def generate_simple(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a simple text response using the current model.

        Used for internal operations like summarization. Does not go through
        the full agent workflow.

        Args:
            prompt: The prompt to send.
            max_tokens: Maximum tokens in response.

        Returns:
            Generated text response.

        Note:
            Default implementation raises NotImplementedError.
            Subclasses should override if they support simple generation,
            and report the call's usage with ``_report_side_call``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement generate_simple"
        )

    # -------------------------------------------------------------------------
    # Session save/resume
    # -------------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sessions (native, durable). The backend store (ADK
    # DatabaseSessionService / LangGraph checkpointer) persists conversation
    # state continuously, keyed by session_id; there is no separate snapshot.
    # ------------------------------------------------------------------

    @property
    def supports_sessions(self) -> bool:
        """Whether this backend implements the durable-session hooks.

        Derived from the subclass actually overriding ``session_exists``, so a
        backend opts in by implementing the capability rather than by setting a
        flag that can drift from the code.
        """
        return type(self).session_exists is not BaseWorkflowManager.session_exists

    def _is_default_user(self, user_id: str | None) -> bool:
        """Whether ``user_id`` is (or defaults to) the configured default user.

        Base-class helpers call user-scoped backend hooks *without* the
        ``user_id`` keyword in that case, so a backend that predates the
        parameter (LangGraph, and any downstream override) keeps working; an
        explicit non-default user is always passed through, so it can never be
        silently serviced as the default user.
        """
        return user_id is None or user_id == self._settings.default_user

    @property
    def session_store(self) -> str:
        """The session store it keeps conversations in: the one it was built
        for, else ``settings.session_store``.

        A CLI run keeps one store for its whole life (a change of
        ``session_store`` takes effect at the next start), so the factory
        builds every manager of the run for that store
        (``create_workflow_manager_from_settings(session_store=)``). Set it
        before ``initialize_services``, which opens the store.
        """
        return self._built_for_store or self._settings.session_store

    @session_store.setter
    def session_store(self, store: str) -> None:
        self._built_for_store = store

    def session_ref(
        self, session_id: str | None = None, user_id: str | None = None
    ) -> SessionRef:
        """Resolve a full :class:`SessionRef` from partial identity.

        Unsupplied parts default to this manager's app name, the configured
        ``default_user`` and the manager's current session id. Defaults apply
        only to what the caller omitted: an explicit ``user_id`` is never
        replaced by the default user.
        """
        return SessionRef(
            app_name=self.app_name,
            user_id=user_id or self._settings.default_user,
            session_id=session_id or self.session_id,
        )

    async def save_session(
        self, session_id: str | None = None, *, user_id: str | None = None
    ) -> dict:
        """No-op flush — durable stores persist as the turn runs.

        Kept for API compatibility / explicit "checkpoint now" intent.

        Args:
            session_id: Session to report (defaults to the manager's current one).
            user_id: Owner (defaults to ``settings.default_user``).

        Returns:
            ``{"success": True, "session_id": ..., "user_id": ...}`` — the full
            identity, so a caller working on behalf of another user can tell
            which conversation was meant.
        """
        ref = self.session_ref(session_id, user_id)
        return {
            "success": True,
            "session_id": ref.session_id,
            "user_id": ref.user_id,
        }

    async def load_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        """Adopt ``session_id`` for resume; the native store already holds it.

        Returns True if that session already has content (i.e. a real resume),
        False if it's new — but the id is adopted either way so the next turn
        continues it, once the store was asked (a cancel or an error before
        that adopts nothing). A backend without durable sessions has nothing
        to resume, so it adopts the id and returns False. Its sub-agent
        records load with it (``load_agents``), read again from the store
        whenever the conversation is opened (another process may have had it
        in between); records that cannot be read now are warned of and read
        again at its next turn's start. A conversation this manager holds
        what its store lacks of (a failed write, a run still working) is not
        read again.

        Args:
            session_id: Session to adopt.
            user_id: Owner to look the session up as (defaults to
                ``settings.default_user``).
        """
        if not self.supports_sessions:
            self.session_id = session_id
            # A resumed conversation takes a new memory index (spec §11.1).
            self._forget_memory_index(self.session_ref(session_id, user_id))
            logger.info(
                "session_adopted", session_id=session_id, resumed=False,
                backend_sessions=False,
            )
            return False
        if self._is_default_user(user_id):
            exists = await self.session_exists(session_id)
        else:
            exists = await self.session_exists(session_id, user_id=user_id)
        self.session_id = session_id
        ref = self.session_ref(session_id, user_id)
        self._forget_memory_index(ref)
        logger.info("session_adopted", session_id=session_id, resumed=exists)
        # Its sub-agent records load with it, read again from the store
        # (another process may have had the conversation since this manager
        # last did: its own writes land first), unless this manager holds what
        # the store lacks: a write that failed, or a run of it still working.
        await self._agent_records_written(ref)
        if not self._holds_unsaved_agents(ref):
            self._agents_loaded.discard(ref)
        try:
            await self.load_agents(ref)
        except Exception as error:  # noqa: BLE001 - the conversation is adopted all the same
            logger.warning(
                "subagent_records_unread", session_id=session_id,
                error=str(error) or type(error).__name__,
            )
            logger.debug("subagent_records_unread", exc_info=error)
        return exists

    # ---- Backend hooks (override in ADK / LangGraph managers) ----
    #
    # Each takes an optional ``user_id`` so a session created for one user
    # stays reachable through the public API. Backends that cannot persist
    # sessions must not answer with a misleading "no" — the base raises.

    def _no_session_support(self, operation: str) -> NotImplementedError:
        """Error for a session operation the backend does not implement."""
        return NotImplementedError(
            f"{type(self).__name__} does not implement durable sessions "
            f"({operation}). Check ``supports_sessions`` before calling."
        )

    async def session_exists(self, session_id: str, *, user_id: str | None = None) -> bool:
        """Whether the native store already holds this session's state."""
        raise self._no_session_support("session_exists")

    async def recent_messages(
        self, session_id: str, limit: int = 20, *, user_id: str | None = None
    ) -> list[dict]:
        """Recent ``{role, content}`` text messages from the native session.

        Used for session-end fact extraction; text-only (no tool-call fidelity).
        """
        raise self._no_session_support("recent_messages")

    async def list_sessions(
        self, *, user_id: str | None = None, limit: int | None = None
    ) -> list[SessionInfo]:
        """The stored conversations of one user, newest first.

        Args:
            user_id: Their owner (defaults to ``settings.default_user``).
            limit: Only the newest this many (None: all of them).
        """
        raise self._no_session_support("list_sessions")

    async def session_info(
        self, session_id: str, *, user_id: str | None = None
    ) -> SessionInfo | None:
        """One stored conversation's row, or None when it is not stored.

        Args:
            session_id: The conversation.
            user_id: Its owner (defaults to ``settings.default_user``).
        """
        raise self._no_session_support("session_info")

    async def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        """Delete a persisted session from the native store."""
        raise self._no_session_support("delete_session")


