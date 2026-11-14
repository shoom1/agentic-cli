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
import functools
import inspect
import warnings
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from types import MappingProxyType
from typing import Any, AsyncGenerator, Awaitable, Callable, Iterator, Mapping, TYPE_CHECKING

from agentic_cli.workflow.events import (
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
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.agents.runs import AgentRuns
from agentic_cli.workflow.usage import UsageLedger, UsageLedgers
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
    SessionInfo,
    SessionRef,
    close_turn_scope,
    get_active_tool_call,
    get_active_turn,
    open_turn_scope,
    reset_active_turn,
    set_active_turn,
    turn_has_ended,
)
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
        ``resume_with_job_result()`` and ``compact()`` enter through
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
        one. The release clears the snapshots.

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
                    return validate_agent_graph(self._agent_configs, backend=self.backend_type)

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

    def __init__(
        self,
        agent_configs: list[AgentConfig],
        settings: "BaseSettings | None" = None,
        app_name: str | None = None,
        model: str | None = None,
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
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
        """
        from agentic_cli.config import get_settings

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

    @property
    def agent_configs(self) -> list[AgentConfig]:
        """Get the agent configurations."""
        return self._agent_configs

    @property
    def agent_runs(self) -> AgentRuns:
        """This manager's sub-agent runs, kept by conversation (provisional).

        Named by the backends' event processors and recorded from their
        ``AGENT_RUN`` events (``workflow/agents/runs.py``): ``records(ref)``
        lists a conversation's in the order they started; a turn ends the
        runs it leaves open (``_turn``). In memory, for the process; made on
        first use.
        """
        runs = self.__dict__.get("_agent_runs")
        if runs is None:
            runs = self.__dict__["_agent_runs"] = AgentRuns()
        return runs

    @property
    def usage(self) -> UsageLedger:
        """The open conversation's token usage: every model call made in it (provisional).

        A row per call, the root's and its sub-agents' steps and the side
        calls (summaries, ``generate_simple``), each counted alike on every
        backend (``workflow/usage.py``): totals by run, agent, model, purpose
        and turn. In memory, for the process. ``usage_of`` reads another
        conversation's.
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
            answer = await self.generate_simple(self._fit_prompt(prompt, 2000), max_tokens=2000)
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
        """Drop a conversation's snapshot (the turn's by default): its next request takes a new one."""
        self._memory_index.forget(ref)

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
    def root_agent_name(self) -> str | None:
        """The root agent's name, from the validated agent graph.

        None before initialization, and on a manager that validates no graph
        of its own (LangGraph).
        """
        return self._root_agent_name if self._initialized else None

    def context_budget(self, agent_name: str | None = None) -> "ContextBudget":
        """An agent's context numbers, as its backend summarizes and trims by them.

        Its window (or the 128k floor), input budget, summary trigger and the
        tail a summary keeps (``workflow/context.py``), computed when the
        models were resolved, before any service was built: the numbers
        ``/status`` shows.

        Args:
            agent_name: The agent; None for the root agent (``root_agent_name``).

        Raises:
            RuntimeError: Before initialization, or after cleanup.
            KeyError: ``agent_name`` names no agent whose context this manager
                manages: one whose model did not resolve (a native ADK YAML
                app's). LangGraph manages none: every name is unknown there,
                the root's included.
        """
        if not self._initialized:
            raise RuntimeError(
                f"{type(self).__name__} is not initialized: its agents' context "
                "budgets are computed when it initializes."
            )
        name = self.root_agent_name if agent_name is None else agent_name
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
        if errors:
            raise SettingsValidationError("\n".join(dict.fromkeys(errors)))
        resolved.update(self._resolve_extra_models(roles, providers))

        self._default_model, self._resolved_models = default, resolved
        logger.info(
            "models_resolved",
            default=str(default.ref),
            agents={name: str(model.ref) for name, model in resolved.items()},
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
        ``resume_with_job_result``, ``compact``): the conversation is
        resolved as ``session_ref`` resolves it (the manager's current
        session and the default user for what is omitted) and named in the
        log context; then turn admission (:meth:`_turn_admission`: the
        manager initialized outside the turn lock, the lock taken, the
        backend re-checked under it) and the workflow context
        (:meth:`_workflow_context`: the settings, the services and the
        active turn, reset as the block ends) are entered, in that order.
        As the block ends, however it ends, the sub-agent runs it left open
        are recorded interrupted (``agent_runs``), with what they used; each
        turn is counted in its conversation's usage ledger (``usage_of``).

        Yields:
            The turn's conversation.

        Raises:
            RuntimeError: The backend cannot be made ready (``_turn_admission``).
        """
        ref = self.session_ref(session_id, user_id)
        bind_context(session_id=ref.session_id, user_id=ref.user_id)
        async with self._turn_admission():
            with self._workflow_context(session_id=ref.session_id, user_id=ref.user_id):
                ledger = self.usage_of(ref)
                ledger.begin_turn()
                try:
                    yield ref
                finally:
                    # A sub-agent run the turn leaves open (a cancel, a
                    # failure) ends with it: no event can say so any more.
                    self.agent_runs.interrupt_open(ref, usage=ledger.run_usage)

    async def _release_resources(
        self, keep_session_service: bool = False, keep_checkpointer: bool = False
    ) -> None:
        """Release everything this manager owns. Idempotent.

        The one release protocol: ``cleanup()``, ``reinitialize()`` and the
        rollback of a failed initialization (``_initialize_locked``) all run
        it, and a backend takes part through two hooks rather than an
        override. In order:

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

        self._initialized = False
        self._budgets = {}
        self._memory_index.clear()
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

        Args:
            request: The user input request.

        Returns:
            User's response string.

        Raises:
            UserInputUnavailable: If no callback is registered, or the turn
                this context comes from has ended (a RuntimeError).
        """
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
            Subclasses should override if they support simple generation.
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
        to resume, so it adopts the id and returns False.

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
        self._forget_memory_index(self.session_ref(session_id, user_id))
        logger.info("session_adopted", session_id=session_id, resumed=exists)
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


