"""Abstract base class for workflow managers.

This module defines the interface that all workflow orchestration backends
must implement, enabling pluggable orchestrators (ADK, LangGraph, etc.).

The base class provides auto-detection of required managers (memory, planning,
HITL) based on tool requirements, creating them lazily when needed.

It also provides shared implementations for:
- User input handling (callback-based)
- Model resolution (lazy resolution from settings)
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from typing import Any, AsyncGenerator, Awaitable, Callable, Iterator, TYPE_CHECKING

from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.models import ModelRegistry
from agentic_cli.workflow.sessions import (
    SessionRef,
    get_active_turn,
    reset_active_turn,
    set_active_turn,
)
from agentic_cli.workflow.service_registry import (
    set_service_registry,
    ARXIV_SOURCE,
    JOB_MANAGER,
    KB_MANAGER,
    LLM_SUMMARIZER,
    MEMORY_STORE,
    PERMISSION_ENGINE,
    SANDBOX_MANAGER,
    USER_KB_MANAGER,
    WORKFLOW,
)
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.tools.memory_tools import MemoryStore
    from agentic_cli.knowledge_base import KnowledgeBaseManager
    from agentic_cli.tools.sandbox.manager import SandboxManager

logger = Loggers.workflow()


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
        A manager runs **one turn at a time**. ``process()`` and
        ``resume_with_job_result()`` serialize on an internal turn lock, so
        overlapping callers queue rather than interleave — the backend's
        per-invocation event buffers are manager-scoped, and interleaving would
        let one invocation drain another's events. Run separate managers for
        genuine parallelism. (The HITL input callback is context-local, so it
        does not depend on that serialization; see ``set_input_callback``.)

        Lifecycle mutation (``initialize_services``/``reinitialize``/
        ``cleanup``) additionally takes the turn lock, so the backend is never
        torn down under a running generator. Both locks are released on
        cancellation. The active session/user identity remains a ContextVar,
        so it stays correct for nested and task-spawned work.

    Example:
        class CustomWorkflowManager(BaseWorkflowManager):
            async def initialize_services(self) -> None:
                # Setup custom backend
                pass

            async def process(self, message, user_id, session_id=None):
                # Process through custom backend
                async for event in self._process_internal():
                    yield event
    """

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
            app_name: Application name for services.
            model: Model override (auto-detected from API keys if not provided).
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
        # _turn_lock serializes turns, and is taken by lifecycle mutation so it
        # cannot tear the backend down under a running generator.
        # Lock order is lifecycle → turn: a turn releases the lifecycle lock
        # (inside _ensure_initialized) *before* taking the turn lock, so the
        # two can never deadlock.
        self._lifecycle_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._on_event = on_event

        # Model resolution (lazy)
        self._model: str | None = model
        self._model_resolved: bool = model is not None

        # User input handling (callback-only), context-local — see
        # set_input_callback(). One ContextVar per manager; there are only ever
        # a handful of managers in a process.
        self._user_input_callback: ContextVar[
            Callable[[UserInputRequest], Awaitable[str]] | None
        ] = ContextVar(f"agentic_cli_input_callback_{id(self):x}", default=None)

        # Model registry
        self._model_registry = ModelRegistry()

        # Resolve string / dotted-path tool refs to callables BEFORE manager
        # detection and tool assembly (both key on ``tool.__name__``).
        self._resolve_config_tool_refs()

        # Auto-detect required managers from tools
        self._required_managers = self._detect_required_managers()

        # Service registry — complex services (KB, sandbox, etc.)
        # Plan/task state lives in native backend state (ToolContext.state
        # for ADK, graph state for LangGraph), not here.
        self._services: dict[str, Any] = {}

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
    def model_registry(self) -> ModelRegistry:
        """Get the model registry."""
        return self._model_registry

    @property
    def required_managers(self) -> set[str]:
        """Get the set of required manager types detected from tools."""
        return self._required_managers

    @property
    def services(self) -> dict[str, Any]:
        """Get the service registry dict."""
        return self._services

    @property
    def memory_manager(self) -> "MemoryStore | None":
        """Get the memory manager (if required by tools)."""
        return self._services.get(MEMORY_STORE)

    @property
    def kb_manager(self) -> "KnowledgeBaseManager | None":
        """Get the project-scoped knowledge base manager (if required by tools)."""
        return self._services.get(KB_MANAGER)

    @property
    def user_kb_manager(self) -> "KnowledgeBaseManager | None":
        """Get the user-scoped knowledge base manager (if required by tools)."""
        return self._services.get(USER_KB_MANAGER)

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
        different code, and it is denied at permission time anyway. It is
        therefore passed through untouched, as is a tool registered into some
        other ``ToolRegistry``.

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

        Only creates tools for services that have been initialized.
        """
        from agentic_cli.tools.factories import (
            make_memory_tools,
            make_kb_tools,
            make_webfetch_tool,
            make_sandbox_tool,
            make_interaction_tools,
            make_arxiv_tools,
            make_ingest_arxiv_tool,
        )

        tool_map: dict[str, Callable] = {}
        s = self._services

        if s.get(MEMORY_STORE):
            for t in make_memory_tools(s[MEMORY_STORE]):
                tool_map[t.__name__] = t
        if s.get(KB_MANAGER):
            for t in make_kb_tools(s[KB_MANAGER], s.get(USER_KB_MANAGER)):
                tool_map[t.__name__] = t
        if s.get(LLM_SUMMARIZER):
            tool_map["web_fetch"] = make_webfetch_tool(s[LLM_SUMMARIZER])
        if s.get(SANDBOX_MANAGER):
            tool_map["sandbox_execute"] = make_sandbox_tool(s[SANDBOX_MANAGER], self)
        if s.get(ARXIV_SOURCE):
            for t in make_arxiv_tools(s[ARXIV_SOURCE]):
                tool_map[t.__name__] = t
        # ingest_arxiv_paper composes both services
        if s.get(ARXIV_SOURCE) and s.get(KB_MANAGER):
            tool_map["ingest_arxiv_paper"] = make_ingest_arxiv_tool(
                s[ARXIV_SOURCE], s[KB_MANAGER]
            )
        # Workflow manager is always available for interaction tools
        for t in make_interaction_tools(self):
            tool_map[t.__name__] = t

        # The factories bind each closure to the definition of the exact
        # module-level tool it re-binds (see ``factories._issued``), so a
        # variant carries identity only while the framework still owns that
        # tool — never merely because the names match.
        return tool_map

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
            Set of required service keys (e.g. ``{"kb_manager", "memory_store"}``).
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

        Synchronous convenience wrapper around
        :meth:`_build_services`/:meth:`_publish_services`; initialization uses
        the transactional path in :meth:`_construct_services` instead.
        """
        self._publish_services(self._build_services(frozenset(self._services)))

    def _publish_services(self, built: dict[str, Any]) -> None:
        """Adopt constructed services without displacing anything already live."""
        for key, service in built.items():
            self._services.setdefault(key, service)

    def _build_services(
        self, existing: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Construct the required services into a **fresh** dict.

        Pure construction: it never touches ``self._services``. Constructors
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
        """Construct the required services into ``s``. See :meth:`_build_services`."""
        if "memory_store" in self._required_managers and MEMORY_STORE not in existing:
            from agentic_cli.tools.memory_tools import MemoryStore

            embedding_service = None
            if not self._settings.knowledge_base_use_mock:
                from agentic_cli.knowledge_base.embeddings import EmbeddingService
                if EmbeddingService.is_available():
                    embedding_service = EmbeddingService(
                        model_name=self._settings.embedding_model,
                        batch_size=self._settings.embedding_batch_size,
                        device=self._settings.embedding_device,
                    )
            else:
                from agentic_cli.knowledge_base._mocks import MockEmbeddingService
                embedding_service = MockEmbeddingService()

            s[MEMORY_STORE] = MemoryStore(self._settings, embedding_service=embedding_service)

        if "kb_manager" in self._required_managers and KB_MANAGER not in existing:
            from pathlib import Path
            from agentic_cli.knowledge_base import KnowledgeBaseManager

            use_mock = self._settings.knowledge_base_use_mock
            project_kb_dir = Path.cwd() / f".{self._settings.app_name}" / "knowledge_base"
            user_kb_dir = self._settings.knowledge_base_dir

            s[KB_MANAGER] = KnowledgeBaseManager(
                settings=self._settings,
                use_mock=use_mock,
                base_dir=project_kb_dir,
            )

            if project_kb_dir.resolve() != user_kb_dir.resolve():
                s[USER_KB_MANAGER] = KnowledgeBaseManager(
                    settings=self._settings,
                    use_mock=use_mock,
                    base_dir=user_kb_dir,
                )
            else:
                s[USER_KB_MANAGER] = s[KB_MANAGER]

        if "llm_summarizer" in self._required_managers and LLM_SUMMARIZER not in existing:
            s[LLM_SUMMARIZER] = self

        if "sandbox_manager" in self._required_managers and SANDBOX_MANAGER not in existing:
            from agentic_cli.tools.sandbox.manager import SandboxManager
            s[SANDBOX_MANAGER] = SandboxManager(self._settings)

        if "job_manager" in self._required_managers and JOB_MANAGER not in existing:
            from pathlib import Path
            from agentic_cli.tools.jobs import JobManager

            # User-scoped so long jobs persist across projects and CLI restarts.
            jobs_dir = Path.home() / f".{self._settings.app_name}" / "jobs"
            s[JOB_MANAGER] = JobManager(
                self._settings,
                base_dir=jobs_dir,
                max_concurrent=getattr(self._settings, "max_concurrent_jobs", 4),
            )

        if "arxiv_source" in self._required_managers and ARXIV_SOURCE not in existing:
            from agentic_cli.tools.arxiv_source import ArxivSearchSource
            s[ARXIV_SOURCE] = ArxivSearchSource()

        # Always construct the PermissionEngine (all agents may need it)
        if PERMISSION_ENGINE not in existing:
            from pathlib import Path
            from agentic_cli.workflow.permissions import PermissionContext, PermissionEngine
            ctx = PermissionContext(
                workdir=Path.cwd(),
                home=Path.home(),
                app_name=self._settings.app_name,
            )
            s[PERMISSION_ENGINE] = PermissionEngine(
                settings=self._settings, workflow=self, ctx=ctx,
            )

        # Always ensure workflow reference is available
        s[WORKFLOW] = self

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
        """Release services constructed for an attempt that no longer owns init."""
        if build.cancelled() or build.exception() is not None:
            return
        built = build.result()
        if not built:
            return
        logger.warning("services_discarded_after_rollback", services=sorted(built))
        self._close_services(built)

    async def summarize(self, content: str, prompt: str) -> str:
        """Summarize content using the configured LLM.
        Args:
            content: The content to summarize (included in prompt by caller).
            prompt: The full summarization prompt.
        Returns:
            Summarized text response.
        """
        return await self.generate_simple(prompt, max_tokens=12000)

    async def on_session_end(
        self,
        messages: list[dict] | None = None,
        *,
        session: "SessionRef | None" = None,
    ) -> list[str]:
        """Hook called when a session ends. Optionally extracts facts.

        Override in downstream apps for custom session-end behavior.

        Args:
            messages: Recent messages from the session (optional).
            session: Which conversation to read when ``messages`` is omitted.
                Defaults to the turn still in context, else this manager's
                current session under ``settings.default_user`` — so a session
                belonging to another user is read as *that* user rather than
                silently coming back empty.

        Returns:
            List of extracted facts (empty if disabled or no messages).
        """
        if not getattr(self._settings, "auto_extract_session_facts", False):
            return []

        store = self._services.get(MEMORY_STORE)
        if store is None:
            return []

        # When the caller doesn't supply messages, pull them from the live
        # backend session (same source/sid save_session uses) so the CLI can
        # invoke this with no arguments on exit.
        if messages is None:
            ref = session or get_active_turn() or self.session_ref()
            try:
                if self._is_default_user(ref.user_id):
                    # Compatible call for backends that predate the user_id
                    # parameter (see _user_scoped_kwargs).
                    messages = await self.recent_messages(ref.session_id)
                else:
                    messages = await self.recent_messages(
                        ref.session_id, user_id=ref.user_id
                    )
            except Exception:
                logger.debug("session_fact_extraction_extract_failed", exc_info=True)
                return []

        if not messages:
            return []

        prompt = (
            "Extract key facts, decisions, and user preferences from this conversation. "
            "Return each fact as a single concise sentence on its own line. "
            "Only include facts worth remembering for future conversations. "
            "If there are no notable facts, return an empty response.\n\n"
        )
        content = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages[-20:]
        )

        try:
            summary = await self.generate_simple(prompt + content, max_tokens=2000)
        except Exception:
            logger.debug("session_fact_extraction_failed", exc_info=True)
            return []

        facts = [line.strip() for line in summary.strip().split("\n") if line.strip()]
        for fact in facts:
            store.store(fact, tags=["auto-extracted", "session"])
        return facts

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

    async def can_resume(self, record) -> bool:
        """Whether a finished job can be resumed into its conversation now.

        Base default: False (no resume support). Backends that implement
        ``resume_with_job_result`` override this to report whether the
        originating conversation is still available — e.g. the ADK session that
        holds the pending call. Used by the harness to resume vs. surface a
        "finished while its conversation was unavailable" notice.

        Sessions are durable by default (``session_store='sqlite'``), so a
        conversation normally survives a CLI restart and stays resumable. It is
        unavailable when, for example, it was deleted, the record is missing its
        session/user/call identifiers, or the run used the explicitly ephemeral
        ``session_store='memory'`` and the process restarted.
        """
        return False

    @contextlib.contextmanager
    def _workflow_context(
        self, session_id: str | None = None, user_id: str | None = None
    ) -> Iterator[None]:
        """Context manager that exposes the service registry to tools.

        Sets ContextVars (settings, the service registry, and the active turn)
        so tools can call ``get_service(key)`` during execution and the
        JobManager can associate a launched job with the conversation that
        started it.

        All three are restored from tokens on exit, so a nested context
        restores the outer turn rather than clearing it, and concurrent turns
        on one manager never see each other's identity.
        """
        from agentic_cli.config import set_context_settings

        settings_token = set_context_settings(self._settings)
        registry_token = set_service_registry(self._services)
        turn_token = set_active_turn(self.session_ref(session_id, user_id))
        try:
            yield
        finally:
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

    # Services that own OS resources, and the sync method that releases them.
    _SYNC_SERVICE_CLOSERS = (
        (SANDBOX_MANAGER, "cleanup"),
        (JOB_MANAGER, "close"),
    )

    @classmethod
    def _close_services(cls, services: dict[str, Any]) -> None:
        """Release the owned resources in a service mapping. Never raises.

        Each closer is isolated — one raising must not leave the rest open.
        Used both for the live registry and for services a rolled-back
        initialization constructed but never published.
        """
        for key, method in cls._SYNC_SERVICE_CLOSERS:
            service = services.get(key)
            if service is None:
                continue
            try:
                getattr(service, method)()
            except Exception as exc:  # noqa: BLE001 - shutdown must not fail
                logger.warning("resource_close_failed", resource=key, error=str(exc))

    def _cleanup_managers(self) -> None:
        """Release the synchronous resources this manager owns.

        Only services this manager created are released (see
        ``_build_services``). Idempotent: the registry is emptied, so a second
        call finds nothing. The registry is cleared regardless of failures.
        """
        try:
            self._close_services(self._services)
        finally:
            self._services = {}

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
        """Get the model name, resolving from settings if needed.

        This is resolved lazily to allow startup without API keys.
        Subclasses can override if they need custom model resolution.
        """
        if not self._model_resolved:
            self._model = self._settings.get_model()
            self._model_resolved = True
            logger.info("model_resolved", model=self._model)
        return self._model  # type: ignore[return-value]

    async def initialize_services(self, validate: bool = True) -> None:
        """Initialize backend services asynchronously.

        Concurrency-safe, idempotent and transactional: concurrent callers
        serialize on the lifecycle lock, late arrivals see ``_initialized`` and
        return, and a failed attempt releases whatever it had allocated instead
        of leaving the manager half-built.

        Args:
            validate: If True, validate settings before initialization.
        Raises:
            SettingsValidationError: If settings validation fails.
        """
        async with self._lifecycle_lock:
            await self._initialize_locked(validate=validate)

    async def _initialize_locked(self, validate: bool = True) -> None:
        """Initialization body. The caller must hold ``_lifecycle_lock``."""
        if self._initialized:
            return

        # Validate the declared agent graph before anything is allocated or
        # any network call is made: a bad graph is a static configuration
        # error and should not cost a model listing or an embedding model.
        self._validate_agent_graph()

        self._settings.export_api_keys_to_env()

        # Discover models BEFORE validating them: the static fallback list
        # would otherwise reject a model that exists but predates the list.
        await self._model_registry.refresh(
            google_api_key=self._settings.google_api_key,
            anthropic_api_key=self._settings.anthropic_api_key,
        )
        self._settings.set_model_registry(self._model_registry)

        if validate:
            self._validate_models()

        # Create services BEFORE backend init so _build_tools() can
        # produce factory-bound tools during agent/graph creation.
        # Construction is offloaded to a worker thread (heavy constructors)
        # and published transactionally — see _construct_services.
        try:
            await self._construct_services()
            await self._do_initialize()
        except BaseException:
            # Roll back: services (and any backend resource the partial
            # _do_initialize created) must not outlive the failed attempt.
            await self._release_resources()
            raise
        self._initialized = True

    # Label for this manager's own model in validation errors.
    _MODEL_OVERRIDE_LABEL = "workflow model"

    def _validate_models(self) -> None:
        """Validate every model this manager will actually use, and normalize it.

        ``settings.default_model`` and the per-agent overrides are validated by
        ``validate_settings``. This manager's *own* model is not in settings at
        all — it comes from ``Manager(model=...)``, ``reinitialize(model=...)``,
        or a ``settings.get_model()`` cached before discovery ran — so it is
        passed into the same all-or-nothing pass rather than checked separately:
        an unusable id must fail startup, and a deprecated one must be replaced
        by the id the runtime then sends.

        Raises:
            SettingsValidationError: If any effective model is unusable.
        """
        from agentic_cli.config import _validate_settings_with_models

        extras: list[tuple[str, str]] = []
        if self._model_resolved and self._model:
            extras.append((self._MODEL_OVERRIDE_LABEL, self._model))

        resolved = _validate_settings_with_models(
            self._settings, self._agent_configs, extras
        )

        replacement = resolved.get(self._MODEL_OVERRIDE_LABEL)
        if replacement is not None and replacement != self._model:
            logger.info(
                "model_override_resolved", requested=self._model, model=replacement
            )
            self._model = replacement

    def _validate_agent_graph(self) -> None:
        """Validate the declared agent graph. Backends may narrow this.

        Runs before discovery and service creation so a static configuration
        error surfaces immediately and costs nothing.
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

        Raises:
            RuntimeError: If the backend cannot be made ready.
        """
        for attempt in (1, 2):
            await self._ensure_initialized()
            await self._turn_lock.acquire()
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

    async def _release_resources(self) -> None:
        """Release everything this manager owns. Idempotent, never raises.

        Backends override to add their own resources (ADK closes the session
        service); the base releases the service registry.
        """
        self._cleanup_managers()

    @abstractmethod
    async def _do_initialize(self) -> None:
        """Backend-specific initialization (create agents/graph).

        Subclasses implement this instead of ``initialize_services()``.
        """
        ...

    def _reset_model(self, model: str | None) -> None:
        """Reset model state for reinitialisation."""
        if model is not None:
            self._model = model
            self._model_resolved = True
        else:
            self._model = None
            self._model_resolved = False

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

    @abstractmethod
    async def reinitialize(
        self,
        model: str | None = None,
        preserve_sessions: bool = True,
    ) -> None:
        """Reinitialize the workflow manager with new configuration.

        Use this method when settings change (e.g., model switch) to
        properly recreate agents and runners.

        Args:
            model: Optional new model to use.
            preserve_sessions: If True, keeps existing session data.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up workflow manager resources.

        Release resources and reset state. Should be called before
        shutting down or when reinitializing with new settings.
        """
        pass

    # User input handling — callback-only

    async def request_user_input(self, request: UserInputRequest) -> str:
        """Request user input from the CLI via callback.

        Called by tools that need user interaction. Requires
        ``set_input_callback()`` to be set by the consumer (e.g.
        MessageProcessor) before any tool invokes this method. The callback is
        resolved from the *current context*, so a tool always reaches the
        consumer that started its turn.

        Args:
            request: The user input request.

        Returns:
            User's response string.

        Raises:
            RuntimeError: If no callback is registered.
        """
        logger.debug(
            "user_input_requested",
            request_id=request.request_id,
            tool_name=request.tool_name,
        )

        callback = self._user_input_callback.get()
        if callback is None:
            raise RuntimeError(
                "No user input callback registered. "
                "Call set_input_callback() before invoking tools that require user input."
            )

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

    def update_settings(self, settings: "BaseSettings") -> None:
        """Update the settings instance.

        Note: This only updates the settings reference. To apply changes
        that affect agent behavior, call reinitialize().

        Args:
            settings: New settings instance.
        """
        self._settings = settings

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
            session_id=session_id or getattr(self, "session_id", "default_session"),
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
        continues it. A backend without durable sessions has nothing to resume,
        so it adopts the id and returns False.

        Args:
            session_id: Session to adopt.
            user_id: Owner to look the session up as (defaults to
                ``settings.default_user``).
        """
        if hasattr(self, "session_id"):
            self.session_id = session_id
        if not self.supports_sessions:
            logger.info(
                "session_adopted", session_id=session_id, resumed=False,
                backend_sessions=False,
            )
            return False
        if self._is_default_user(user_id):
            exists = await self.session_exists(session_id)
        else:
            exists = await self.session_exists(session_id, user_id=user_id)
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

    async def list_sessions(self, *, user_id: str | None = None) -> list[dict]:
        """List persisted sessions from the native store (most recent first)."""
        raise self._no_session_support("list_sessions")

    async def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        """Delete a persisted session from the native store."""
        raise self._no_session_support("delete_session")


