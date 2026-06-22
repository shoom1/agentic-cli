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

import contextlib
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Awaitable, Callable, Iterator, TYPE_CHECKING

from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.models import ModelRegistry
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
        self._on_event = on_event

        # Model resolution (lazy)
        self._model: str | None = model
        self._model_resolved: bool = model is not None

        # User input handling (callback-only)
        self._user_input_callback: Callable[[UserInputRequest], Awaitable[str]] | None = None

        # Active turn — set per process() call via _workflow_context(); read by
        # JobManager to associate a long-running job with the session/user that
        # launched it (phase 2 push/resume). None when no turn is in flight.
        self._active_session_id: str | None = None
        self._active_user_id: str | None = None

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
    ) -> None:
        """Register a callback for handling user input requests from tools."""
        self._user_input_callback = callback

    def clear_input_callback(self) -> None:
        """Remove the registered user input callback."""
        self._user_input_callback = None

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
        """
        if service_map is None:
            service_map = self._get_service_tool_map()

        result = []
        for tool in config.tools or []:
            name = getattr(tool, "__name__", "")
            if name in service_map:
                result.append(service_map[name])
            else:
                result.append(tool)

        if config.include_state_tools:
            result.extend(self._get_state_tools())

        return result

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
            tool_map["sandbox_execute"] = make_sandbox_tool(s[SANDBOX_MANAGER])
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

        return tool_map

    @abstractmethod
    def _get_state_tools(self) -> list[Callable]:
        """Return backend-specific state tools (plan/task management).

        Subclasses must override to return ADK or LangGraph native tools.
        """
        ...

    # Mapping from tool function name to the service(s) it requires.
    # Value may be a single service key or a tuple of keys for tools
    # that compose multiple services.
    _TOOL_SERVICE_MAP: dict[str, str | tuple[str, ...]] = {
        "save_memory": "memory_store",
        "search_memory": "memory_store",
        "update_memory": "memory_store",
        "delete_memory": "memory_store",
        "kb_search": "kb_manager",
        "kb_ingest_text": "kb_manager",
        "kb_ingest_file": "kb_manager",
        "kb_ingest_url": "kb_manager",
        "kb_read": "kb_manager",
        "kb_list": "kb_manager",
        "kb_write_concept": "kb_manager",
        "kb_search_concepts": "kb_manager",
        "web_fetch": "llm_summarizer",
        "sandbox_execute": "sandbox_manager",
        "search_arxiv": "arxiv_source",
        "fetch_arxiv_paper": "arxiv_source",
        "ingest_arxiv_paper": ("arxiv_source", "kb_manager"),
        # Long-running jobs: the observe-only tools need the JobManager service.
        # ``run_shell_job`` is an application-provided typed starter (see
        # examples/jobs_demo.py), not a framework tool — its name is mapped here
        # by convention so an app can add it without also adding observe tools.
        "run_shell_job": "job_manager",
        "job_status": "job_manager",
        "job_result": "job_manager",
        "job_logs": "job_manager",
        "job_cancel": "job_manager",
        "job_list": "job_manager",
    }

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
        """Detect which services are needed by scanning tool names.

        Returns:
            Set of required service keys (e.g. ``{"kb_manager", "memory_store"}``).
        """
        required: set[str] = set()
        for config in self._agent_configs:
            for tool in config.tools or []:
                name = getattr(tool, "__name__", "")
                service = self._TOOL_SERVICE_MAP.get(name)
                if service is None:
                    continue
                if isinstance(service, tuple):
                    required.update(service)
                else:
                    required.add(service)
        return required

    def _ensure_managers_initialized(self) -> None:
        """Create managers based on detected requirements.

        Called during initialize_services() to lazily create only the
        managers that are actually needed by the configured tools.
        Populates ``self._services`` which is exposed to tools via
        the service registry ContextVar.
        """
        s = self._services

        if "memory_store" in self._required_managers and MEMORY_STORE not in s:
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

        if "kb_manager" in self._required_managers and KB_MANAGER not in s:
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

        if "llm_summarizer" in self._required_managers and LLM_SUMMARIZER not in s:
            s[LLM_SUMMARIZER] = self

        if "sandbox_manager" in self._required_managers and SANDBOX_MANAGER not in s:
            from agentic_cli.tools.sandbox.manager import SandboxManager
            s[SANDBOX_MANAGER] = SandboxManager(self._settings)

        if "job_manager" in self._required_managers and JOB_MANAGER not in s:
            from pathlib import Path
            from agentic_cli.tools.jobs import JobManager

            # User-scoped so long jobs persist across projects and CLI restarts.
            jobs_dir = Path.home() / f".{self._settings.app_name}" / "jobs"
            s[JOB_MANAGER] = JobManager(
                self._settings,
                base_dir=jobs_dir,
                max_concurrent=getattr(self._settings, "max_concurrent_jobs", 4),
            )

        if "arxiv_source" in self._required_managers and ARXIV_SOURCE not in s:
            from agentic_cli.tools.arxiv_source import ArxivSearchSource
            s[ARXIV_SOURCE] = ArxivSearchSource()

        # Always construct the PermissionEngine (all agents may need it)
        if PERMISSION_ENGINE not in s:
            from pathlib import Path
            from agentic_cli.workflow.permissions import PermissionContext, PermissionEngine
            ctx = PermissionContext(workdir=Path.cwd(), home=Path.home())
            s[PERMISSION_ENGINE] = PermissionEngine(
                settings=self._settings, workflow=self, ctx=ctx,
            )

        # Always ensure workflow reference is available
        s[WORKFLOW] = self

    async def summarize(self, content: str, prompt: str) -> str:
        """Summarize content using the configured LLM.
        Args:
            content: The content to summarize (included in prompt by caller).
            prompt: The full summarization prompt.
        Returns:
            Summarized text response.
        """
        return await self.generate_simple(prompt, max_tokens=12000)

    async def on_session_end(self, messages: list[dict] | None = None) -> list[str]:
        """Hook called when a session ends. Optionally extracts facts.

        Override in downstream apps for custom session-end behavior.

        Args:
            messages: Recent messages from the session (optional).

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
            sid = getattr(self, "session_id", "default_session")
            try:
                messages = await self.recent_messages(sid)
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
    def active_session_id(self) -> str | None:
        """Session id of the in-flight ``process()`` call, or None when idle."""
        return self._active_session_id

    @property
    def active_user_id(self) -> str | None:
        """User id of the in-flight ``process()`` call, or None when idle."""
        return self._active_user_id

    async def can_resume(self, record) -> bool:
        """Whether a finished job can be resumed into its conversation now.

        Base default: False (no resume support). Backends that implement
        ``resume_with_job_result`` override this to report whether the
        originating conversation is still available — e.g. the ADK session that
        holds the pending call. Used by the harness to resume vs. surface a
        "finished while its conversation was unavailable" notice (after a CLI
        restart the default in-memory session is gone).
        """
        return False

    @contextlib.contextmanager
    def _workflow_context(
        self, session_id: str | None = None, user_id: str | None = None
    ) -> Iterator[None]:
        """Context manager that exposes the service registry to tools.

        Sets a single ContextVar (the service registry) so tools can
        call ``get_service(key)`` during execution, and records the active
        session/user for the duration of the turn so JobManager can associate
        a launched job with the conversation that started it.
        """
        from agentic_cli.config import set_context_settings

        settings_token = set_context_settings(self._settings)
        registry_token = set_service_registry(self._services)
        self._active_session_id = session_id
        self._active_user_id = user_id
        try:
            yield
        finally:
            self._active_session_id = None
            self._active_user_id = None
            registry_token.var.reset(registry_token)
            settings_token.var.reset(settings_token)

    def _cleanup_managers(self) -> None:
        """Clean up all manager resources (call from subclass cleanup)."""
        sandbox = self._services.get(SANDBOX_MANAGER)
        if sandbox is not None:
            sandbox.cleanup()
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
        Args:
            validate: If True, validate settings before initialization.
        Raises:
            SettingsValidationError: If settings validation fails.
        """
        if self._initialized:
            return

        from agentic_cli.config import validate_settings

        if validate:
            validate_settings(self._settings)

        self._settings.export_api_keys_to_env()

        # Refresh model registry from APIs
        await self._model_registry.refresh(
            google_api_key=self._settings.google_api_key,
            anthropic_api_key=self._settings.anthropic_api_key,
        )
        self._settings.set_model_registry(self._model_registry)

        # Create services BEFORE backend init so _build_tools() can
        # produce factory-bound tools during agent/graph creation.
        # Offloaded to a worker thread because constructors here may
        # load heavy dependencies (e.g. the sentence-transformers model
        # inside EmbeddingService) that would otherwise block the event
        # loop — which keeps the prompt unresponsive at startup.
        import asyncio as _asyncio

        await _asyncio.to_thread(self._ensure_managers_initialized)
        await self._do_initialize()
        self._initialized = True

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
        MessageProcessor) before any tool invokes this method.

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

        if self._user_input_callback is None:
            raise RuntimeError(
                "No user input callback registered. "
                "Call set_input_callback() before invoking tools that require user input."
            )

        return await self._user_input_callback(request)

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

    async def save_session(self, session_id: str | None = None) -> dict:
        """No-op flush — durable stores persist as the turn runs.

        Kept for API compatibility / explicit "checkpoint now" intent. Returns
        the session id that is (already) persisted.
        """
        sid = session_id or getattr(self, "session_id", "default_session")
        return {"success": True, "session_id": sid}

    async def load_session(self, session_id: str) -> bool:
        """Adopt ``session_id`` for resume; the native store already holds it.

        Returns True if that session already has content (i.e. a real resume),
        False if it's new — but the id is adopted either way so the next turn
        continues it.
        """
        if hasattr(self, "session_id"):
            self.session_id = session_id
        exists = await self.session_exists(session_id)
        logger.info("session_adopted", session_id=session_id, resumed=exists)
        return exists

    # ---- Backend hooks (override in ADK / LangGraph managers) ----

    async def session_exists(self, session_id: str) -> bool:
        """Whether the native store already holds this session's state."""
        return False

    async def recent_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        """Recent ``{role, content}`` text messages from the native session.

        Used for session-end fact extraction; text-only (no tool-call fidelity).
        """
        return []

    async def list_sessions(self) -> list[dict]:
        """List persisted sessions from the native store (most recent first)."""
        return []

    async def delete_session(self, session_id: str) -> bool:
        """Delete a persisted session from the native store."""
        return False


