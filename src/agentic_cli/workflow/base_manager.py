"""Abstract base class for workflow managers.

This module defines the interface that all workflow orchestration backends
must implement, enabling pluggable orchestrators (ADK, LangGraph, etc.).

The base class provides auto-detection of required managers (memory, planning,
HITL) based on tool requirements, creating them lazily when needed.

It also provides shared implementations for:
- User input handling (pending request dict + Future pattern)
- Model resolution (lazy resolution from settings)
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Awaitable, Callable, Iterator, TYPE_CHECKING

from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.config import set_context_settings
from agentic_cli.workflow.context import (
    set_context_workflow,
    set_context_memory_store,
    set_context_plan_store,
    set_context_task_store,
    set_context_kb_manager,
    set_context_user_kb_manager,
    set_context_approval_manager,
    set_context_llm_summarizer,
)
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.tools.memory_tools import MemoryStore
    from agentic_cli.tools.planning_tools import PlanStore
    from agentic_cli.tools.task_tools import TaskStore
    from agentic_cli.knowledge_base import KnowledgeBaseManager
    from agentic_cli.tools.hitl_tools import ApprovalManager

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
    ) -> None:
        """Initialize the workflow manager base.

        Args:
            agent_configs: List of agent configurations defining the workflow.
            settings: Application settings (resolved via get_settings() if None).
            app_name: Application name for services.
            model: Model override (auto-detected from API keys if not provided).
        """
        from agentic_cli.config import get_settings

        self._agent_configs = agent_configs
        self._settings = settings or get_settings()
        self._app_name = app_name or self._settings.app_name
        self._initialized = False

        # Model resolution (lazy)
        self._model: str | None = model
        self._model_resolved: bool = model is not None

        # User input handling
        self._pending_input: dict[str, tuple[UserInputRequest, asyncio.Future[str]]] = {}
        self._user_input_callback: Callable[[UserInputRequest], Awaitable[str]] | None = None

        # Auto-detect required managers from tools
        self._required_managers = self._detect_required_managers()

        # Manager slots (created lazily by _ensure_managers_initialized)
        self._memory_manager: "MemoryStore | None" = None
        self._plan_store: "PlanStore | None" = None
        self._task_store: "TaskStore | None" = None
        self._kb_manager: "KnowledgeBaseManager | None" = None
        self._user_kb_manager: "KnowledgeBaseManager | None" = None
        self._approval_manager: "ApprovalManager | None" = None
        self._llm_summarizer: Any | None = None

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
    def required_managers(self) -> set[str]:
        """Get the set of required manager types detected from tools."""
        return self._required_managers

    @property
    def memory_manager(self) -> "MemoryStore | None":
        """Get the memory manager (if required by tools)."""
        return self._memory_manager

    @property
    def plan_store(self) -> "PlanStore | None":
        """Get the plan store (if required by tools)."""
        return self._plan_store

    @property
    def task_store(self) -> "TaskStore | None":
        """Get the task store (if required by tools)."""
        return self._task_store

    @property
    def kb_manager(self) -> "KnowledgeBaseManager | None":
        """Get the project-scoped knowledge base manager (if required by tools)."""
        return self._kb_manager

    @property
    def user_kb_manager(self) -> "KnowledgeBaseManager | None":
        """Get the user-scoped knowledge base manager (if required by tools)."""
        return self._user_kb_manager

    @property
    def approval_manager(self) -> "ApprovalManager | None":
        """Get the approval manager (if required by tools)."""
        return self._approval_manager

    @property
    def llm_summarizer(self) -> Any | None:
        """Get the LLM summarizer (if required by tools)."""
        return self._llm_summarizer

    def _detect_required_managers(self) -> set[str]:
        """Scan all agent tools for 'requires' metadata.

        Tools decorated with @requires("memory_manager") etc. will have
        their requirements detected here.

        Returns:
            Set of required manager types.
        """
        required: set[str] = set()
        for config in self._agent_configs:
            for tool in config.tools or []:
                if hasattr(tool, "requires"):
                    required.update(tool.requires)
        return required

    def _ensure_managers_initialized(self) -> None:
        """Create managers based on detected requirements.

        Called during initialize_services() to lazily create only the
        managers that are actually needed by the configured tools.
        """
        if "memory_manager" in self._required_managers and self._memory_manager is None:
            from agentic_cli.tools.memory_tools import MemoryStore
            self._memory_manager = MemoryStore(self._settings)

        if "plan_store" in self._required_managers and self._plan_store is None:
            from agentic_cli.tools.planning_tools import PlanStore
            self._plan_store = PlanStore()

        if "task_store" in self._required_managers and self._task_store is None:
            from agentic_cli.tools.task_tools import TaskStore
            self._task_store = TaskStore(self._settings)

        if "kb_manager" in self._required_managers and self._kb_manager is None:
            from pathlib import Path
            from agentic_cli.knowledge_base import KnowledgeBaseManager

            use_mock = self._settings.knowledge_base_use_mock
            project_kb_dir = Path.cwd() / f".{self._settings.app_name}" / "knowledge_base"
            user_kb_dir = self._settings.knowledge_base_dir

            # Project KB (agent read-write)
            self._kb_manager = KnowledgeBaseManager(
                settings=self._settings,
                use_mock=use_mock,
                base_dir=project_kb_dir,
            )

            # User KB (agent read-only) — reuse project instance if paths overlap
            if project_kb_dir.resolve() != user_kb_dir.resolve():
                self._user_kb_manager = KnowledgeBaseManager(
                    settings=self._settings,
                    use_mock=use_mock,
                    base_dir=user_kb_dir,
                )
            else:
                self._user_kb_manager = self._kb_manager

        if "approval_manager" in self._required_managers and self._approval_manager is None:
            from agentic_cli.tools.hitl_tools import ApprovalManager
            self._approval_manager = ApprovalManager()

        if "llm_summarizer" in self._required_managers and self._llm_summarizer is None:
            self._llm_summarizer = self._create_summarizer()

    def _create_summarizer(self) -> Any:
        """Create an LLM summarizer for webfetch.

        Subclasses should override this to return a framework-specific
        summarizer that implements the LLMSummarizer protocol.

        Returns:
            An LLMSummarizer implementation, or None.
        """
        return None

    @contextlib.contextmanager
    def _workflow_context(self) -> Iterator[None]:
        """Context manager for settings, workflow, and manager contexts.

        Sets context variables that allow tools to access settings,
        the workflow manager, and feature managers during execution.
        Uses token-based reset to correctly restore parent context.
        """
        tokens = [
            set_context_settings(self._settings),
            set_context_workflow(self),
            set_context_memory_store(self._memory_manager),
            set_context_plan_store(self._plan_store),
            set_context_task_store(self._task_store),
            set_context_kb_manager(self._kb_manager),
            set_context_user_kb_manager(self._user_kb_manager),
            set_context_approval_manager(self._approval_manager),
            set_context_llm_summarizer(self._llm_summarizer),
        ]
        try:
            yield
        finally:
            for token in tokens:
                token.var.reset(token)

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

        Template method that handles shared scaffolding (guard, validation,
        key export, manager init, flag) and delegates backend-specific work
        to :meth:`_do_initialize`.

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
        await self._do_initialize()
        self._ensure_managers_initialized()
        self._initialized = True

    @abstractmethod
    async def _do_initialize(self) -> None:
        """Backend-specific initialization (create agents/graph).

        Subclasses implement this instead of ``initialize_services()``.
        """
        ...

    def _cancel_pending_inputs(self) -> None:
        """Cancel all pending user input requests."""
        for request_id, (request, future) in self._pending_input.items():
            if not future.done():
                future.cancel()
        self._pending_input.clear()

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

    # User input handling methods - concrete implementations

    def has_pending_input(self) -> bool:
        """Check if there are pending user input requests."""
        return len(self._pending_input) > 0

    def get_pending_input_request(self) -> UserInputRequest | None:
        """Get the next pending input request without removing it."""
        if self._pending_input:
            request_id = next(iter(self._pending_input))
            return self._pending_input[request_id][0]
        return None

    async def request_user_input(self, request: UserInputRequest) -> str:
        """Request user input from the CLI.

        Called by tools that need user interaction.

        When ``_user_input_callback`` is set (by MessageProcessor), the
        callback is invoked directly, which avoids the deadlock inherent
        in the Future pattern (the Future can never be resolved while the
        ADK runner is blocked awaiting it).

        When no callback is set (e.g. in tests), falls back to the
        original Future pattern for backward compatibility.

        Args:
            request: The user input request.

        Returns:
            User's response string.
        """
        logger.debug(
            "user_input_requested",
            request_id=request.request_id,
            tool_name=request.tool_name,
        )

        if self._user_input_callback is not None:
            return await self._user_input_callback(request)

        # Fallback: Future pattern (for tests without callback)
        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_input[request.request_id] = (request, future)
        return await future

    def provide_user_input(self, request_id: str, response: str) -> bool:
        """Provide user input for a pending request.

        Called by CLI when user responds to a USER_INPUT_REQUIRED event.

        Args:
            request_id: The request ID from the event metadata.
            response: User's response.

        Returns:
            True if request was found and resolved, False otherwise.
        """
        if request_id not in self._pending_input:
            logger.warning("unknown_input_request", request_id=request_id)
            return False

        request, future = self._pending_input.pop(request_id)
        future.set_result(response)

        logger.debug(
            "user_input_provided",
            request_id=request_id,
            tool_name=request.tool_name,
        )
        return True

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

    def _emit_task_progress_event(self) -> WorkflowEvent | None:
        """Build a TASK_PROGRESS event from TaskStore or PlanStore.

        Delegates to :func:`~agentic_cli.workflow.task_progress.build_task_progress_event`.
        """
        from agentic_cli.workflow.task_progress import build_task_progress_event

        return build_task_progress_event(self._task_store, self._plan_store)

