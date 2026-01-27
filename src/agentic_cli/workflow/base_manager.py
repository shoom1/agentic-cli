"""Abstract base class for workflow managers.

This module defines the interface that all workflow orchestration backends
must implement, enabling pluggable orchestrators (ADK, LangGraph, etc.).

The base class provides auto-detection of required managers (memory, planning,
HITL) based on tool requirements, creating them lazily when needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, TYPE_CHECKING

from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest
from agentic_cli.workflow.config import AgentConfig

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.memory import MemoryManager
    from agentic_cli.planning import TaskGraph
    from agentic_cli.hitl import ApprovalManager, CheckpointManager


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
        self._model_override = model
        self._initialized = False

        # Auto-detect required managers from tools
        self._required_managers = self._detect_required_managers()

        # Manager slots (created lazily by _ensure_managers_initialized)
        self._memory_manager: "MemoryManager | None" = None
        self._task_graph: "TaskGraph | None" = None
        self._approval_manager: "ApprovalManager | None" = None
        self._checkpoint_manager: "CheckpointManager | None" = None

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
    def memory_manager(self) -> "MemoryManager | None":
        """Get the memory manager (if required by tools)."""
        return self._memory_manager

    @property
    def task_graph(self) -> "TaskGraph | None":
        """Get the task graph (if required by tools)."""
        return self._task_graph

    @property
    def approval_manager(self) -> "ApprovalManager | None":
        """Get the approval manager (if required by tools)."""
        return self._approval_manager

    @property
    def checkpoint_manager(self) -> "CheckpointManager | None":
        """Get the checkpoint manager (if required by tools)."""
        return self._checkpoint_manager

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
            from agentic_cli.memory import MemoryManager
            self._memory_manager = MemoryManager(self._settings)

        if "task_graph" in self._required_managers and self._task_graph is None:
            from agentic_cli.planning import TaskGraph
            self._task_graph = TaskGraph()

        if "approval_manager" in self._required_managers and self._approval_manager is None:
            from agentic_cli.hitl import ApprovalManager, HITLConfig, ApprovalRule
            # Build config from settings
            hitl_config = HITLConfig(
                approval_rules=[
                    ApprovalRule(**r) for r in getattr(self._settings, "hitl_default_rules", [])
                ],
                checkpoint_enabled=getattr(self._settings, "hitl_checkpoint_enabled", True),
                feedback_enabled=getattr(self._settings, "hitl_feedback_enabled", True),
            )
            self._approval_manager = ApprovalManager(hitl_config)

        if "checkpoint_manager" in self._required_managers and self._checkpoint_manager is None:
            from agentic_cli.hitl import CheckpointManager
            self._checkpoint_manager = CheckpointManager()

    @property
    @abstractmethod
    def model(self) -> str:
        """Get the model name being used.

        Implementations may resolve this lazily from settings.
        """
        pass

    @abstractmethod
    async def initialize_services(self, validate: bool = True) -> None:
        """Initialize backend services asynchronously.

        This method should set up all necessary services for the workflow
        backend (session services, runners, agents, etc.).

        Args:
            validate: If True, validate settings before initialization.

        Raises:
            SettingsValidationError: If settings validation fails.
        """
        pass

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

    # User input handling methods

    @abstractmethod
    def has_pending_input(self) -> bool:
        """Check if there are pending user input requests."""
        pass

    @abstractmethod
    def get_pending_input_request(self) -> UserInputRequest | None:
        """Get the next pending input request without removing it."""
        pass

    @abstractmethod
    async def request_user_input(self, request: UserInputRequest) -> str:
        """Request user input from the CLI.

        Called by tools that need user interaction. This method blocks
        until provide_user_input() is called with the response.

        Args:
            request: The user input request.

        Returns:
            User's response string.
        """
        pass

    @abstractmethod
    def provide_user_input(self, request_id: str, response: str) -> bool:
        """Provide user input for a pending request.

        Called by CLI when user responds to a USER_INPUT_REQUIRED event.

        Args:
            request_id: The request ID from the event metadata.
            response: User's response.

        Returns:
            True if request was found and resolved, False otherwise.
        """
        pass

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
