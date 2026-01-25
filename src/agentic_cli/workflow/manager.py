"""Workflow Manager for agentic CLI applications using Google ADK.

This module provides a complete, config-based WorkflowManager that handles
agent orchestration, session management, and event streaming using Google ADK.

Domain applications define agents using AgentConfig and pass them to WorkflowManager.

For alternative orchestration backends (e.g., LangGraph), see the base_manager module.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncGenerator, Any, Callable, Iterator

from google.genai import types
from google.adk import Runner
from google.adk.agents import LlmAgent, Agent
from google.adk.planners import BuiltInPlanner
from google.adk.sessions import InMemorySessionService, BaseSessionService

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.memory import ConversationMemory, create_summarizer
from agentic_cli.workflow.retry import RetryConfig, RetryHandler
from agentic_cli.workflow.session_handler import SessionHandler
from agentic_cli.workflow.event_processor import EventProcessor
from agentic_cli.config import (
    BaseSettings,
    get_settings,
    set_context_settings,
    set_context_workflow,
    validate_settings,
    SettingsValidationError,
)
from agentic_cli.logging import Loggers, bind_context

logger = Loggers.workflow()


class WorkflowManager(BaseWorkflowManager):
    """Config-based workflow manager for agentic applications.

    This manager handles all the infrastructure for agent orchestration:
    - Agent creation from declarative configs
    - Lazy initialization of services
    - Session management
    - Event streaming with thinking detection

    Domain applications only need to:
    1. Define agent configs (name, prompt, tools)
    2. Create WorkflowManager with those configs

    Example:
        configs = [
            AgentConfig(
                name="coordinator",
                prompt=get_coordinator_prompt,
                tools=[search_kb, ask_user],
                sub_agents=["specialist_a", "specialist_b"],
            ),
            AgentConfig(name="specialist_a", prompt=get_a_prompt, tools=[...]),
            AgentConfig(name="specialist_b", prompt=get_b_prompt, tools=[...]),
        ]

        manager = WorkflowManager(agent_configs=configs, settings=my_settings)

        async for event in manager.process("Hello", user_id="user1"):
            print(event.content)
    """

    def __init__(
        self,
        agent_configs: list[AgentConfig],
        settings: BaseSettings | None = None,
        app_name: str | None = None,
        model: str | None = None,
        session_service_uri: str | None = None,
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
    ) -> None:
        """Initialize the workflow manager.

        Args:
            agent_configs: List of agent configurations. First agent with sub_agents
                          is treated as the root/coordinator agent.
            settings: Application settings (uses get_settings() if not provided)
            app_name: Application name for services (uses settings.app_name if not provided)
            model: Model override (auto-detected from API keys if not provided)
            session_service_uri: Optional URI for remote session service
            on_event: Optional hook to transform/filter events before yielding
        """
        super().__init__(
            agent_configs=agent_configs,
            settings=settings,
            app_name=app_name,
            model=model,
        )

        self._on_event = on_event
        self.session_service_uri = session_service_uri
        self.session_id = "default_session"

        # Model is resolved lazily to allow startup without API keys
        self._model: str | None = model
        self._model_resolved: bool = model is not None

        # Lazy-initialized ADK components
        self._session_service: BaseSessionService | None = None
        self._session_handler: SessionHandler | None = None
        self._root_agent: Agent | None = None
        self._runner: Runner | None = None

        # Event processor for transforming ADK events
        self._event_processor = EventProcessor(on_event=on_event)

        # Retry handler for transient errors
        self._retry_handler = RetryHandler(RetryConfig())

        # User input request handling
        self._pending_input: dict[str, tuple[UserInputRequest, asyncio.Future[str]]] = {}

        # Conversation memory with auto-summarization
        self._memory = ConversationMemory()

        logger.debug(
            "workflow_manager_created",
            app_name=self.app_name,
            model_override=model,
            agent_count=len(agent_configs),
        )

    @property
    def model(self) -> str:
        """Get the model name, resolving from settings if needed."""
        if not self._model_resolved:
            self._model = self._settings.get_model()
            self._model_resolved = True
            logger.info("model_resolved", model=self._model)
        return self._model  # type: ignore[return-value]

    @property
    def session_service(self) -> BaseSessionService | None:
        """Get the session service."""
        return self._session_service

    @property
    def root_agent(self) -> Agent | None:
        """Get the root agent."""
        return self._root_agent

    @property
    def runner(self) -> Runner | None:
        """Get the runner."""
        return self._runner

    @property
    def memory(self) -> ConversationMemory:
        """Get the conversation memory."""
        return self._memory

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

        Called by tools that need user interaction. This method blocks
        until provide_user_input() is called with the response.

        Args:
            request: The user input request

        Returns:
            User's response string
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_input[request.request_id] = (request, future)

        logger.debug(
            "user_input_requested",
            request_id=request.request_id,
            tool_name=request.tool_name,
        )

        return await future

    def provide_user_input(self, request_id: str, response: str) -> bool:
        """Provide user input for a pending request.

        Called by CLI when user responds to a USER_INPUT_REQUIRED event.

        Args:
            request_id: The request ID from the event metadata
            response: User's response

        Returns:
            True if request was found and resolved, False otherwise
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

    async def generate_simple(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a simple text response using the current model.

        Used for internal operations like summarization. Does not go through
        the full agent workflow.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        await self._ensure_initialized()

        from google import genai

        client = genai.Client()
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
            ),
        )

        return response.text or ""

    async def cleanup(self) -> None:
        """Clean up workflow manager resources.

        Releases resources and resets state. Call this before
        shutting down or when reinitializing with new settings.
        """
        logger.debug("cleaning_up_workflow_manager")

        # Clear runner and agents
        self._runner = None
        self._root_agent = None
        self._session_service = None
        self._session_handler = None
        self._initialized = False

        # Cancel any pending input requests
        for request_id, (request, future) in self._pending_input.items():
            if not future.done():
                future.cancel()
        self._pending_input.clear()

        logger.info("workflow_manager_cleaned_up")

    async def reinitialize(
        self,
        model: str | None = None,
        preserve_sessions: bool = True,
    ) -> None:
        """Reinitialize the workflow manager with new configuration.

        Use this method when settings change (e.g., model switch) to
        properly recreate agents and runners with the new configuration.

        Args:
            model: Optional new model to use. If None, re-resolves from settings.
            preserve_sessions: If True, keeps existing session data (default).
                             If False, creates fresh session service.
        """
        logger.info(
            "reinitializing_workflow_manager",
            new_model=model,
            preserve_sessions=preserve_sessions,
        )

        # Store session service if preserving
        old_session_service = self._session_service if preserve_sessions else None

        # Clean up current state
        await self.cleanup()

        # Update model if provided
        if model is not None:
            self._model = model
            self._model_resolved = True
        else:
            # Re-resolve model from settings
            self._model = None
            self._model_resolved = False

        # Reinitialize services
        await self.initialize_services()

        # Restore session service if preserving
        if old_session_service is not None and preserve_sessions:
            self._session_service = old_session_service
            self._session_handler = SessionHandler(self._session_service, self.app_name)
            # Update runner with preserved session service
            if self._runner and self._root_agent:
                self._runner = Runner(
                    app_name=self.app_name,
                    agent=self._root_agent,
                    session_service=self._session_service,
                )

        logger.info(
            "workflow_manager_reinitialized",
            model=self.model,
            sessions_preserved=preserve_sessions,
        )

    def update_settings(self, settings: BaseSettings) -> None:
        """Update the settings instance.

        Note: This only updates the settings reference. To apply changes
        that affect agent behavior (like model changes), call reinitialize().

        Args:
            settings: New settings instance
        """
        self._settings = settings
        logger.debug("settings_updated")

    def _get_planner(self) -> BuiltInPlanner | None:
        """Get planner with thinking configuration."""
        thinking_effort = self._settings.thinking_effort

        if thinking_effort == "none":
            return None

        if not self._settings.supports_thinking_effort(self.model):
            logger.debug(
                "thinking_not_supported",
                model=self.model,
                effort=thinking_effort,
            )
            return None

        thinking_level = None
        if thinking_effort == "low":
            thinking_level = types.ThinkingLevel.LOW
        elif thinking_effort == "high":
            thinking_level = types.ThinkingLevel.HIGH

        thinking_config = types.ThinkingConfig(
            include_thoughts=True,
            thinking_level=thinking_level,
        )

        logger.debug(
            "planner_created",
            effort=thinking_effort,
            level=thinking_level,
        )

        return BuiltInPlanner(thinking_config=thinking_config)

    def _create_agents(self) -> Agent:
        """Create agent hierarchy from configs.

        Returns:
            Root agent (the first agent with sub_agents, or first agent if none have sub_agents)
        """
        # Build a map of agent configs by name
        config_map = {config.name: config for config in self._agent_configs}

        # Build agents (non-coordinators first, then coordinators)
        agent_map: dict[str, Agent] = {}
        planner = self._get_planner()

        # First pass: create agents without sub_agents (leaf agents)
        for config in self._agent_configs:
            if not config.sub_agents:
                agent_map[config.name] = LlmAgent(
                    name=config.name,
                    model=config.model or self.model,
                    instruction=config.get_prompt(),
                    tools=config.tools or [],
                    description=config.description or None,
                    planner=planner,
                )
                logger.debug("agent_created", name=config.name, type="leaf")

        # Second pass: create agents with sub_agents (coordinators)
        for config in self._agent_configs:
            if config.sub_agents:
                sub_agent_instances = []
                for sub_name in config.sub_agents:
                    if sub_name in agent_map:
                        sub_agent_instances.append(agent_map[sub_name])
                    else:
                        logger.warning(
                            "sub_agent_not_found",
                            coordinator=config.name,
                            sub_agent=sub_name,
                        )

                agent_map[config.name] = LlmAgent(
                    name=config.name,
                    model=config.model or self.model,
                    instruction=config.get_prompt(),
                    tools=config.tools or [],
                    description=config.description or None,
                    sub_agents=sub_agent_instances,
                    planner=planner,
                )
                logger.debug(
                    "agent_created",
                    name=config.name,
                    type="coordinator",
                    sub_agents=[a.name for a in sub_agent_instances],
                )

        # Find root agent (first with sub_agents, or first in list)
        root_agent = None
        for config in self._agent_configs:
            if config.sub_agents:
                root_agent = agent_map[config.name]
                break

        if root_agent is None and self._agent_configs:
            root_agent = agent_map[self._agent_configs[0].name]

        if root_agent is None:
            raise RuntimeError("No agents configured")

        logger.info("agents_created", root=root_agent.name, total=len(agent_map))
        return root_agent

    async def initialize_services(self, validate: bool = True) -> None:
        """Initialize ADK services asynchronously.

        Args:
            validate: If True, validate settings before initialization.
                     Set to False to skip validation (e.g., for testing).

        Raises:
            SettingsValidationError: If settings validation fails
        """
        if self._initialized:
            logger.debug("services_already_initialized")
            return

        logger.info("initializing_services", app_name=self.app_name)

        # Validate settings early to provide clear error messages
        if validate:
            validate_settings(self._settings)

        # Export API keys to environment
        self._settings.export_api_keys_to_env()

        # Create session service and handler
        if self.session_service_uri:
            logger.debug("using_remote_session_service", uri=self.session_service_uri)
        else:
            self._session_service = InMemorySessionService()
            self._session_handler = SessionHandler(self._session_service, self.app_name)
            logger.debug("using_in_memory_session_service")

        # Create agent hierarchy from configs
        self._root_agent = self._create_agents()

        # Create runner
        self._runner = Runner(
            app_name=self.app_name,
            agent=self._root_agent,
            session_service=self._session_service,
        )

        self._initialized = True
        logger.info(
            "services_initialized",
            model=self.model,
            agent_name=self._root_agent.name if self._root_agent else None,
        )

    async def _ensure_initialized(self) -> None:
        """Ensure services are initialized before processing."""
        if not self._initialized:
            await self.initialize_services()

    def _validate_initialized(self) -> None:
        """Validate that all required components are initialized.

        Raises:
            RuntimeError: If required components are not initialized
        """
        if not self._runner or not self._session_service or not self._root_agent:
            raise RuntimeError(
                "Workflow Manager failed to initialize. Check API keys and configuration."
            )

    @contextlib.contextmanager
    def _workflow_context(self) -> Iterator[None]:
        """Context manager for settings and workflow context.

        Sets context variables that allow tools to access settings
        and the workflow manager during execution.
        """
        set_context_settings(self._settings)
        set_context_workflow(self)
        try:
            yield
        finally:
            set_context_settings(None)
            set_context_workflow(None)

    async def _prepare_memory(self, message: str) -> None:
        """Prepare memory for processing a new message.

        Adds the user message to memory and performs summarization
        if the conversation has grown too long.

        Args:
            message: User message to add
        """
        self._memory.add_message("user", message)

        if self._memory.should_summarize():
            logger.info("summarizing_conversation")
            summarizer = await create_summarizer(self)
            await self._memory.summarize(summarizer)

    def _finalize_memory(self, response_parts: list[str]) -> None:
        """Finalize memory after processing.

        Adds the assistant response to memory.

        Args:
            response_parts: Collected response text parts
        """
        if response_parts:
            full_response = "\n".join(response_parts)
            self._memory.add_message("assistant", full_response)

    def _create_message(self, message: str) -> types.Content:
        """Create an ADK Content message from text.

        Args:
            message: User message text

        Returns:
            ADK Content object
        """
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)],
        )

    async def _on_retry_callback(
        self, attempt: int, delay: float, error: Exception
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Callback invoked when a retry is about to happen.

        This is a factory that returns events to yield during retry.

        Args:
            attempt: Current attempt number (0-indexed)
            delay: Delay before next retry in seconds
            error: The exception that triggered the retry

        Yields:
            WorkflowEvent describing the retry
        """
        from google.api_core import exceptions as google_exceptions

        if isinstance(error, google_exceptions.ResourceExhausted):
            error_code = "RATE_LIMITED"
            message = f"Rate limited, retrying in {delay}s..."
        else:
            error_code = "TRANSIENT_ERROR"
            message = f"Model temporarily unavailable, retrying in {delay}s..."

        return WorkflowEvent.error(message, error_code=error_code, recoverable=True)

    async def process(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Process user input through the agentic workflow.

        This method sets up a settings context so that all tools called
        during processing will use this manager's settings instance.

        Args:
            message: User message
            user_id: User identifier
            session_id: Optional session identifier

        Yields:
            WorkflowEvent objects representing workflow output
        """
        await self._ensure_initialized()
        self._validate_initialized()

        current_session_id = session_id or self.session_id
        bind_context(session_id=current_session_id, user_id=user_id)

        logger.info("processing_message", message_length=len(message))

        # Memory management
        await self._prepare_memory(message)

        # Context setup
        with self._workflow_context():
            # Session handling
            session = await self._session_handler.get_or_create_session(
                user_id, current_session_id
            )

            # Create message
            new_message = self._create_message(message)

            # Track response for memory
            full_response_parts: list[str] = []
            event_count = 0

            # Create the workflow operation
            def create_workflow():
                return self._runner.run_async(
                    session_id=current_session_id,
                    user_id=user_id,
                    new_message=new_message,
                )

            # Process with retry
            async for adk_event in self._retry_handler.execute_with_retry(
                create_workflow,
                on_retry=self._emit_retry_event,
            ):
                # Process ADK event into workflow events
                async for workflow_event in self._event_processor.process_events(
                    adk_event,
                    current_session_id,
                    self.get_pending_input_request,
                ):
                    event_count += 1

                    # Track text responses for memory
                    if workflow_event.type.value == "text":
                        full_response_parts.append(workflow_event.content)

                    yield workflow_event

            # Update memory with assistant response
            self._finalize_memory(full_response_parts)

            logger.info("message_processed", event_count=event_count)

    async def _emit_retry_event(
        self, attempt: int, delay: float, error: Exception
    ) -> None:
        """Emit a retry event (callback for RetryHandler).

        Note: This callback doesn't yield events directly. The retry events
        are handled differently - we log them but don't yield because we're
        in a callback context. The main process loop handles event yielding.

        Args:
            attempt: Current attempt number (0-indexed)
            delay: Delay before next retry in seconds
            error: The exception that triggered the retry
        """
        # Logging is handled by RetryHandler, this is a hook for custom behavior
        pass
