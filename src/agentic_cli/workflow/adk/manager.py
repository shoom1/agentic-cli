"""Google ADK Workflow Manager for agentic CLI applications.

This module provides a complete, config-based GoogleADKWorkflowManager that handles
agent orchestration, session management, and event streaming using Google ADK.

Domain applications define agents using AgentConfig and pass them to GoogleADKWorkflowManager.

For alternative orchestration backends (e.g., LangGraph), see the base_manager module.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Any, Callable

from google.genai import types
from google.adk import Runner
from google.adk.agents import LlmAgent, Agent
from google.adk.planners import BuiltInPlanner
from google.adk.sessions import InMemorySessionService, BaseSessionService, Session

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.adk.event_processor import ADKEventProcessor
from agentic_cli.workflow.adk.llm_event_logger import LLMEventLogger
from agentic_cli.config import (
    BaseSettings,
    get_settings,
    validate_settings,
    SettingsValidationError,
)
from agentic_cli.logging import Loggers, bind_context

logger = Loggers.workflow()

# Suppress Google GenAI SDK warning about non-text parts (function_call) in
# mixed responses. We already handle all part types individually in process_part.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)


class ADKSummarizer:
    """LLM Summarizer implementation using Google ADK.

    Uses the Gemini API directly (outside the agent loop) to summarize
    web content for the webfetch tool.
    """

    def __init__(self, manager: "GoogleADKWorkflowManager") -> None:
        """Initialize the summarizer.

        Args:
            manager: The workflow manager to use for LLM calls.
        """
        self._manager = manager

    async def summarize(self, content: str, prompt: str) -> str:
        """Summarize content using Gemini.

        Args:
            content: The content to summarize (markdown).
            prompt: The full summarization prompt.

        Returns:
            Summarized text response.
        """
        # Use the manager's generate_simple method
        return await self._manager.generate_simple(prompt, max_tokens=2000)


class GoogleADKWorkflowManager(BaseWorkflowManager):
    """Config-based workflow manager for agentic applications using Google ADK.

    This manager handles all the infrastructure for agent orchestration:
    - Agent creation from declarative configs
    - Lazy initialization of services
    - Session management
    - Event streaming with thinking detection

    Domain applications only need to:
    1. Define agent configs (name, prompt, tools)
    2. Create GoogleADKWorkflowManager with those configs

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

        manager = GoogleADKWorkflowManager(agent_configs=configs, settings=my_settings)

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

        # Lazy-initialized ADK components
        self._session_service: BaseSessionService | None = None
        self._root_agent: Agent | None = None
        self._runner: Runner | None = None

        # Event processor (model set lazily via property)
        self._event_processor = ADKEventProcessor(
            model=model or "",
            on_event=on_event,
        )

        # LLM event logger (initialized lazily when raw_llm_logging is enabled)
        self._llm_event_logger: LLMEventLogger | None = None

        logger.debug(
            "workflow_manager_created",
            app_name=self.app_name,
            model_override=model,
            agent_count=len(agent_configs),
        )

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
        self._initialized = False

        # Clear LLM event logger
        if self._llm_event_logger:
            self._llm_event_logger.clear()
            self._llm_event_logger = None

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

        # Gemini 3 Pro only supports LOW and HIGH thinking levels
        # Gemini 3 Flash supports MINIMAL, LOW, MEDIUM, HIGH
        is_gemini_3_pro = "gemini-3" in self.model and "pro" in self.model

        thinking_level = None
        if thinking_effort == "low":
            thinking_level = types.ThinkingLevel.LOW
        elif thinking_effort == "medium":
            if is_gemini_3_pro:
                # Gemini 3 Pro doesn't support MEDIUM, fall back to HIGH
                thinking_level = types.ThinkingLevel.HIGH
                logger.debug(
                    "thinking_level_fallback",
                    model=self.model,
                    requested="medium",
                    actual="high",
                    reason="Gemini 3 Pro only supports LOW and HIGH",
                )
            else:
                thinking_level = types.ThinkingLevel.MEDIUM
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

    def _get_generate_content_config(self) -> types.GenerateContentConfig:
        """Build GenerateContentConfig with retry-enabled HTTP options.

        Returns:
            GenerateContentConfig with HttpRetryOptions configured from settings.
        """
        http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                initial_delay=self._settings.retry_initial_delay,
                attempts=self._settings.retry_max_attempts,
                exp_base=self._settings.retry_backoff_factor,
                http_status_codes=[500, 502, 503, 504],  # Don't auto-retry 429
            )
        )
        return types.GenerateContentConfig(http_options=http_options)

    def _create_summarizer(self) -> ADKSummarizer:
        """Create an LLM summarizer for webfetch.

        Returns:
            ADKSummarizer instance that uses Gemini for summarization.
        """
        return ADKSummarizer(self)

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
        generate_config = self._get_generate_content_config()

        # Initialize LLM event logger if raw_llm_logging is enabled
        before_callback = None
        after_callback = None
        if self._settings.raw_llm_logging:
            self._llm_event_logger = LLMEventLogger(
                model_name=self.model,
                app_name=self._settings.app_name,
                include_messages=True,
                include_raw_parts=True,
            )
            before_callback = self._llm_event_logger.before_model_callback
            after_callback = self._llm_event_logger.after_model_callback
            logger.info("llm_event_logging_enabled")

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
                    generate_content_config=generate_config,
                    before_model_callback=before_callback,
                    after_model_callback=after_callback,
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
                    generate_content_config=generate_config,
                    before_model_callback=before_callback,
                    after_model_callback=after_callback,
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

        # Create session service
        if self.session_service_uri:
            logger.debug("using_remote_session_service", uri=self.session_service_uri)
        else:
            self._session_service = InMemorySessionService()
            logger.debug("using_in_memory_session_service")

        # Create agent hierarchy from configs
        self._root_agent = self._create_agents()

        # Create runner
        self._runner = Runner(
            app_name=self.app_name,
            agent=self._root_agent,
            session_service=self._session_service,
        )

        # Initialize feature managers based on tool requirements
        self._ensure_managers_initialized()

        self._initialized = True
        logger.info(
            "services_initialized",
            model=self.model,
            agent_name=self._root_agent.name if self._root_agent else None,
            required_managers=list(self._required_managers),
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

    # -------------------------------------------------------------------------
    # Session handling (inlined from SessionHandler)
    # -------------------------------------------------------------------------

    async def _get_or_create_session(
        self,
        user_id: str,
        session_id: str,
    ) -> Session:
        """Get existing session or create a new one.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            The session (existing or newly created)
        """
        session = await self._session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        if session is None:
            session = await self._session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )
            logger.debug("session_created", session_id=session_id)
        else:
            logger.debug("session_resumed", session_id=session_id)

        return session

    # -------------------------------------------------------------------------
    # Main processing
    # -------------------------------------------------------------------------

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

        # Sync event processor model (may have been lazily resolved)
        self._event_processor.model = self.model

        # Context setup
        with self._workflow_context():
            # Session handling
            session = await self._get_or_create_session(user_id, current_session_id)

            # Create message
            new_message = self._create_message(message)

            event_count = 0

            # Build run_config with context window compression if enabled
            run_config = None
            if self._settings.context_window_enabled:
                from google.genai.types import ContextWindowCompressionConfig, SlidingWindow
                from google.adk.agents import RunConfig

                run_config = RunConfig(
                    context_window_compression=ContextWindowCompressionConfig(
                        trigger_tokens=self._settings.context_window_trigger_tokens,
                        sliding_window=SlidingWindow(
                            target_tokens=self._settings.context_window_target_tokens,
                        ),
                    )
                )

            # Process ADK events directly - retry is handled by HttpRetryOptions
            async for adk_event in self._runner.run_async(
                session_id=current_session_id,
                user_id=user_id,
                new_message=new_message,
                run_config=run_config,
            ):
                # Yield LLM events from logger first (Option A - raw capture)
                if self._llm_event_logger:
                    for llm_event in self._llm_event_logger.drain_events():
                        # Apply optional event hook
                        if self._on_event:
                            llm_event = self._on_event(llm_event)
                        if llm_event:
                            event_count += 1
                            yield llm_event

                # Process ADK event into workflow events
                async for workflow_event in self._event_processor.process_event(
                    adk_event,
                    current_session_id,
                ):
                    event_count += 1
                    yield workflow_event

                    # Emit task progress after tool results
                    if workflow_event.type == EventType.TOOL_RESULT:
                        progress_event = self._emit_task_progress_event()
                        if progress_event:
                            if self._on_event:
                                progress_event = self._on_event(progress_event)
                            if progress_event:
                                event_count += 1
                                yield progress_event

            # Drain any remaining LLM events after processing completes
            if self._llm_event_logger:
                for llm_event in self._llm_event_logger.drain_events():
                    if self._on_event:
                        llm_event = self._on_event(llm_event)
                    if llm_event:
                        event_count += 1
                        yield llm_event

            logger.info("message_processed", event_count=event_count)
