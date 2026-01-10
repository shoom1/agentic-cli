"""Workflow Manager for agentic CLI applications.

This module provides a complete, config-based WorkflowManager that handles
agent orchestration, session management, and event streaming.

Domain applications define agents using AgentConfig and pass them to WorkflowManager.
"""

from typing import AsyncGenerator, Any, Callable

from google.genai import types
from google.adk import Runner
from google.adk.agents import LlmAgent, Agent
from google.adk.planners import BuiltInPlanner
from google.adk.sessions import InMemorySessionService, BaseSessionService

from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.config import (
    BaseSettings,
    get_settings,
    set_context_settings,
    validate_settings,
    SettingsValidationError,
)
from agentic_cli.logging import Loggers, bind_context

logger = Loggers.workflow()


class WorkflowManager:
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
        self._agent_configs = agent_configs
        self._settings = settings or get_settings()
        self._on_event = on_event

        self.session_service_uri = session_service_uri
        self.app_name = app_name or self._settings.app_name
        self.session_id = "default_session"

        # Model is resolved lazily to allow startup without API keys
        self._model: str | None = model
        self._model_resolved: bool = model is not None

        # Lazy-initialized components
        self._session_service: BaseSessionService | None = None
        self._root_agent: Agent | None = None
        self._runner: Runner | None = None
        self._initialized: bool = False

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
    def is_initialized(self) -> bool:
        """Check if services have been initialized."""
        return self._initialized

    @property
    def settings(self) -> BaseSettings:
        """Get the settings instance."""
        return self._settings

    async def __aenter__(self) -> "WorkflowManager":
        """Async context manager entry - initialize services."""
        await self.initialize_services()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup resources."""
        await self.cleanup()

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

        if not self._runner or not self._session_service or not self._root_agent:
            raise RuntimeError(
                "Workflow Manager failed to initialize. Check API keys and configuration."
            )

        current_session_id = session_id or self.session_id

        bind_context(session_id=current_session_id, user_id=user_id)

        logger.info("processing_message", message_length=len(message))

        # Set context settings so tools use this manager's settings
        set_context_settings(self._settings)

        try:
            # Get or create session
            session = await self._session_service.get_session(
                app_name=self.app_name, user_id=user_id, session_id=current_session_id
            )
            if session is None:
                session = await self._session_service.create_session(
                    app_name=self.app_name, user_id=user_id, session_id=current_session_id
                )
                logger.debug("session_created", session_id=current_session_id)
            else:
                logger.debug("session_resumed", session_id=current_session_id)

            new_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )

            events_async = self._runner.run_async(
                session_id=current_session_id,
                user_id=user_id,
                new_message=new_message,
            )

            event_count = 0
            async for event in events_async:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        event_count += 1
                        workflow_event = self._process_part(part, current_session_id)
                        if workflow_event:
                            # Apply optional event hook
                            if self._on_event:
                                workflow_event = self._on_event(workflow_event)
                            if workflow_event:
                                yield workflow_event

            logger.info("message_processed", event_count=event_count)
        finally:
            # Clear context settings after processing
            set_context_settings(None)

    def _process_part(self, part: Any, session_id: str) -> WorkflowEvent | None:
        """Process a single part from the agent response."""
        from agentic_cli.llm.thinking import ThinkingDetector

        if part.text:
            result = ThinkingDetector.detect_from_part(part)
            if result.is_thinking:
                return WorkflowEvent.thinking(result.content, session_id)
            return WorkflowEvent.text(result.content, session_id)

        if part.function_call:
            logger.debug("tool_call", tool_name=part.function_call.name)
            return WorkflowEvent.tool_call(part.function_call.name)

        if part.code_execution_result:
            return WorkflowEvent.code_execution(str(part.code_execution_result.outcome))

        if part.executable_code:
            return WorkflowEvent.executable_code(
                part.executable_code.code,
                str(part.executable_code.language),
            )

        if part.file_data:
            return WorkflowEvent.file_data(part.file_data.display_name or "unknown")

        return None
