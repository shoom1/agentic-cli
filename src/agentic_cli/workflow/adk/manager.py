"""Google ADK Workflow Manager for agentic CLI applications.

This module provides a complete, config-based GoogleADKWorkflowManager that handles
agent orchestration, session management, and event streaming using Google ADK.

Domain applications define agents using AgentConfig and pass them to GoogleADKWorkflowManager.

For alternative orchestration backends (e.g., LangGraph), see the base_manager module.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Any, Callable

from google.genai import types
from google.adk import Runner
from google.adk.agents import LlmAgent, Agent
from google.adk.planners import BuiltInPlanner
from google.adk.sessions import InMemorySessionService, BaseSessionService, Session
from google.adk.events import Event

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.model_settings import ModelSettings, ThinkingSettings
from agentic_cli.workflow.adk.event_processor import ADKEventProcessor
from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin
from agentic_cli.workflow.adk.plugins import LLMLoggingPlugin

from agentic_cli.config import (
    BaseSettings,
    get_settings,
)
from agentic_cli.logging import Loggers, bind_context


logger = Loggers.workflow()

# Suppress Google GenAI SDK warning about non-text parts (function_call) in
# mixed responses. We already handle all part types individually in process_part.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)


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
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
    ) -> None:
        """Initialize the workflow manager.

        Args:
            agent_configs: List of agent configurations. First agent with sub_agents
                          is treated as the root/coordinator agent.
            settings: Application settings (uses get_settings() if not provided)
            app_name: Application name for services (uses settings.app_name if not provided)
            model: Model override (auto-detected from API keys if not provided)
            on_event: Optional hook to transform/filter events before yielding
        """
        super().__init__(
            agent_configs=agent_configs,
            settings=settings,
            app_name=app_name,
            model=model,
            on_event=on_event,
        )
        self.session_id = "default_session"

        self._session_service: BaseSessionService | None = None
        self._root_agent: Agent | None = None
        self._runner: Runner | None = None

        self._event_processor = ADKEventProcessor(
            model=model or "",
            on_event=on_event,
        )

        # Plugins (initialized lazily in _init_plugins, but we need the
        # reference to exist before process() is called)
        self._llm_logging_plugin: LLMLoggingPlugin | None = None
        self._task_progress_plugin: "TaskProgressPlugin | None" = None

        logger.debug(
            "workflow_manager_created",
            app_name=self.app_name,
            model_override=model,
            agent_count=len(agent_configs),
        )

    @property
    def backend_type(self) -> str:
        """Return 'adk'."""
        return "adk"

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

        # Clear LLM logging plugin
        if self._llm_logging_plugin:
            self._llm_logging_plugin.clear()
            self._llm_logging_plugin = None

        # Clean up managers (sandbox, etc.)
        self._cleanup_managers()

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

        # Update model
        self._reset_model(model)

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
                    plugins=self._init_plugins(),
                )

        logger.info(
            "workflow_manager_reinitialized",
            model=self.model,
            sessions_preserved=preserve_sessions,
        )

    def _resolve_model_for_config(self, config: "AgentConfig | None") -> str:
        """Return the effective model for an agent (per-config override or default)."""
        if config is not None and config.model:
            return config.model
        return self.model

    def _resolve_thinking(
        self, config: "AgentConfig | None"
    ) -> ThinkingSettings | None:
        """Resolve thinking settings: per-agent override, else global effort.

        Returns None when thinking is disabled (no per-agent setting and the
        global ``thinking_effort`` is ``"none"``).
        """
        ms = config.model_settings if config is not None else None
        if ms is not None and ms.thinking is not None:
            return ms.thinking
        global_effort = self._settings.thinking_effort
        if global_effort == "none":
            return None
        return ThinkingSettings(mode=global_effort)

    def _get_planner(
        self, config: "AgentConfig | None" = None
    ) -> BuiltInPlanner | None:
        """Get planner with thinking configuration for an agent.

        Thinking is resolved per-agent (``config.model_settings.thinking``) with
        a fallback to the global ``settings.thinking_effort``.

        Gemini 3 models take a discrete ``thinking_level``; Gemini 2.5 models
        only understand a numeric ``thinking_budget`` and reject ``thinking_level``
        outright (HTTP 400 "Thinking level is not supported for this model").
        We therefore choose the field that matches the model generation —
        sending ``thinking_level`` to a 2.5 model breaks every request.
        """
        thinking = self._resolve_thinking(config)
        if thinking is None or thinking.mode == "none":
            return None

        model = self._resolve_model_for_config(config)

        if not self._settings.supports_thinking_effort(model):
            logger.debug("thinking_not_supported", model=model, mode=thinking.mode)
            return None

        if thinking.mode == "budget":
            budget = (
                thinking.budget_tokens if thinking.budget_tokens is not None else 12288
            )
            thinking_config = types.ThinkingConfig(
                include_thoughts=True, thinking_budget=budget
            )
        elif "gemini-3" in model:
            thinking_config = self._gemini3_thinking_config(thinking.mode, model)
        else:
            thinking_config = self._gemini25_thinking_config(thinking.mode)

        logger.debug("planner_created", model=model, mode=thinking.mode)
        return BuiltInPlanner(thinking_config=thinking_config)

    def _gemini3_thinking_config(
        self, effort: str, model: str
    ) -> "types.ThinkingConfig":
        """Build a Gemini 3 thinking config using the discrete ``thinking_level``.

        Gemini 3 Pro supports only LOW and HIGH (MEDIUM falls back to HIGH);
        Gemini 3 Flash additionally supports MINIMAL/MEDIUM.
        """
        is_pro = "pro" in model

        if effort == "low":
            level = types.ThinkingLevel.LOW
        elif effort == "medium":
            if is_pro:
                level = types.ThinkingLevel.HIGH
                logger.debug(
                    "thinking_level_fallback",
                    model=model,
                    requested="medium",
                    actual="high",
                    reason="Gemini 3 Pro only supports LOW and HIGH",
                )
            else:
                level = types.ThinkingLevel.MEDIUM
        else:  # high
            level = types.ThinkingLevel.HIGH

        return types.ThinkingConfig(include_thoughts=True, thinking_level=level)

    def _gemini25_thinking_config(self, effort: str) -> "types.ThinkingConfig":
        """Build a Gemini 2.5 thinking config using the numeric ``thinking_budget``.

        2.5 models reject ``thinking_level``. Budgets are chosen within the
        range valid across the 2.5 family (Flash/Pro/Flash-Lite all accept
        4096–24576 tokens).
        """
        budget = {"low": 4096, "medium": 12288, "high": 24576}[effort]
        return types.ThinkingConfig(include_thoughts=True, thinking_budget=budget)

    def _get_generate_content_config(
        self, config: "AgentConfig | None" = None
    ) -> types.GenerateContentConfig:
        """Build GenerateContentConfig with retry HTTP options and per-agent params.

        Args:
            config: Agent config whose ``model_settings`` (if any) supply
                generation params merged on top of the retry options.

        Returns:
            GenerateContentConfig with HttpRetryOptions plus any per-agent params.
        """
        http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                initial_delay=self._settings.retry_initial_delay,
                attempts=self._settings.retry_max_attempts,
                exp_base=self._settings.retry_backoff_factor,
                http_status_codes=[500, 502, 503, 504],  # Don't auto-retry 429
            )
        )
        kwargs: dict[str, Any] = {"http_options": http_options}
        ms = config.model_settings if config is not None else None
        if ms is not None:
            kwargs.update(self._generate_config_kwargs_from_settings(ms))
        return types.GenerateContentConfig(**kwargs)

    def _generate_config_kwargs_from_settings(
        self, ms: ModelSettings
    ) -> dict[str, Any]:
        """Translate neutral ModelSettings into GenerateContentConfig kwargs.

        Maps neutral field names (``max_tokens`` -> ``max_output_tokens``) and
        passes ``extra`` through, filtered to valid GenerateContentConfig fields
        (unknown keys are logged and dropped). Thinking is handled by the planner.
        """
        out: dict[str, Any] = {}
        if ms.temperature is not None:
            out["temperature"] = ms.temperature
        if ms.top_p is not None:
            out["top_p"] = ms.top_p
        if ms.top_k is not None:
            out["top_k"] = ms.top_k
        if ms.max_tokens is not None:
            out["max_output_tokens"] = ms.max_tokens
        if ms.stop_sequences is not None:
            out["stop_sequences"] = ms.stop_sequences
        if ms.extra:
            valid = set(types.GenerateContentConfig.model_fields)
            for key, value in ms.extra.items():
                if key in valid:
                    out[key] = value
                else:
                    logger.warning("model_settings_extra_ignored", key=key)
        return out

    def _get_state_tools(self) -> list:
        """Return ADK-native state tools using ToolContext.state."""
        from agentic_cli.tools.adk.state_tools import (
            save_plan, get_plan, save_tasks, get_tasks,
        )
        return [save_plan, get_plan, save_tasks, get_tasks]

    def _assemble_agent_tools(
        self, config: "AgentConfig", service_map: dict
    ) -> list:
        """Build an agent's tools: framework tools + state tools + MCP toolsets.

        MCP servers declared on the config are materialized into ADK
        ``MCPToolset`` objects (ADK connects lazily) and appended after the
        regular tools.
        """
        tools = self._build_tools(config, service_map)
        mcp_servers = getattr(config, "mcp_servers", None) or []
        if mcp_servers:
            from agentic_cli.workflow.mcp import to_adk_toolset

            for server in mcp_servers:
                tools.append(to_adk_toolset(server))
                logger.debug(
                    "mcp_toolset_attached", agent=config.name, server=server.name
                )

        skill_refs = getattr(config, "skills", None) or []
        if skill_refs:
            toolset = self._build_skill_toolset(skill_refs)
            if toolset is not None:
                tools.append(toolset)
                logger.debug("skill_toolset_attached", agent=config.name)
        return tools

    def _build_skill_toolset(self, skill_refs: list[str]):
        """Resolve skill refs and build an ADK SkillToolset (scripts gated).

        Script execution is disabled unless ``settings.skill_scripts_enabled``
        is True (and a code executor is wired — a future enhancement), so by
        default only discovery/read tools are exposed.
        """
        from agentic_cli.tools.skills import SkillStore, make_skill_toolset

        store = SkillStore(getattr(self._settings, "skills_dirs", []) or [])
        skills = store.resolve(skill_refs)
        if not skills:
            return None
        scripts_enabled = getattr(self._settings, "skill_scripts_enabled", False)
        return make_skill_toolset(skills, scripts_enabled=scripts_enabled)

    def _create_agents(self) -> Agent:
        """Create agent hierarchy from configs.

        Returns:
            Root agent (the first agent with sub_agents, or first agent if none have sub_agents)
        """
        # Build a map of agent configs by name
        config_map = {config.name: config for config in self._agent_configs}

        # Build agents (non-coordinators first, then coordinators)
        agent_map: dict[str, Agent] = {}
        service_map = self._get_service_tool_map()

        # First pass: create agents without sub_agents (leaf agents)
        for config in self._agent_configs:
            if not config.sub_agents:
                agent_map[config.name] = LlmAgent(
                    name=config.name,
                    model=config.model or self.model,
                    instruction=config.get_prompt(),
                    tools=self._assemble_agent_tools(config, service_map),
                    description=config.description or None,
                    planner=self._get_planner(config),
                    generate_content_config=self._get_generate_content_config(config),
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
                    tools=self._assemble_agent_tools(config, service_map),
                    description=config.description or None,
                    sub_agents=sub_agent_instances,
                    planner=self._get_planner(config),
                    generate_content_config=self._get_generate_content_config(config),
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

    def _init_plugins(self) -> list:
        """Create ADK plugins and store references for later access.

        Returns:
            List of BasePlugin instances to pass to Runner(plugins=...).
        """
        from agentic_cli.workflow.adk.task_progress_plugin import TaskProgressPlugin

        plugins: list = [PermissionPlugin()]

        # Task progress tracking via ToolContext.state
        self._task_progress_plugin = TaskProgressPlugin()
        plugins.append(self._task_progress_plugin)

        if self._settings.raw_llm_logging:
            self._llm_logging_plugin = LLMLoggingPlugin(
                model_name=self.model,
                app_name=self._settings.app_name,
                include_messages=True,
                include_raw_parts=True,
            )
            plugins.append(self._llm_logging_plugin)
            logger.info("llm_logging_plugin_enabled")
        else:
            self._llm_logging_plugin = None

        return plugins

    async def _do_initialize(self) -> None:
        """ADK-specific initialization: session service, agents, runner."""
        logger.info("initializing_services", app_name=self.app_name)

        # Create session service
        self._session_service = InMemorySessionService()
        logger.debug("using_in_memory_session_service")

        # Create agent hierarchy from configs
        self._root_agent = self._create_agents()

        # Create runner with plugins
        self._runner = Runner(
            app_name=self.app_name,
            agent=self._root_agent,
            session_service=self._session_service,
            plugins=self._init_plugins(),
        )

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
        if not self._runner or not self._session_service or not self._root_agent:
            raise RuntimeError(
                "Workflow Manager failed to initialize. Check API keys and configuration."
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
            new_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )

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
                # Yield LLM events from plugin first (raw capture)
                if self._llm_logging_plugin:
                    for llm_event in self._llm_logging_plugin.drain_events():
                        llm_event = self._apply_event_hook(llm_event)
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

                    # Drain task progress events buffered by the plugin
                    if workflow_event.type == EventType.TOOL_RESULT and self._task_progress_plugin:
                        for progress_event in self._task_progress_plugin.drain_events():
                            progress_event = self._apply_event_hook(progress_event)
                            if progress_event:
                                event_count += 1
                                yield progress_event

            # Final drain — catches progress from the last tool call
            if self._task_progress_plugin:
                for progress_event in self._task_progress_plugin.drain_events():
                    progress_event = self._apply_event_hook(progress_event)
                    if progress_event:
                        event_count += 1
                        yield progress_event

            # Drain any remaining LLM events after processing completes
            if self._llm_logging_plugin:
                for llm_event in self._llm_logging_plugin.drain_events():
                    llm_event = self._apply_event_hook(llm_event)
                    if llm_event:
                        event_count += 1
                        yield llm_event

            logger.info("message_processed", event_count=event_count)

    # -------------------------------------------------------------------------
    # Session save/resume hooks
    # -------------------------------------------------------------------------

    async def _extract_session_data(self, session_id: str) -> tuple[list[dict], str | None]:
        """Extract normalized messages and current agent from ADK session.

        Returns:
            Tuple of (messages list, current agent name or None).
        """
        if not self._session_service:
            return [], None

        session = await self._session_service.get_session(
            app_name=self.app_name,
            user_id=self._settings.default_user,
            session_id=session_id,
        )
        if session is None or not session.events:
            return [], None

        messages: list[dict] = []
        for event in session.events:
            content = getattr(event, "content", None)
            if content is None:
                continue

            role = getattr(content, "role", None)
            parts = getattr(content, "parts", None) or []

            for part in parts:
                # Text part
                if hasattr(part, "text") and part.text:
                    if role == "user":
                        messages.append({"role": "user", "content": part.text})
                    elif role == "model":
                        messages.append({"role": "assistant", "content": part.text})

                # Function call part
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_call = {
                        "id": getattr(fc, "id", fc.name),
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                    }
                    # Attach to preceding assistant message or create one
                    if messages and messages[-1]["role"] == "assistant":
                        messages[-1].setdefault("tool_calls", []).append(tool_call)
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [tool_call],
                        })

                # Function response part
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    response_content = json.dumps(fr.response) if isinstance(fr.response, dict) else str(fr.response)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": getattr(fr, "id", fr.name),
                        "name": fr.name,
                        "content": response_content,
                    })

        current_agent = getattr(session.events[-1], "author", None)
        return messages, current_agent

    async def _inject_session_messages(
        self,
        session_id: str,
        messages: list[dict],
        current_agent: str | None = None,
    ) -> None:
        """Inject normalized messages into the ADK session as real events.

        Uses ``append_event`` so events land in the *stored* session.
        ``create_session`` returns a copy of the stored session, so the old
        approach of appending to that copy left the stored session empty and
        silently lost the restored history on resume.
        """
        if not self._session_service:
            raise RuntimeError("Session service not initialized")

        # Create a fresh session for the restored conversation
        session = await self._session_service.create_session(
            app_name=self.app_name,
            user_id=self._settings.default_user,
            session_id=session_id,
        )

        async def _add(content: types.Content, author: str) -> None:
            await self._session_service.append_event(
                session, Event(author=author or "user", content=content)
            )

        for msg in messages:
            role = msg["role"]

            if role == "user":
                content = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg["content"])],
                )
                await _add(content, "user")

            elif role == "assistant":
                parts = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=msg["content"]))
                for tc in msg.get("tool_calls", []):
                    parts.append(types.Part.from_function_call(
                        name=tc["name"],
                        args=tc.get("args", {}),
                    ))
                content = types.Content(role="model", parts=parts)
                await _add(content, current_agent or "model")

            elif role == "tool":
                try:
                    response = json.loads(msg["content"])
                except (json.JSONDecodeError, TypeError):
                    response = {"result": msg["content"]}
                parts = [types.Part.from_function_response(
                    name=msg.get("name", "unknown"),
                    response=response,
                )]
                content = types.Content(role="user", parts=parts)
                await _add(content, current_agent or "user")
