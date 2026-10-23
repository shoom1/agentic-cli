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
from google.adk.apps import App
from google.adk.apps.app import validate_app_name
from google.adk.agents import LlmAgent, Agent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions import InMemorySessionService, BaseSessionService, Session
from google.adk.tools import LongRunningFunctionTool

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.workflow.config import AgentConfig, validate_agent_graph
from agentic_cli.workflow.adk.event_processor import ADKEventProcessor
from agentic_cli.workflow.adk.model_factory import build_adk_model
from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin
from agentic_cli.workflow.adk.plugins import LLMLoggingPlugin
from agentic_cli.workflow.adk.tool_result_cap_plugin import ToolResultCapPlugin
from agentic_cli.workflow.service_registry import JOB_MANAGER
from agentic_cli.workflow.tool_output import tool_output_dir

from agentic_cli.config import (
    BaseSettings,
    get_settings,
)
from agentic_cli.logging import Loggers, bind_context
from agentic_cli.models import CacheMechanism
from agentic_cli.tools.jobs.resume import job_result_payload


logger = Loggers.workflow()

# Suppress Google GenAI SDK warning about non-text parts (function_call) in
# mixed responses. We already handle all part types individually in process_part.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

# The cache mechanisms ADK's context cache serves: Claude's breakpoints and
# Gemini's server-side caches.
_CACHING = (CacheMechanism.ANTHROPIC, CacheMechanism.GEMINI)

# How long ADK may take to create a Gemini context cache, in milliseconds.
# Generation's own timeout sits on each request's config, not on the client
# that creates the cache, so without this a hung create would hang the turn. A
# create that fails costs only that turn's cache: ADK logs it and sends the
# request uncached, and ContextCachePlugin keeps it from trying again at every
# tool step of the turn.
_CACHE_CREATE_TIMEOUT_MS = 60_000


def _check_app_name(app_name: str) -> None:
    """Refuse an ``app_name`` ADK cannot run, naming the setting.

    ADK's ``App`` validates its name, so an unsuitable one would otherwise
    fail on the first turn, inside initialization. ADK's own rule is applied,
    so it cannot drift from what ADK accepts.
    """
    try:
        validate_app_name(app_name)
    except ValueError as exc:
        raise ValueError(
            f"app_name {app_name!r} cannot be used with the ADK backend: {exc}"
        ) from exc


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
        adk_config_path: str | None = None,
    ) -> None:
        """Initialize the workflow manager.

        Args:
            agent_configs: List of agent configurations. First agent with sub_agents
                          is treated as the root/coordinator agent.
            settings: Application settings (uses get_settings() if not provided)
            app_name: Application name for services (uses settings.app_name if not provided)
            model: This manager's default model, a ``provider:model`` reference
                or a model id (see ``BaseWorkflowManager``)
            on_event: Optional hook to transform/filter events before yielding
            adk_config_path: Optional path to a native ADK ``root_agent.yaml``.
                When set, the agent tree is built via ADK ``from_config`` instead
                of from ``agent_configs`` (full ADK fidelity; framework service /
                state tool injection does not apply).
        """
        super().__init__(
            agent_configs=agent_configs,
            settings=settings,
            app_name=app_name,
            model=model,
            on_event=on_event,
        )
        _check_app_name(self.app_name)
        self.session_id = "default_session"
        self._adk_config_path = adk_config_path

        self._session_service: BaseSessionService | None = None
        # True while reinitialize(preserve_sessions=True) is in flight: the
        # live session service is being carried across and must survive both a
        # successful rebuild and a rollback.
        self._session_service_pinned = False
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

        # Resolve bare ``claude-*`` strings (e.g. from agent YAML / native ADK
        # configs) to the direct-API DirectAnthropicLlm instead of Vertex Claude.
        from agentic_cli.workflow.adk.anthropic_llm import register_direct_anthropic

        register_direct_anthropic()

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
        """Generate a short text with the default model, outside the agent workflow.

        Used for internal operations such as summaries. The default model is
        built through the model factory, as an agent's is, with thinking off
        where the model allows it and ``max_tokens`` as its limit, so every
        provider can write them.

        Args:
            prompt: The prompt to send.
            max_tokens: The output limit.

        Returns:
            The answer's text, without the model's thoughts.
        """
        await self._ensure_initialized()
        built = build_adk_model(
            self._simple_model(max_tokens), settings=self._settings, app_name=self.app_name
        )
        request = LlmRequest(
            model=built.model.model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=built.generate_content_config,
        )
        if built.planner is not None:
            built.planner.apply_thinking_config(request)

        text: list[str] = []
        async for response in built.model.generate_content_async(request, stream=False):
            parts = response.content.parts if response.content is not None else None
            text.extend(part.text for part in parts or [] if part.text and not part.thought)
        return "".join(text)

    async def cleanup(self) -> None:
        """Clean up workflow manager resources.

        Releases resources and resets state. Call this before shutting down or
        when reinitializing with new settings. Idempotent: references are
        detached before being closed, so a repeat call is a no-op. Takes the
        lifecycle and turn locks, so it cannot tear the runner down while a
        turn is streaming.
        """
        async with self._lifecycle_lock:
            async with self._turn_lock:
                await self._release_resources()

    async def _release_resources(self, keep_session_service: bool = False) -> None:
        """Release owned resources; never raises.

        The session service is *closed*, not merely dropped — the durable
        ``DatabaseSessionService`` holds a SQLAlchemy engine whose connection
        pool leaks otherwise. ``reinitialize(preserve_sessions=True)`` keeps it
        instead, and then reuses that same instance rather than building a
        replacement that would be discarded unclosed.

        Args:
            keep_session_service: Retain (do not close) the session service.
        """
        logger.debug("cleaning_up_workflow_manager")

        # Detach first so a concurrent/repeat cleanup can't double-close. A
        # pinned service is being carried across a reinitialize, so a rollback
        # inside that window must not close it either.
        session_service = None
        if not (keep_session_service or self._session_service_pinned):
            session_service, self._session_service = self._session_service, None
        self._runner = None
        self._root_agent = None
        self._initialized = False

        # Clear LLM logging plugin
        if self._llm_logging_plugin:
            self._llm_logging_plugin.clear()
            self._llm_logging_plugin = None

        await self._aclose_owned(session_service, "session_service")
        await self._close_catalog()

        # Clean up managers (sandbox, jobs, etc.)
        self._cleanup_managers()

        logger.info("workflow_manager_cleaned_up")

    async def reinitialize(
        self,
        model: str | None = None,
        preserve_sessions: bool = True,
    ) -> None:
        """Reinitialize the workflow manager with new configuration.

        Use this method when settings change (e.g. a model switch) to recreate
        agents and runners with the new configuration.

        Transactional: on failure the manager ends up *uninitialized* (
        ``is_initialized`` False) with nothing left allocated, rather than
        holding a half-built runner that would answer as if it were ready. The
        caller (``WorkflowController``) turns that into a FAILED state. A
        preserved session service survives both outcomes, so a retry can still
        continue the same conversations; so do the services scoped to them
        (session grants, sandbox kernels, jobs — see ``_carrying_services``).

        Args:
            model: Optional new model to use. If None, re-resolves from settings.
            preserve_sessions: If True, keeps the existing session service
                (default) and the services (permission engine, sandbox, jobs,
                knowledge bases) and reuses them for the new runner. If False,
                they are released and fresh ones created.

        Raises:
            Exception: Whatever initialization raised, after rollback.
        """
        logger.info(
            "reinitializing_workflow_manager",
            new_model=model,
            preserve_sessions=preserve_sessions,
        )

        async with self._lifecycle_lock:
            async with self._turn_lock:
                preserved = self._session_service if preserve_sessions else None
                with self._carrying_services(preserve_sessions):
                    await self._release_resources(keep_session_service=preserve_sessions)
                    self._reset_model(model)
                    self._session_service_pinned = preserve_sessions
                    try:
                        # _do_initialize reuses self._session_service when set, so
                        # no replacement service is built for a preserved one.
                        await self._initialize_locked()
                    except BaseException:
                        # _initialize_locked already rolled back what it created;
                        # restore the preserved service so a retry can use it.
                        if preserved is not None:
                            self._session_service = preserved
                        raise
                    finally:
                        self._session_service_pinned = False

        logger.info(
            "workflow_manager_reinitialized",
            model=self.model,
            sessions_preserved=preserve_sessions,
        )

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
        regular tools. Skills are appended as a ``SkillToolset``. The caller is
        responsible for ``_wrap_long_running`` over the result.
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
        """Resolve skill refs and build an ADK SkillToolset (discovery/read only).

        This path supplies no code executor, so ``run_skill_script`` is not
        exposed: only the discovery/read tools and the L1 metadata injection.
        (There used to be a ``skill_scripts_enabled`` setting that advertised
        the tool anyway; with no executor it could only ever answer
        ``NO_CODE_EXECUTOR``.)
        """
        from agentic_cli.tools.skills import SkillStore, make_skill_toolset

        store = SkillStore(getattr(self._settings, "skills_dirs", []) or [])
        skills = store.resolve(skill_refs)
        if not skills:
            return None
        return make_skill_toolset(skills)

    def _wrap_long_running(self, tools: list[Callable]) -> list:
        """Wrap tools flagged ``long_running`` as ADK ``LongRunningFunctionTool``.

        A long-running tool returns a ``job_id`` immediately and the model is
        instructed not to re-call it while pending; the eventual result is
        delivered later as a ``FunctionResponse`` (see ``resume_with_job_result``).
        Detection is by registry identity — the exact object the default
        registry issued, never a matching name — so a plain callable an
        application names after a long-running tool keeps its ordinary
        call-and-return contract instead of being told to leave a job pending
        that nothing will ever complete. Non-long-running tools and any
        already-wrapped tools (including toolset objects) pass through
        unchanged. Permission gating is unaffected: ADK gates via
        ``PermissionPlugin``, which resolves the same identity.
        """
        from agentic_cli.tools.registry import identify_tool

        wrapped: list = []
        for tool in tools:
            defn = identify_tool(tool)
            name = defn.name if defn is not None else getattr(tool, "__name__", "")
            if (
                defn is not None
                and defn.long_running
                and not isinstance(tool, LongRunningFunctionTool)
            ):
                wrapped.append(LongRunningFunctionTool(func=tool))
                logger.debug("long_running_tool_wrapped", tool=name)
            else:
                wrapped.append(tool)
        return wrapped

    def _build_agent(
        self,
        config: "AgentConfig",
        service_map: dict,
        sub_agents: list[Agent] | None = None,
    ) -> Agent:
        """Construct one ``LlmAgent`` from a config (with resolved sub-agents).

        Its model, generation settings and thinking come from the model
        initialization resolved for it, built by the model factory.
        """
        built = build_adk_model(
            self._resolved_models[config.name], settings=self._settings, app_name=self.app_name
        )
        return LlmAgent(
            name=config.name,
            model=built.model,
            instruction=config.get_prompt(self._settings),
            tools=self._wrap_long_running(
                self._assemble_agent_tools(config, service_map)
            ),
            description=config.description,
            sub_agents=sub_agents or [],
            planner=built.planner,
            generate_content_config=built.generate_content_config,
        )

    def _create_agents(self) -> Agent:
        """Create the agent hierarchy from configs, in dependency order.

        The graph is validated first (``validate_agent_graph``) so duplicate
        names, dangling ``sub_agents`` references, self-references, cycles and
        shapes ADK cannot represent fail with the offending agent named,
        instead of silently producing a coordinator with missing children.

        Construction then follows a topological order, so an agent is always
        built after the agents it delegates to — the previous leaves-then-
        coordinators split silently dropped a sub-agent that was itself a
        coordinator.

        Prompt factories are called under this manager's settings context (see
        ``AgentConfig.get_prompt``), so a prompt that reads settings sees the
        manager's instance rather than the global singleton.

        Returns:
            Root agent (the first agent with sub_agents, or the first config).

        Raises:
            AgentGraphError: If the configured graph is invalid.
        """
        graph = validate_agent_graph(self._agent_configs, backend=self.backend_type)

        agent_map: dict[str, Agent] = {}
        service_map = self._get_service_tool_map()

        from agentic_cli.config import SettingsContext

        # Prompt factories run inside the manager's settings context.
        with SettingsContext(self._settings):
            for name in graph.build_order:
                config = graph.config_map[name]
                sub_agents = [agent_map[sub] for sub in config.sub_agents]
                agent_map[name] = self._build_agent(config, service_map, sub_agents)
                logger.debug(
                    "agent_created",
                    name=name,
                    type="coordinator" if sub_agents else "leaf",
                    sub_agents=[a.name for a in sub_agents],
                )

        root_agent = agent_map[graph.root_name]
        logger.info("agents_created", root=root_agent.name, total=len(agent_map))
        return root_agent

    def _init_plugins(self) -> list:
        """Create ADK plugins and store references for later access.

        Returns:
            List of BasePlugin instances for the runner's App(plugins=...).
        """
        from agentic_cli.workflow.adk.task_progress_plugin import TaskProgressPlugin
        from agentic_cli.workflow.adk.tool_error_plugin import ToolErrorPlugin

        # ToolErrorPlugin: ADK ends the turn when a tool raises; this turns the
        # exception into an error result the model can read.
        plugins: list = [PermissionPlugin(), ToolErrorPlugin()]

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

    def _make_session_service(self) -> BaseSessionService:
        """Build the ADK session service from ``session_store`` settings.

        ``memory`` → ephemeral ``InMemorySessionService``; ``sqlite``/``postgres``
        → native ``DatabaseSessionService`` (durable across restarts, full event
        fidelity). The sqlite parent dir is created so the engine can open it.
        """
        db_url = self._settings.session_db_url()
        if db_url is None:
            logger.debug("using_in_memory_session_service")
            return InMemorySessionService()

        if db_url.startswith("sqlite"):
            # sqlite+aiosqlite:///<path> — ensure the directory exists.
            from pathlib import Path

            path = db_url.split(":///", 1)[-1]
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)

        from google.adk.sessions import DatabaseSessionService

        logger.info("using_database_session_service", store=self._settings.session_store)
        return DatabaseSessionService(db_url=db_url)

    async def _do_initialize(self) -> None:
        """ADK-specific initialization: session service, agents, runner."""
        logger.info("initializing_services", app_name=self.app_name)

        # Create the session service (durable DatabaseSessionService by
        # default; InMemory only when session_store='memory') — unless one is
        # already held, which is how reinitialize(preserve_sessions=True)
        # carries live conversations across without building a replacement
        # that would then be discarded unclosed.
        if self._session_service is None:
            self._session_service = self._make_session_service()

        # Create agent hierarchy — natively from an ADK config, or from configs.
        if self._adk_config_path:
            from agentic_cli.workflow.adk_config_bridge import load_adk_agent_native

            self._root_agent = load_adk_agent_native(self._adk_config_path)
            logger.info("adk_native_config_loaded", path=self._adk_config_path)
        else:
            self._root_agent = self._create_agents()

        # The plugins belong to the App; ADK deprecates Runner(plugins=...).
        context_cache = self._context_cache_config()
        plugins = self._init_plugins()
        if not self._adk_config_path:
            from agentic_cli.workflow.adk.foreign_reasoning_plugin import ForeignReasoningPlugin

            # A model switch keeps the session: no model is sent the thoughts
            # another provider wrote, which it would refuse.
            plugins.append(ForeignReasoningPlugin({
                name: model.record.kind.value for name, model in self._resolved_models.items()
            }))
        if context_cache is not None:
            from agentic_cli.workflow.adk.context_cache_plugin import ContextCachePlugin

            # One Gemini cache attempt a turn, not one per tool step.
            plugins.append(ContextCachePlugin())
        if not self._adk_config_path:
            # Every tool result over the limit is cut. After the task progress
            # plugin: ADK stops at the first plugin whose after_tool_callback
            # returns a value, and this one returns the cut. Not for a native
            # ADK app, whose agents' own after_tool_callbacks it would skip.
            plugins.append(ToolResultCapPlugin(
                self._settings.tool_result_max_chars, tool_output_dir(self._settings.app_name)
            ))
        app = App(
            name=self.app_name,
            root_agent=self._root_agent,
            plugins=plugins,
            context_cache_config=context_cache,
        )
        self._runner = Runner(app=app, session_service=self._session_service)
        # Usage events name the default model, resolved for this initialization.
        self._event_processor.model = self.model

        logger.info(
            "services_initialized",
            model=self.model,
            agent_name=self._root_agent.name if self._root_agent else None,
            required_managers=list(self._required_managers),
            context_cache=app.context_cache_config is not None,
        )

    def _context_cache_config(self) -> ContextCacheConfig | None:
        """The app's context cache, or None for no caching.

        ADK caches for every agent of an app through one config, so it is set
        when ``prompt_caching_enabled`` is on and any agent's model caches (an
        agent without a model is resolved to the default). A native ADK YAML
        app builds no agent from a resolved model, so its default model
        decides. ADK then marks cache breakpoints on every Claude request, and
        keeps a server-side cache per Gemini agent from its second turn
        (Google bills its storage while it lives); GPT, OpenRouter and
        endpoint models are untouched.

        ADK's defaults stand: a 1800-second TTL, which keeps Claude on its
        cheaper 5-minute cache, and a Gemini cache replaced after 10
        invocations. Only creating a Gemini cache gets a bound,
        ``_CACHE_CREATE_TIMEOUT_MS``.
        """
        if not self._settings.prompt_caching_enabled:
            return None
        if self._adk_config_path:
            models = [self._default_model]
        else:
            models = list(self._resolved_models.values())
        if not any(m is not None and m.record.cache in _CACHING for m in models):
            return None
        return ContextCacheConfig(
            create_http_options=types.HttpOptions(timeout=_CACHE_CREATE_TIMEOUT_MS)
        )

    def _validate_agent_graph(self) -> None:
        """Reject an unbuildable agent graph before anything is allocated.

        Runs at the top of initialization — ahead of model listings, service
        construction and the session service — so a static configuration error
        costs no network call and leaves nothing to roll back.
        """
        if self._adk_config_path:
            return  # native ADK config: ADK owns the topology
        validate_agent_graph(self._agent_configs, backend=self.backend_type)

    async def _ensure_initialized(self) -> None:
        """Ensure services are initialized before processing.

        Called *before* the turn lock is taken, so a turn never waits on the
        lifecycle lock while holding the turn lock (that ordering is what keeps
        cleanup/reinitialize from deadlocking against a running turn). Because
        of that ordering a cleanup can still land in between, which is what
        ``_turn_admission`` re-checks with ``_backend_ready``.
        """
        if not self._initialized:
            await self.initialize_services()
        if not self._backend_ready():
            raise RuntimeError(
                "Workflow Manager failed to initialize. Check API keys and configuration."
            )

    def _backend_ready(self) -> bool:
        """True when the runner, session service and agent tree are all live."""
        return bool(
            self._initialized
            and self._runner is not None
            and self._session_service is not None
            and self._root_agent is not None
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

        Holds the manager's turn lock for the whole stream: the ADK plugins'
        event buffers are manager-scoped, so overlapping turns would drain each
        other's events. Overlapping callers queue; the lock is released on
        cancellation. Admission also re-verifies that the backend is still live
        (a cleanup can land between initialization and the turn lock).

        Args:
            message: User message
            user_id: User identifier
            session_id: Optional session identifier

        Yields:
            WorkflowEvent objects representing workflow output
        """
        current_session_id = session_id or self.session_id
        bind_context(session_id=current_session_id, user_id=user_id)

        logger.info("processing_message", message_length=len(message))

        async with self._turn_admission():
            # Context setup
            with self._workflow_context(session_id=current_session_id, user_id=user_id):
                # Session handling
                await self._get_or_create_session(user_id, current_session_id)

                # Create message
                new_message = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                )

                async for event in self._run_and_stream(
                    session_id=current_session_id,
                    user_id=user_id,
                    new_message=new_message,
                ):
                    yield event

    async def _run_and_stream(
        self,
        *,
        session_id: str,
        user_id: str,
        new_message: "types.Content",
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run the ADK runner for one invocation and stream WorkflowEvents.

        Shared by ``process()`` (user message) and ``resume_with_job_result()``
        (function response). The caller must already be inside
        ``_workflow_context`` with the session prepared.
        """
        event_count = 0

        # Process ADK events directly - retry is handled by HttpRetryOptions
        async for adk_event in self._runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=new_message,
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
                session_id,
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

    async def can_resume(self, record) -> bool:
        """True iff the originating ADK session still exists to resume into.

        Requires the resume ids and a session holding the pending call. The
        session service is durable by default (``session_store='sqlite'``), so
        the session normally survives a CLI restart and the job stays
        resumable. This returns False — and the harness surfaces a notice
        instead of a dead resume turn — when the session is genuinely gone: it
        was deleted, the record lacks its session/user/call ids, or the run used
        the explicitly ephemeral ``session_store='memory'`` and the process
        restarted.
        """
        if not (record.session_id and record.user_id and record.call_id):
            return False
        if self._session_service is None:
            return False
        session = await self._session_service.get_session(
            app_name=self.app_name, user_id=record.user_id, session_id=record.session_id,
        )
        return session is not None

    async def resume_with_job_result(
        self, record, result: Any = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Resume the agent with a finished long-running job's result.

        Delivers the result to the pending long-running tool call as an ADK
        ``FunctionResponse`` (matching ``record.call_id``) and re-invokes the
        runner, streaming the follow-up turn's events. The originating session
        must still contain the pending call (it is persisted); if it's gone or
        the record lacks the ids needed to resume, this yields nothing.

        Args:
            record: The terminal ``JobRecord`` to resume from.
            result: The job's result; fetched from the JobManager if omitted.
        """
        session_id = record.session_id
        user_id = record.user_id
        if not session_id or not user_id or not record.call_id:
            logger.warning(
                "job_resume_missing_ids",
                job_id=getattr(record, "job_id", None),
                have_session=bool(session_id),
                have_user=bool(user_id),
                have_call_id=bool(record.call_id),
            )
            return

        bind_context(session_id=session_id, user_id=user_id)

        # Same admission as process(): a resume is a turn, must not interleave
        # with a user turn, and must not run against a released backend.
        async with self._turn_admission():
            with self._workflow_context(session_id=session_id, user_id=user_id):
                # The pending call lives in the existing session; don't create a
                # new empty one (that would have no call to answer).
                session = await self._session_service.get_session(
                    app_name=self.app_name, user_id=user_id, session_id=session_id,
                )
                if session is None:
                    logger.warning("job_resume_session_missing", job_id=record.job_id,
                                   session_id=session_id)
                    return

                if result is None:
                    jm = self._services.get(JOB_MANAGER)
                    if jm is not None:
                        result = jm.result(record.job_id)

                function_response = types.FunctionResponse(
                    id=record.call_id,
                    name=record.call_name or record.tool,
                    # A summary and a pointer, not the raw result.
                    response=job_result_payload(record, result),
                )
                new_message = types.Content(
                    role="user",
                    parts=[types.Part(function_response=function_response)],
                )

                logger.info("job_resume_started", job_id=record.job_id,
                            call_id=record.call_id, state=record.state.value)
                async for event in self._run_and_stream(
                    session_id=session_id,
                    user_id=user_id,
                    new_message=new_message,
                ):
                    yield event

    # -------------------------------------------------------------------------
    # Sessions (native — DatabaseSessionService persists events continuously)
    # -------------------------------------------------------------------------

    async def session_exists(self, session_id: str, *, user_id: str | None = None) -> bool:
        """True if the store holds this session with any events.

        Args:
            session_id: Session to look up.
            user_id: Owner of the session (defaults to ``settings.default_user``).
        """
        if not self._session_service:
            return False
        ref = self.session_ref(session_id, user_id)
        session = await self._session_service.get_session(
            app_name=ref.app_name,
            user_id=ref.user_id,
            session_id=ref.session_id,
        )
        return session is not None and bool(getattr(session, "events", None))

    async def list_sessions(self, *, user_id: str | None = None) -> list[dict]:
        """List persisted sessions for a user (most recent first).

        Args:
            user_id: Owner whose sessions to list (defaults to
                ``settings.default_user``).
        """
        if not self._session_service:
            return []
        ref = self.session_ref(user_id=user_id)
        resp = await self._session_service.list_sessions(
            app_name=ref.app_name, user_id=ref.user_id,
        )
        sessions = [
            {
                "session_id": s.id,
                "last_update": getattr(s, "last_update_time", None),
                "message_count": len(getattr(s, "events", None) or []),
            }
            for s in getattr(resp, "sessions", [])
        ]
        sessions.sort(key=lambda x: x["last_update"] or 0, reverse=True)
        return sessions

    async def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        """Delete a persisted session from the store.

        Args:
            session_id: Session to delete.
            user_id: Owner of the session (defaults to ``settings.default_user``).
        """
        if not self._session_service:
            return False
        ref = self.session_ref(session_id, user_id)
        await self._session_service.delete_session(
            app_name=ref.app_name,
            user_id=ref.user_id,
            session_id=ref.session_id,
        )
        return True

    async def recent_messages(
        self, session_id: str, limit: int = 20, *, user_id: str | None = None
    ) -> list[dict]:
        """Recent text messages from the stored session (for fact extraction).

        Args:
            session_id: Session to read.
            limit: Maximum number of messages returned (most recent last).
            user_id: Owner of the session (defaults to ``settings.default_user``).
        """
        if not self._session_service:
            return []
        ref = self.session_ref(session_id, user_id)
        session = await self._session_service.get_session(
            app_name=ref.app_name,
            user_id=ref.user_id,
            session_id=ref.session_id,
        )
        if session is None or not getattr(session, "events", None):
            return []
        out: list[dict] = []
        for event in session.events:
            content = getattr(event, "content", None)
            if content is None:
                continue
            role = getattr(content, "role", None)
            text = " ".join(
                p.text for p in (getattr(content, "parts", None) or [])
                if getattr(p, "text", None)
            )
            if text:
                out.append(
                    {"role": "user" if role == "user" else "assistant", "content": text}
                )
        return out[-limit:]
