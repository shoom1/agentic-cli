"""Google ADK Workflow Manager for agentic CLI applications.

This module provides a complete, config-based GoogleADKWorkflowManager that handles
agent orchestration, session management, and event streaming using Google ADK.

Domain applications define agents using AgentConfig and pass them to GoogleADKWorkflowManager.

Deep Agents is the other backend (``workflow/deepagents/manager.py``;
LangGraph, deprecated, goes in 0.8.0). Each builds on
``base_manager.BaseWorkflowManager``, which holds the lifecycle and the turn,
and ``settings.orchestrator`` picks one (``workflow/factory.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
from typing import TYPE_CHECKING, AsyncGenerator, Any, Callable

from google.genai import types
from google.adk import Runner
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig, validate_app_name
from google.adk.agents import LlmAgent, Agent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig
from google.adk.sessions import BaseSessionService, Session
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.errors import StaleSessionError
from google.adk.tools import LongRunningFunctionTool

from agentic_cli.workflow import live_events
from agentic_cli.workflow.base_manager import BaseWorkflowManager, Closer
from agentic_cli.workflow.context import CompactResult
from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.workflow.config import AgentConfig, AgentGraph, validate_agent_graph
from agentic_cli.workflow.adk import delegation
from agentic_cli.workflow.adk.compaction import (
    SUMMARY_MAX_TOKENS,
    AgentCliSummarizer,
    MainAgentPlugin,
    NoAfterTurnCompactionRunner,
    WrittenSummary,
    compact_session,
)
from agentic_cli.workflow.adk.event_processor import ADKEventProcessor
from agentic_cli.workflow.adk.fit_guard_plugin import FitGuardPlugin
from agentic_cli.workflow.adk.interrupted_calls_plugin import (
    InterruptedCallsPlugin,
    left_open,
    unsealed,
)
from agentic_cli.workflow.adk.model_factory import build_adk_model, generate_text
from agentic_cli.workflow.adk.overflow_plugin import OVERFLOW_PLUGIN_FAILED, OverflowPlugin
from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin
from agentic_cli.workflow.adk.plugins import LLMLoggingPlugin, SentRequestLoggingPlugin
from agentic_cli.workflow.adk.sessions import (
    ServiceSessions,
    make_session_service,
    turn_state,
)
from agentic_cli.workflow.adk.tool_result_cap_plugin import ToolResultCapPlugin
from agentic_cli.workflow.outside_content import (
    BEFORE_0_7_0,
    JOB_RESULT,
    OUTSIDE_CONTENT_KEY,
)
from agentic_cli.workflow.sessions import ConversationChangedError, SessionInfo
from agentic_cli.workflow.tool_output import tool_output_dir

from agentic_cli.config import (
    BaseSettings,
    get_settings,
)
from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers
from agentic_cli.models import CacheMechanism
from agentic_cli.tools.jobs.resume import JOB_NOTICE_HEADER

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentic_cli.models import Provider, ResolvedModel, RoleMap
    from agentic_cli.workflow.adk.build import ADKBuild
    from agentic_cli.workflow.sessions import SessionRef


logger = Loggers.workflow()

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

# Worker threads for a config-built app's synchronous tools (ADK's tool
# thread pool, one per event loop). A call a cancel left behind keeps its
# thread until it returns, so there are more than ADK's default 4.
_TOOL_THREADS = 8

# The longest error text a failed turn's log line carries.
_MAX_LOGGED_ERROR = 500


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


def _plugin_cause(error: RuntimeError) -> BaseException:
    """The provider's error behind ADK's wrapper of what the overflow handler raised.

    A retry that fails for another reason propagates, and ADK's plugin manager
    wraps it (``overflow_plugin.OVERFLOW_PLUGIN_FAILED``); the turn fails
    with the provider's own error instead, as any failed call does. Any other
    error is ``error`` itself, the turn's error as it is.
    """
    cause = error.__cause__
    if str(error).startswith(OVERFLOW_PLUGIN_FAILED) and cause is not None:
        return cause
    return error


def _fails_turn(event: WorkflowEvent) -> bool:
    """Whether ``event`` is an ``ERROR`` that is not recoverable: a turn's one."""
    return event.type is EventType.ERROR and not event.metadata.get("recoverable")


def _as_written(prompt: str) -> Callable[[Any], str]:
    """``prompt`` as an instruction ADK sends as written.

    ADK fills ``{name}`` in a string instruction from the session state, and
    raises ``KeyError`` for a name the state lacks (``{{name}}`` too), so a
    prompt quoting code or JSON failed every turn, and one naming ``{plan}``
    had the session's plan spliced in. What an instruction provider returns
    is never filled in, as Deep Agents never fills a prompt in.
    """
    return lambda _context: prompt


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

    #: A native ADK YAML app's agents, loaded once per manager, when they are
    #: first validated (``_native_root``). A class default, so a manager made
    #: without ``__init__`` (a test's) has none.
    _loaded_native_root: LlmAgent | None = None
    #: What the live app was built from (``_build_app``), None while there is
    #: none. A class default, as above.
    _build: "ADKBuild | None" = None
    #: The reply set of the build ``_build_app`` is making, or made last, for
    #: the parts it builds through hooks (the gate, ``_init_plugins``).
    _reply_agents: frozenset[str] | None = None

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
            agent_configs: List of agent configurations. The root (main) agent
                is the one no other agent delegates to (``validate_agent_graph``).
            settings: Application settings (uses get_settings() if not provided)
            app_name: Application name for sessions: ADK's app name and the key
                of every conversation, and what an OpenRouter provider is told
                (``settings.app_name`` if not provided).
            model: This manager's default model, a ``provider:model`` reference
                or a model id (see ``BaseWorkflowManager``)
            on_event: Optional hook to transform/filter events before yielding
            adk_config_path: Optional path to a native ADK ``root_agent.yaml``.
                When set, ADK builds the agent tree from it (``from_config``)
                instead of the manager building it from ``agent_configs``,
                which are then ignored, with a warning. Only the building
                differs: the app runs as one built from configs
                (``workflow/adk/build.py``). Library-only for now: no
                ``BaseCLIApp`` starts such an app.
        """
        if adk_config_path and agent_configs:
            # ADK builds the agents from the YAML: configs beside it would be
            # validated, resolved and given services for nothing.
            logger.warning(
                "adk_native_configs_ignored",
                path=str(adk_config_path),
                agents=[config.name for config in agent_configs],
            )
            agent_configs = []
        super().__init__(
            agent_configs=agent_configs,
            settings=settings,
            app_name=app_name,
            model=model,
            on_event=on_event,
        )
        _check_app_name(self.app_name)
        self._adk_config_path = adk_config_path
        # The graph an initialization validated, until its agents are built.
        self._validated_graph: AgentGraph | None = None

        self._session_service: BaseSessionService | None = None
        self._root_agent: Agent | None = None
        self._runner: Runner | None = None

        # Each initialization builds the processor again, from the app it built
        # (_do_initialize); this one stands until then.
        self._event_processor = ADKEventProcessor(model=model or "", on_event=on_event)

        # The raw LLM log's plugin (made in _init_plugins), whose buffer a
        # turn drains: the reference exists before process() is called.
        self._llm_logging_plugin: LLMLoggingPlugin | None = None

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
        text, _ = await generate_text(
            self._simple_model(max_tokens), prompt, settings=self._settings,
            app_name=self.app_name,
        )
        return text

    def _detach_backend(self) -> list[Closer]:
        """The runner and the agent tree, and the LLM logging plugin.

        The runner closes the agents' toolsets (an MCP server stops with its
        toolset) and plugins; the session service it only flushes, so one a
        model switch keeps stays open. It closes before the session service
        (the base runs the backend's closers before the store's).
        """
        runner, self._runner = self._runner, None
        self._root_agent = None
        self._build = None
        self._reply_agents = None
        # A native app's agents stay (_native_root): ADK gives an agent one
        # parent, so a YAML naming a module-level agent cannot be loaded
        # again, and the runner's close leaves their toolsets reusable.
        if self._llm_logging_plugin:
            self._llm_logging_plugin.clear()
            self._llm_logging_plugin = None
        return [("runner", functools.partial(self._aclose_owned, runner, "runner"))]

    def _detach_store(self) -> list[Closer]:
        """The session service.

        It is *closed*, not merely dropped: the durable
        ``DatabaseSessionService`` holds a SQLAlchemy engine whose connection
        pool leaks otherwise. A model switch keeps it instead
        (``reinitialize(preserve_sessions=True)``), and ``_do_initialize``
        then reuses that same instance rather than building a replacement
        that would be discarded unclosed.
        """
        service, self._session_service = self._session_service, None
        close = functools.partial(self._aclose_owned, service, "session_service")
        return [("session_service", close)]

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
        ``McpToolset`` objects (ADK connects lazily) and appended after the
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
        told later (``resume_with_job_result``: a notice).
        Detection is by registry identity — the exact object the default
        registry issued, never a matching name — so a plain callable an
        application names after a long-running tool keeps its ordinary
        call-and-return contract instead of being told to leave a job pending
        that nothing will ever complete. Non-long-running tools and any
        already-wrapped tools (including toolset objects) pass through
        unchanged. Permission gating is unaffected: ADK gates via
        ``PermissionPlugin``, which resolves the same identity.

        Only the root agent's tools are wrapped (``_build_agent``): ADK pauses
        the whole invocation on a pending long-running call, inside a
        delegation too, so a sub-agent's starter stays an ordinary call that
        returns the job's id, and its finished job is told to the root
        (``resume_with_job_result``).
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
        *,
        root: bool,
    ) -> Agent:
        """Construct one ``LlmAgent`` from a config (with resolved sub-agents).

        Its model, generation settings and thinking come from the model
        initialization resolved for it, built by the model factory.

        Every agent but the root is a sub-agent its parent delegates to and
        gets back in the same turn (``single_turn``, no transfers; its
        ``inherit_context`` decides what it sees; a placeholder description
        when it has none), and ADK offers each of ``sub_agents`` to this agent
        as a tool named after it; those tools are bound to the delegation
        declaration as soon as this agent exists. Only the root's
        long-running tools are wrapped: a pending one would pause a
        delegation for good. See ``workflow/adk/delegation.py``. The root
        stays in chat mode.

        Raises:
            AgentGraphError: A sub-agent is named like one of this agent's
                own tools (its MCP tools, known only once they connect,
                excepted).
        """
        children = sub_agents or []
        built = build_adk_model(
            self._resolved_models[config.name], settings=self._settings, app_name=self.app_name
        )
        tools = self._assemble_agent_tools(config, service_map)
        if root:
            tools = self._wrap_long_running(tools)
        delegation.check_delegation_names(config, tools)
        fields = {"description": config.description}
        instruction = config.get_prompt(self._settings)
        if not root:
            fields.update(delegation.subagent_fields(config))
            instruction = delegation.subagent_instruction(config, instruction)
        agent = LlmAgent(
            name=config.name,
            model=built.model,
            # None at all for no prompt: ADK adds an instruction only when
            # there is one, and a provider always looks like one.
            instruction=_as_written(instruction) if instruction else "",
            tools=tools,
            sub_agents=children,
            planner=built.planner,
            generate_content_config=built.generate_content_config,
            **fields,
        )
        delegation.bind_delegation_tools(agent, children)
        return agent

    def _create_agents(self) -> Agent:
        """Create the agent hierarchy from configs, in dependency order.

        The graph is the one this initialization validated before anything
        was allocated (``_validate_agent_graph``), so duplicate names,
        dangling ``sub_agents`` references, self-references, cycles and
        shapes ADK cannot represent failed there with the offending agent
        named, instead of silently producing a coordinator with missing
        children. Called outside an initialization, it validates the graph
        itself.

        Construction then follows a topological order, so an agent is always
        built after the agents it delegates to — the previous leaves-then-
        coordinators split silently dropped a sub-agent that was itself a
        coordinator.

        Prompt factories are called under this manager's settings context (see
        ``AgentConfig.get_prompt``), so a prompt that reads settings sees the
        manager's instance rather than the global singleton.

        Returns:
            The root agent: the one no agent delegates to.

        Raises:
            AgentGraphError: If the configured graph is invalid.
        """
        # The graph validated for this build, taken once: a later build
        # validates its own.
        graph, self._validated_graph = getattr(self, "_validated_graph", None), None
        if graph is None:
            graph = validate_agent_graph(self._agent_configs, backend=self.backend_type)

        agent_map: dict[str, Agent] = {}
        service_map = self._get_service_tool_map()

        from agentic_cli.config import SettingsContext

        # Prompt factories run inside the manager's settings context.
        with SettingsContext(self._settings):
            for name in graph.build_order:
                config = graph.config_map[name]
                sub_agents = [agent_map[sub] for sub in config.sub_agents]
                agent_map[name] = self._build_agent(
                    config, service_map, sub_agents, root=name == graph.root_name
                )
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
        """The head of the app's plugin chain (``_plugin_chain``), not the whole of it.

        The permission gate, the tool-error answer and, with
        ``raw_llm_logging``, the raw LLM log of each call as the model
        answers it, in that order; the logging plugin is kept, since the
        manager drains its buffer. The gate writes the outside-content mark
        into the session state at the model calls of the agents that reply:
        the build's (``_build_app``), else the root's tree's. Task progress is the event processor's
        (``ADKEventProcessor``, from each event's state delta). The whole
        order is pinned by ``tests/workflow/adk/test_plugin_chain.py``.

        Returns:
            Those plugins, in chain order.
        """
        from agentic_cli.workflow.adk.build import reply_agents
        from agentic_cli.workflow.adk.tool_error_plugin import ToolErrorPlugin

        root = self._root_agent
        replies = self._reply_agents
        if replies is None:
            replies = reply_agents(root) if root else frozenset()
        # ToolErrorPlugin: ADK ends the turn when a tool raises; this turns the
        # exception into an error result the model can read. It records its
        # answer to an unknown tool, whose after-tool callbacks ADK skips.
        plugins: list = [
            PermissionPlugin(
                self._open_calls,
                marks=self._outside_marks,
                root=root.name if root else None,
                reply_agents=replies,
            ),
            ToolErrorPlugin(self._open_calls),
        ]

        if self._settings.raw_llm_logging:
            self._llm_logging_plugin = LLMLoggingPlugin(
                model_name=self.model,
                app_name=self._settings.app_name,
                include_messages=True,
                include_raw_parts=True,
                # A request is recorded as sent, after the fit guard
                # (SentRequestLoggingPlugin, last).
                log_requests=False,
                # Each call's records name its agent's own model.
                models=self._agent_model_refs(),
            )
            plugins.append(self._llm_logging_plugin)
            logger.info("llm_logging_plugin_enabled")
        else:
            self._llm_logging_plugin = None

        return plugins

    def _plugin_chain(
        self, replies: frozenset[str], cache: ContextCacheConfig | None
    ) -> list:
        """The app's whole plugin chain, top to bottom, whoever built its agents.

        ADK runs each callback of each plugin in turn and stops at the first
        that returns a value; an agent's own callbacks (a native ADK YAML
        app's) run after every plugin, and only when none answered. So:

        - ``MainAgentPlugin`` first: it records the running agent, which the
          summarizer reads, and makes its sub-agent run the active run.
        - ``_init_plugins``: the permission gate (a denial is the call's
          answer), the tool-error answer, the raw log as the model answers.
        - ``SubAgentOutcomePlugin``, when the tree delegates: a sub-agent's
          final answer that a model refused or cut is said so (and marked) as
          its parent reads it; after the raw log, which records the answer as
          given, and it returns nothing, so every plugin after it still runs.
        - ``ForeignReasoningPlugin``: a model switch keeps the session, and no
          model is sent the thoughts another provider wrote (which it would
          refuse), before the request is measured.
        - ``ContextCachePlugin``, when the app caches: one Gemini cache
          attempt a turn, not one per tool step.
        - ``InterruptedCallsPlugin`` before the cap: it records every
          finished result, for the calls a cancel leaves open, and the cap
          returns its cut, which ends ADK's after-tool callbacks.
        - ``ToolResultCapPlugin``: every result over the limit is cut, but an
          agent's with after-tool callbacks of its own, which a cut would
          skip (``build.own_after_tool_agents``).
        - ``MemoryIndexPlugin``, when the main agent has a memory tool:
          before the overflow handler and the fit guard, which keep it.
        - ``OverflowPlugin``: a call refused as too long is summarized (the
          main agent) or trimmed, retried once from the request the fit guard
          saw, then one clear answer; for an agent that replies, the turn's.
        - ``FitGuardPlugin`` last but the raw log: it measures the request
          every other plugin edited, and drops the oldest turns of one over
          its budget.
        - ``SentRequestLoggingPlugin``, with ``raw_llm_logging``: the request
          as sent.

        Args:
            replies: The agents that reply to the user (``build.reply_agents``).
            cache: The app's context cache, or None.
        """
        from agentic_cli.workflow.adk.build import delegated_agents, own_after_tool_agents
        from agentic_cli.workflow.adk.foreign_reasoning_plugin import ForeignReasoningPlugin

        root = self._root_agent
        kinds = {name: model.record.kind for name, model in self._resolved_models.items()}
        plugins: list = [MainAgentPlugin(self.agent_runs), *self._init_plugins()]
        if delegated_agents(root):
            from agentic_cli.workflow.adk.subagent_outcome_plugin import SubAgentOutcomePlugin

            plugins.append(SubAgentOutcomePlugin())
        plugins.append(ForeignReasoningPlugin({name: kind.value for name, kind in kinds.items()}))
        if cache is not None:
            from agentic_cli.workflow.adk.context_cache_plugin import ContextCachePlugin

            plugins.append(ContextCachePlugin())
        cap = ToolResultCapPlugin(
            self._settings.tool_result_max_chars,
            tool_output_dir(self._settings.app_name),
            skip_agents=own_after_tool_agents(root),
        )
        plugins.append(InterruptedCallsPlugin(self._open_calls, cap, runs=self.agent_runs))
        plugins.append(cap)
        if self._memory_index_wanted(root.name):
            from agentic_cli.workflow.adk.memory_index_plugin import MemoryIndexPlugin

            plugins.append(MemoryIndexPlugin(root=root.name, index_for=self._memory_index_now))
        fit_guard = FitGuardPlugin(self._budgets, kinds)
        plugins.append(OverflowPlugin(
            root=root.name,
            reply_agents=replies,
            budgets=self._budgets,
            kinds=kinds,
            models=self._agent_model_refs(),
            summarize_now=self._summarize_now,
            fit_guard=fit_guard,
        ))
        plugins.append(fit_guard)
        if self._llm_logging_plugin is not None:
            plugins.append(SentRequestLoggingPlugin(self._llm_logging_plugin))
        return plugins

    def _agent_model_refs(self) -> dict[str, str]:
        """Each agent's model reference, by name.

        Every agent built from a config, and each of a native ADK YAML app's
        agents whose model resolved (``_resolve_extra_models``); usage and the
        raw log name the default model for any other.
        """
        return {name: str(model.ref) for name, model in self._resolved_models.items()}

    def _make_session_service(self) -> BaseSessionService:
        """Build the ADK session service of its ``session_store``
        (``adk/sessions.py::make_session_service``, which its reader opens
        too): in memory, or durable on sqlite or Postgres."""
        return make_session_service(self._settings, self.session_store)

    @property
    def _stored(self) -> ServiceSessions | None:
        """The session hooks over its session service, None before it has one."""
        if self._session_service is None:
            return None
        return ServiceSessions(self._session_service)

    async def _do_initialize(self) -> None:
        """ADK-specific initialization: session service, agents, the app and its runner.

        The agents are built from the agent configs, or by ADK from a native
        ADK YAML (``_native_root``): the one place the two kinds of app
        differ. The app is then built from the tree alone (``_build_app``),
        so both run alike: the whole plugin chain, the main agent's
        conversation summarized at the trigger of its context numbers
        (``_budgets``, which the base computed before any service was built)
        with ADK's after-turn pass switched off
        (``NoAfterTurnCompactionRunner``), and an event processor that knows
        which agents reply.
        """
        from agentic_cli.workflow.adk.build import delegated_agents

        logger.info("initializing_services", app_name=self.app_name)

        # Create the session service (durable DatabaseSessionService by
        # default; InMemory only when session_store='memory') — unless one is
        # already held, which is how reinitialize(preserve_sessions=True)
        # carries live conversations across without building a replacement
        # that would then be discarded unclosed.
        if self._session_service is None:
            self._session_service = self._make_session_service()

        # Create agent hierarchy — by ADK from a native YAML, or from configs.
        if self._adk_config_path:
            self._root_agent = self._native_root()
        else:
            self._root_agent = self._create_agents()

        build = self._build = self._build_app()
        # The plugins belong to the App; ADK deprecates Runner(plugins=...).
        app = App(
            name=self.app_name,
            root_agent=build.root,
            plugins=list(build.plugins),
            context_cache_config=build.cache,
            events_compaction_config=build.compaction,
        )
        self._runner = NoAfterTurnCompactionRunner(app=app, session_service=self._session_service)
        # A call's usage names its agent's own model and context numbers (the
        # default model stands in for an agent without one); only the agents
        # that reply write the reply, another's text is a sub-agent's work.
        self._event_processor = ADKEventProcessor(
            model=self.model,
            on_event=self._on_event,
            root=build.root.name,
            models=build.models,
            budgets=self._budgets,
            reply_agents=build.reply_agents,
            runs=self.agent_runs,
            delegates=delegated_agents(build.root),
        )

        logger.info(
            "services_initialized",
            model=self.model,
            agent_name=build.root.name,
            required_managers=list(self._required_managers),
            context_cache=build.cache is not None,
            summarized=build.compaction is not None,
        )

    def _build_app(self) -> ADKBuild:
        """What the app is built from: the root agent's tree and what the run reads of it.

        From the tree alone (``_root_agent``, set first), whoever built it,
        through the hooks a subclass may override (``_plugin_chain``,
        ``_compaction_config``, ``_context_cache_config``,
        ``_agent_model_refs``). The reply set is computed once, and every
        part reads it: the gate through ``_reply_agents``.
        """
        from agentic_cli.workflow.adk import build as adk_build

        root = self._root_agent
        replies = self._reply_agents = adk_build.reply_agents(root)
        cache = self._context_cache_config()
        return adk_build.ADKBuild(
            root=root,
            reply_agents=replies,
            plugins=tuple(self._plugin_chain(replies, cache)),
            compaction=self._compaction_config(),
            cache=cache,
            models=self._agent_model_refs(),
        )

    def _compaction_config(self) -> EventsCompactionConfig | None:
        """The main agent's conversation summarized at its trigger; None without a budget.

        ADK checks before each model call of every agent; the summarizer
        writes only for the main agent, on its own model (thinking off,
        ``SUMMARY_MAX_TOKENS``), keeping its ``keep`` tokens word for word.
        ``event_retention_size`` 0: the summarizer picks the range itself.
        ADK marks the config experimental; the CLI logs that notice at debug.
        A main agent whose model did not resolve (a native ADK YAML app's,
        ``_resolve_extra_models``) has no budget, and is not summarized.
        """
        if self._root_agent.name not in self._budgets:
            return None
        return EventsCompactionConfig(
            summarizer=self._summarizer(),
            token_threshold=self._budgets[self._root_agent.name].trigger,
            event_retention_size=0,
        )

    def _summarizer(self, **options: Any) -> AgentCliSummarizer:
        """A summarizer of the main agent's conversation, on its own model."""
        root = self._root_agent.name
        return AgentCliSummarizer(
            root=root,
            model=self._simple_model(SUMMARY_MAX_TOKENS, agent=root),
            budget=self._budgets[root],
            settings=self._settings,
            app_name=self.app_name,
            on_written=self._forget_memory_index,
            **options,
        )

    async def _summarize_now(self, callback_context: Any) -> WrittenSummary | None:
        """Summarize the main agent's conversation now, for a call refused as too long.

        Within that call's turn, on its session and session service, as ADK
        compacts before a model call, but whatever the last prompt's size and
        whatever the saving (``OverflowPlugin``).

        Returns:
            The summary written and how many turns it kept, or None when
            nothing was summarized.

        Raises:
            Exception: Why the summary failed (the summarizer's
                ``last_error``), for the plugin to report.
        """
        context = callback_context.get_invocation_context()
        summarizer = self._summarizer(manual=True, source="summarization")
        if await compact_session(
            session=context.session,
            session_service=context.session_service,
            root_agent=self._root_agent,
            summarizer=summarizer,
        ):
            summarizer.report_stored()
            return WrittenSummary(summarizer.last_summary, summarizer.last_kept_turns)
        if summarizer.last_error is not None:
            raise summarizer.last_error
        return None

    def _context_cache_config(self) -> ContextCacheConfig | None:
        """The app's context cache, or None for no caching.

        ADK caches for every agent of an app through one config, so it is set
        when ``prompt_caching_enabled`` is on and any agent's model caches (an
        agent without a model is resolved to the default; a native ADK YAML
        app's agent to the model ADK runs it on, ``_resolve_extra_models``,
        and one whose model did not resolve is not counted). ADK then marks
        cache breakpoints on every Claude request, and
        keeps a server-side cache per Gemini agent from its second turn
        (Google bills its storage while it lives); GPT and the configured
        providers' models are untouched.

        ADK's defaults stand: a 1800-second TTL, which keeps Claude on its
        cheaper 5-minute cache, and a Gemini cache replaced after 10
        invocations. Only creating a Gemini cache gets a bound,
        ``_CACHE_CREATE_TIMEOUT_MS``.
        """
        if not self._settings.prompt_caching_enabled:
            return None
        models = list(self._resolved_models.values())
        if not any(m is not None and m.record.cache in _CACHING for m in models):
            return None
        return ContextCacheConfig(
            create_http_options=types.HttpOptions(timeout=_CACHE_CREATE_TIMEOUT_MS)
        )

    def _run_config(self) -> RunConfig | None:
        """A turn's ``RunConfig``: the agents' synchronous tools run in ADK's tool thread pool.

        Without the pool ADK calls a synchronous function tool on the event
        loop, which a two-minute sandbox cell then holds: the spinner, the job
        monitor and Ctrl+C all wait for it. In the pool only the function
        leaves the loop (plugins, callbacks and async tools stay on it), with
        a copy of the turn's context variables, and the calls of one model
        step overlap, as on Deep Agents.
        """
        return RunConfig(
            tool_thread_pool_config=ToolThreadPoolConfig(max_workers=_TOOL_THREADS)
        )

    def _manages_context(self) -> bool:
        """True: each agent with a resolved model is summarized and trimmed by its budget.

        A native ADK YAML app's agents too: each is resolved by the model ADK
        runs it on (``_resolve_extra_models``), and one whose model does not
        resolve has no budget, so the fit guard and the overflow handler
        leave it alone.
        """
        return True

    def _detect_required_managers(self) -> set[str]:
        """The services the agents' registered tools declare.

        The configs' (the base's), and a native ADK YAML app's: the tools of
        the tree it loaded when it validated its agents, found as the
        permission gate finds them (``tool_definition``: a callable the
        registry issued, or one of ADK's own function tools around one), so
        a ``kb_*`` tool the YAML names gets its knowledge base.
        """
        required = super()._detect_required_managers()
        root = self._loaded_native_root
        if root is None:
            return required
        from agentic_cli.workflow.adk.build import walk
        from agentic_cli.workflow.adk.permission_plugin import tool_definition

        for agent in walk(root):
            for tool in getattr(agent, "tools", None) or ():
                definition = tool_definition(tool)
                if definition is not None:
                    required.update(definition.requires)
        return required

    def _resolve_extra_models(
        self, roles: "RoleMap", providers: "Mapping[str, Provider]"
    ) -> "dict[str, ResolvedModel]":
        """A native ADK YAML app's agents' models (the base's hook), by name.

        Its agents run on the models ADK resolves for them: the one its YAML
        names (a model string or ``model_code``), else its parent's, else
        ADK's default. Each is resolved through the catalog by that model's
        name, as an agent config naming it would be. Leniently: an agent
        whose model ADK or the catalog cannot resolve is left out with a
        warning, and has no context budget.
        """
        root = self._loaded_native_root
        if root is None:
            return {}
        from agentic_cli.workflow.adk.build import walk

        resolved: dict[str, ResolvedModel] = {}
        for agent in walk(root):
            if not isinstance(agent, LlmAgent):
                continue
            spec: str | None = None
            try:
                spec = agent.canonical_model.model
                config = AgentConfig(name=agent.name, prompt="", model=spec)
                resolved[agent.name] = self._resolve_for(config, roles, self._catalog, providers)
            except (ValueError, ImportError) as exc:
                logger.warning(
                    "adk_agent_model_unresolved", agent=agent.name, model=spec, error=str(exc)
                )
        return resolved

    def _root_memory_tools(self, root: str | None = None) -> tuple[str, ...]:
        """The memory tools the main agent has, by name: the built root's.

        Read from the agent as built (``tool_definition``, as the permission
        gate finds a tool), so a native ADK YAML app's root that names a
        memory tool carries the memory index too. Before the root is built,
        its config's (the base's).
        """
        built = self._root_agent
        if built is None or built.name != (root or self._root_agent_name):
            return super()._root_memory_tools(root)
        from agentic_cli.memory import ALL_MEMORY_TOOLS
        from agentic_cli.workflow.adk.permission_plugin import tool_definition

        names = [
            definition.name
            for definition in (tool_definition(tool) for tool in getattr(built, "tools", ()) or ())
            if definition is not None and definition.name in ALL_MEMORY_TOOLS
        ]
        return tuple(dict.fromkeys(names))

    def _validate_agent_graph(self) -> AgentGraph | None:
        """Reject an unbuildable agent graph before anything is allocated.

        Runs at the top of initialization — ahead of model listings, service
        construction and the session service — so a static configuration error
        costs no network call and leaves nothing to roll back. The graph is
        kept for ``_create_agents``, which builds this initialization's agents
        from it rather than validating again.

        A native ADK YAML app's agents are built here the first time, by
        ADK (``_native_root``), and kept for ``_do_initialize``: its code is
        imported now, and a root that is not an ``LlmAgent`` or a task-mode
        agent is refused before anything else is done.

        Returns:
            The graph; for a native ADK YAML app, one naming its root alone
            (ADK built the tree, from no configs).
        """
        if self._adk_config_path:
            root = self._native_root()
            self._required_managers = self._detect_required_managers()
            return AgentGraph(config_map={}, build_order=(), root_name=root.name)
        self._validated_graph = validate_agent_graph(
            self._agent_configs, backend=self.backend_type
        )
        return self._validated_graph

    def _native_root(self) -> LlmAgent:
        """The native ADK YAML app's agents, loaded the first time they are asked for.

        Once per manager: ADK gives an agent one parent, so a YAML naming a
        module-level agent (``code:``, ADK's usual form) cannot be loaded
        twice in a process. A retry after a failed start, a model switch and
        a start after ``cleanup()`` build the app on the same tree: the
        closed runner closed its toolsets, and an MCP toolset starts its
        server again for the next runner. An edit of the YAML takes effect
        in a new manager.

        The settings' API keys are exported first (the base exports them
        only after validation), so a module the YAML names that reads a key
        as it is imported finds it, as it did when the YAML was loaded with
        the app.
        """
        if self._loaded_native_root is None:
            from agentic_cli.workflow.adk.build import load_native_root

            self._settings.export_api_keys_to_env()
            self._loaded_native_root = load_native_root(self._adk_config_path)
        return self._loaded_native_root

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

    async def _answer_left_open(self, session: Session) -> Session:
        """Answer the calls nothing answered (``interrupted_calls_plugin.left_open``).

        A crash, or a cancel from before ``abort_signal`` was set, leaves
        them, and so does a stop whose own answers to the calls ADK's seals
        missed could not be written (``_answer_unsealed``); ADK would keep a
        long-running one in every request, which Claude and OpenAI refuse,
        and drop any other one with a warning.

        ADK keeps two kinds of call open on purpose. A task-mode agent's
        delegation cannot be built (a native ADK YAML app with one is
        refused, ``build.load_native_root``). Its client calls for a human
        (``adk_request_confirmation``, ``adk_request_credential``,
        ``adk_request_input``: a tool that requires confirmation, an MCP
        toolset that needs auth) are answered as interrupted, as any other:
        the manager offers no way to answer them.

        Raises:
            ConversationChangedError: Another process changed the session
                since it was read (ADK's ``StaleSessionError``), before the
                run: the turn raises it, as it cannot start.
        """
        answers = left_open(session.events, self._open_calls)
        for event in answers:
            try:
                await self._session_service.append_event(session, event)
            except StaleSessionError:
                raise ConversationChangedError(session.id) from None
        if answers:
            logger.debug("left_open_calls_answered", count=sum(
                len(event.content.parts) for event in answers
            ))
        return session

    async def _answer_unsealed(self, ref: SessionRef) -> None:
        """Answer, from the record, the calls a stopped turn's seals missed; then clear it.

        ADK seals a stopped invocation's open calls pairing a result with a
        call by id alone, so a call whose id a sub-agent's result used is
        left open (``interrupted_calls_plugin.unsealed``): answered here, a
        finished one keeps its result and one at its prompt was not run.
        Run shielded, so a second Ctrl+C neither loses the answers nor
        clears the record before they are read. Answers that cannot be
        written are given at the next turn (``left_open``), from no record.
        """
        try:
            session = await self._session_service.get_session(
                app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
            )
            answers = (
                unsealed(session.events, self._open_calls, self.agent_runs)
                if session is not None else []
            )
            for event in answers:
                await self._session_service.append_event(session, event)
            if answers:
                logger.debug("unsealed_calls_answered", count=sum(
                    len(event.content.parts) for event in answers
                ))
        except Exception as error:
            logger.debug(
                "unsealed_calls_not_answered", error_type=type(error).__name__, exc_info=error,
            )
        finally:
            self._open_calls.clear(ref)

    def _seed_outside_content(self, ref: SessionRef, session: Session) -> dict[str, Any]:
        """Seed the marks from the session's state; the state delta a turn writes.

        A conversation with events but no record is from before 0.7.0 and
        counts as marked; a new one records ``""``. A mark this process holds
        that the state lacks is written, never ``""`` over it.
        """
        value = session.state.get(OUTSIDE_CONTENT_KEY)
        if isinstance(value, str):
            self._outside_marks.seed(ref, value)
            return {}
        self._outside_marks.seed(ref, BEFORE_0_7_0 if session.events else "")
        return {OUTSIDE_CONTENT_KEY: self._outside_marks.get(ref) or ""}

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

        Runs as one turn (the base's ``_turn``), which holds the manager's
        turn lock for the whole stream: the ADK plugins' event buffers are
        manager-scoped, so overlapping turns would drain each other's events.
        Overlapping callers queue; the lock is released on cancellation.
        Admission also re-verifies that the backend is still live (a cleanup
        can land between initialization and the turn lock). A turn that
        fails yields one ``ERROR`` and ends (``_run_and_stream``, the base
        ``process()``'s contract).

        Args:
            message: User message
            user_id: User identifier
            session_id: Optional session identifier

        Yields:
            WorkflowEvent objects representing workflow output
        """
        async with self._turn(session_id, user_id) as ref:
            logger.info("processing_message", message_length=len(message))
            session = await self._get_or_create_session(ref.user_id, ref.session_id)
            # A crash, an old cancel, or a stop whose answers could not be
            # written may have left calls without a result.
            session = await self._answer_left_open(session)
            state_delta = {
                **turn_state(session, message),
                **self._seed_outside_content(ref, session),
            }

            new_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )

            # aclosing: a consumer that closes this generator closes the
            # run now, inside the turn, so its calls are answered first.
            async with contextlib.aclosing(self._run_and_stream(
                session_id=ref.session_id,
                user_id=ref.user_id,
                new_message=new_message,
                # The conversation's row (workflow/adk/sessions.py) and its mark.
                state_delta=state_delta,
            )) as events:
                async for event in events:
                    yield event

    async def _run_and_stream(
        self,
        *,
        session_id: str,
        user_id: str,
        new_message: "types.Content",
        state_delta: dict[str, Any] | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run the ADK runner for one invocation and stream WorkflowEvents.

        Shared by ``process()`` (user message) and ``_deliver_job_notice()``
        (a job's notice). The caller
        must already be inside the turn (the base's ``_turn``) with the
        session prepared. ``state_delta`` is saved with ``new_message`` (a
        user turn's row).

        What is reported from inside a model call (a summary that starts, its
        usage, what it did) is merged into the stream as it happens
        (``workflow/live_events.py``), so it shows while the runner still
        waits on the call. Retries are the provider client's
        (``HttpRetryOptions``).

        A run that fails yields the turn's one ``ERROR``
        (``WorkflowEvent.turn_failed``, the base ``process()``'s contract),
        last, and ends: the provider's error behind ADK's plugin-manager
        wrapper (``_plugin_cause``), or ``ConversationChangedError`` for a
        session another process changed. A run that already yielded an
        ``ERROR`` that is not recoverable (the overflow handler's) yields no
        second one.
        """
        event_count = 0
        # Each turn reports the task list it first sees.
        self._event_processor.start_turn()
        # An ERROR the turn has yielded that is not recoverable: its one.
        reported_failure = False
        failure: BaseException | None = None
        live = live_events.LiveEvents()
        # Set when the turn is stopped: ADK then answers, as it closes, the
        # calls the invocation leaves open (InterruptedCallsPlugin words them).
        aborted = asyncio.Event()
        with live_events.reporting(live):
            stream = self._runner.run_async(
                session_id=session_id,
                user_id=user_id,
                new_message=new_message,
                state_delta=state_delta,
                run_config=self._run_config(),
                abort_signal=aborted,
            )
            try:
                # aclosing: a cancelled turn closes the merge, which closes the
                # runner, and its tasks, now, in the task that iterates it.
                # A stop sets the abort signal before the runner is cancelled.
                merging = live_events.merged(stream, live, on_stop=aborted.set)
                async with contextlib.aclosing(merging) as items:
                    try:
                        async for item in items:
                            if isinstance(item, WorkflowEvent):
                                reported = self._apply_event_hook(item)
                                if reported:
                                    reported_failure |= _fails_turn(reported)
                                    event_count += 1
                                    yield reported
                                continue
                            async for workflow_event in self._workflow_events(item, session_id):
                                reported_failure |= _fails_turn(workflow_event)
                                event_count += 1
                                yield workflow_event
                    except StaleSessionError:
                        failure = ConversationChangedError(session_id)
                    except RuntimeError as error:
                        failure = _plugin_cause(error)
                    except Exception as error:
                        failure = error
            finally:
                # Every call of the turn is answered now, sealed or returned,
                # but for one a stop's seals took for answered (they pair by
                # id alone): it is answered here, from the record, before the
                # record is cleared. One a failure left open is answered at
                # the next turn (left_open) as interrupted: the state it
                # changed, which ADK stores with its result, is gone.
                ref = self.session_ref(session_id, user_id)
                if aborted.is_set():
                    # Should those answers not be written, left_open gives
                    # them as the user's.
                    self._open_calls.cancelled(ref)
                    await asyncio.shield(self._answer_unsealed(ref))
                else:
                    self._open_calls.clear(ref)

        # Drain any remaining LLM events after processing completes
        if self._llm_logging_plugin:
            for llm_event in self._llm_logging_plugin.drain_events():
                llm_event = self._apply_event_hook(llm_event)
                if llm_event:
                    event_count += 1
                    yield llm_event

        if failure is None:
            logger.info("message_processed", event_count=event_count)
            return
        # The ERROR is the report; the log is for diagnosis, its traceback at
        # debug, so nothing is said twice over the CLI's prompt.
        logger.info(
            "adk_turn_failed",
            error_type=type(failure).__name__,
            error=truncate(str(failure), _MAX_LOGGED_ERROR),
            reported=reported_failure,
        )
        logger.debug("adk_turn_failed", exc_info=failure)
        if not reported_failure:
            error = self._apply_event_hook(WorkflowEvent.turn_failed(failure))
            if error is not None:
                yield error

    async def _workflow_events(
        self, adk_event: Any, session_id: str
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """One ADK event's WorkflowEvents, after what the plugins buffered before it."""
        # LLM events from the logging plugin first (raw capture).
        if self._llm_logging_plugin:
            for llm_event in self._llm_logging_plugin.drain_events():
                llm_event = self._apply_event_hook(llm_event)
                if llm_event:
                    yield llm_event

        async for workflow_event in self._event_processor.process_event(adk_event, session_id):
            yield workflow_event

    async def _conversation_exists(self, ref: SessionRef) -> bool:
        """Whether ``ref``'s ADK session is stored, with events or none.

        What ``can_resume`` and a job's resume ask (the base's templates).
        The session service is durable by default (``session_store='sqlite'``),
        so a session normally survives a CLI restart; it is gone once deleted,
        or with ``session_store='memory'`` once the process restarted. It reads
        none of the session's events (``num_recent_events=0``): a long
        session read whole costs tens of milliseconds, and the status bar
        asks again every 30 seconds.
        """
        if self._session_service is None:
            return False
        session = await self._session_service.get_session(
            app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
            config=GetSessionConfig(num_recent_events=0),
        )
        return session is not None

    async def _deliver_job_notice(
        self, notice: str, record: Any, result: Any, ref: SessionRef
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run the agents on a finished job's notice (the base's ``resume_with_job_result``).

        The conversation is told in one user-role text, the notice, as on
        Deep Agents, whoever started the job: the call already has its result
        (the job's id, or "pending"), and a result for the call itself made
        ADK send the call, that result and nothing between (its rearranger),
        so the agent answered without the turns since. A call still open (a
        starter that returned nothing, a crash's) is answered first. ADK runs
        the notice as any user turn: on the root, or on the agent it
        transferred the conversation to. The notice marks the conversation as
        having read outside content, in its state too.
        """
        session = await self._session_service.get_session(
            app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
        )
        if session is None:  # deleted since the base looked
            logger.warning("job_resume_session_missing", job_id=record.job_id,
                           session_id=ref.session_id)
            return
        # A call still open gets its result before the notice: Claude and
        # OpenAI refuse a call without one.
        session = await self._answer_left_open(session)

        # The job's output is outside content: its notice marks the conversation.
        state_delta = self._seed_outside_content(ref, session)
        if self._outside_marks.mark(ref, JOB_RESULT):
            state_delta[OUTSIDE_CONTENT_KEY] = JOB_RESULT

        new_message = types.Content(role="user", parts=[types.Part.from_text(text=notice)])

        async with contextlib.aclosing(self._run_and_stream(
            session_id=ref.session_id,
            user_id=ref.user_id,
            new_message=new_message,
            state_delta=state_delta or None,
        )) as events:
            async for event in events:
                yield event

    @property
    def supports_compact(self) -> bool:
        """True unless the app is built and its main agent is not summarized.

        A main agent whose model did not resolve (a native ADK YAML app's)
        has no context budget and no summary (``compact`` raises
        ``NotImplementedError`` there). That is known once the app is built:
        before, the answer is True.
        """
        build = self._build
        return build is None or build.compaction is not None

    def _no_compact_support(self) -> NotImplementedError:
        """Why this app's main agent has no ``compact``."""
        return NotImplementedError(
            "The main agent's conversation is not summarized: its model has no "
            "context budget (it did not resolve through the model catalog)."
        )

    async def _compact_turn(
        self, session: SessionRef, focus: str | None
    ) -> CompactResult | None:
        """Summarize the main agent's conversation now (the base's ``compact``, ``/compact``).

        ADK's compaction runs once on a fresh read of the session
        (``compact_session``), with the main agent's summarizer asked for now
        (``manual``): written whatever the last prompt's size and the saving,
        ``focus`` spliced into its prompt. The range, the call, the event and
        what is reported are a summary's in a turn, but ``CONTEXT_TRIMMED``
        names ``"compact"`` and comes once ADK has stored the summary. The
        base runs it as a turn, so a turn or a job's resume never writes to
        the session between its read and the summary's append (ADK's
        ``StaleSessionError``), and nothing is written until the summary is:
        a cancel leaves the session as it was, and says nothing was made. A
        main agent without a context budget has none (``supports_compact``).

        Returns:
            What the summary did; None with no conversation yet, or nothing
            before the kept tail.

        Raises:
            NotImplementedError: The main agent is not summarized
                (``supports_compact``).
            Exception: Why the summary failed (the summarizer's
                ``last_error``, the provider's error).
        """
        if not self.supports_compact:
            raise self._no_compact_support()
        stored = await self._session_service.get_session(
            app_name=session.app_name,
            user_id=session.user_id,
            session_id=session.session_id,
        )
        if stored is None:
            return None
        summarizer = self._summarizer(manual=True, focus=focus)
        if await compact_session(
            session=stored,
            session_service=self._session_service,
            root_agent=self._root_agent,
            summarizer=summarizer,
        ):
            summarizer.report_stored()
            result = summarizer.last_result
            logger.info(
                "conversation_compacted",
                messages=result.messages_summarized,
                tokens_before=result.tokens_before,
                tokens_after=result.tokens_after,
                focus=focus is not None,
            )
            return result
        if summarizer.last_error is not None:
            raise summarizer.last_error
        return None

    # -------------------------------------------------------------------------
    # Sessions (native — DatabaseSessionService persists events continuously)
    # -------------------------------------------------------------------------

    async def session_exists(self, session_id: str, *, user_id: str | None = None) -> bool:
        """True if the store holds this session with any events.

        It reads the newest event alone (``num_recent_events=1``), not the
        whole conversation.

        Args:
            session_id: Session to look up.
            user_id: Owner of the session (defaults to ``settings.default_user``).
        """
        stored = self._stored
        if stored is None:
            return False
        return await stored.exists(self.session_ref(session_id, user_id))

    async def list_sessions(
        self, *, user_id: str | None = None, limit: int | None = None
    ) -> list[SessionInfo]:
        """A user's stored conversations, newest first.

        Each row comes from its session's state, which ADK's listing returns
        without the events; a conversation from before those keys is read
        once, for the rows returned (``workflow/adk/sessions.py``). A session
        without events is no conversation and is left out.

        Args:
            user_id: Their owner (defaults to ``settings.default_user``).
            limit: Only the newest this many (None: all of them).
        """
        stored = self._stored
        if stored is None:
            return []
        ref = self.session_ref(user_id=user_id)

        async def row_of(session_id: str) -> SessionInfo | None:
            return await self.session_info(session_id, user_id=ref.user_id)

        return await stored.list(ref.app_name, ref.user_id, limit=limit, row_of=row_of)

    async def session_info(
        self, session_id: str, *, user_id: str | None = None
    ) -> SessionInfo | None:
        """One stored conversation's row, or None when it is not stored.

        Args:
            session_id: The conversation.
            user_id: Its owner (defaults to ``settings.default_user``).
        """
        stored = self._stored
        if stored is None:
            return None
        return await stored.info(self.session_ref(session_id, user_id))

    async def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        """Delete a persisted session from the store.

        ADK's own delete answers nothing, for an unknown id too, so the
        session is looked up first, without its events
        (``num_recent_events=0``).

        Args:
            session_id: Session to delete.
            user_id: Owner of the session (defaults to ``settings.default_user``).

        Returns:
            False if there is no such session.
        """
        stored = self._stored
        if stored is None:
            return False
        return await stored.delete(self.session_ref(session_id, user_id))

    async def recent_messages(
        self, session_id: str, limit: int = 20, *, user_id: str | None = None
    ) -> list[dict]:
        """Recent text messages from the stored session (fact extraction, a resume's replay).

        Text only: a reply's thinking is left out. The user's and the agents'
        that reply (``build.reply_agents``: the root, and an agent it
        transferred the conversation to): a sub-agent's text is its work,
        which its parent read as a tool result, not a reply. A
        job notice (``job_notice``) is left out, as on Deep Agents: nobody
        typed it. ADK keeps no mark on the message, so it is known by its
        header.

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
        replying = self._build.reply_agents if self._build is not None else None
        out: list[dict] = []
        for event in session.events:
            author = getattr(event, "author", None)
            if replying is not None and author != "user" and author not in replying:
                continue
            content = getattr(event, "content", None)
            if content is None:
                continue
            role = getattr(content, "role", None)
            text = " ".join(
                p.text for p in (getattr(content, "parts", None) or [])
                if getattr(p, "text", None) and not getattr(p, "thought", False)
            )
            if role == "user" and text.startswith(JOB_NOTICE_HEADER):
                continue
            if text:
                out.append(
                    {"role": "user" if role == "user" else "assistant", "content": text}
                )
        return out[-limit:]
