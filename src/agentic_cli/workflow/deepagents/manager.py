"""The Deep Agents workflow manager (``orchestrator = "deepagents"``).

Every ``AgentConfig`` becomes one ``create_agent`` graph on its own chat model
(``agent_builder``), and the root agent runs each turn on the thread of the
turn's ``SessionRef`` (``sessions``). The stream's node outputs become
``WorkflowEvent``s (``event_processor``). Conversations live in a LangGraph
checkpointer opened from ``session_store``, and their rows in a store beside
it (the session index, written as each turn ends), which this manager owns:
their connections are closed on cleanup and by a failed initialization, and
a model switch keeps them. The agents' MCP sessions (``mcp_tools``) are its
too: they open at initialization, and every release closes them, so a model
switch reopens them. An agent's ``skills`` become the skill tools ADK offers
(``tools/skills/langchain_tools.py``), with ADK's instruction appended to its
prompt. With ``raw_llm_logging`` on, a turn's every model call is logged
(``LLM_REQUEST``/``LLM_RESPONSE``, in the stream and in
``~/.{app_name}/logs/llm_events.jsonl``). A sub-agent runs when its parent's ``task`` call
delegates to it (``task_tool.build_task_tool``, the framework's
``Delegations``): its own graph, apart from its parent's stream, in the run's
own task (``_run_agent``), on a checkpointer thread of its instance's own
(``sessions.instance_run_config``), which the instance's follow-up runs
continue. A long-running job that finishes
resumes its conversation with a turn of its own, opened by a notification to
the root agent (``resume_with_job_result``). The root agent's conversation is
summarized on demand by ``compact`` (``/compact``), with its own summarization
middleware, kept when the agents are built.

The lifecycle is the base's (``BaseWorkflowManager``): initialization,
``reinitialize`` and ``cleanup`` hold the lifecycle lock and then the turn
lock, turns enter through ``_turn``, a model switch is
transactional, carrying the conversation's services and the checkpointer
across, and a release closes everything under a shield. This manager
implements the release's hooks (``_detach_backend``, ``_detach_store``).

Installed by the ``deepagents`` extra: ``pip install 'agentic-cli[deepagents]'``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable

from deepagents.middleware.summarization import SUMMARIZATION_EVENT_KEY
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import var_child_runnable_config

from agentic_cli.config import BaseSettings
from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers
from agentic_cli.tools.registry import ToolDefinition, identify_tool
from agentic_cli.tools.skills import SkillAgents, SkillStore
from agentic_cli.tools.skills.langchain_tools import SKILL_TOOL_NAMES, make_skill_tools
from agentic_cli.workflow.base_manager import BaseWorkflowManager, Closer
from agentic_cli.workflow.checkpointers import (
    deepagents_sqlite_path,
    open_checkpointer,
    open_store,
)
from agentic_cli.workflow.config import (
    AgentConfig,
    AgentGraph,
    AgentGraphError,
    validate_agent_graph,
)
from agentic_cli.workflow import live_events
from agentic_cli.workflow.agents.control import KEPT_TOOLS
from agentic_cli.workflow.agents.delegate import AgentRun, RunAnswer, RunFailed, failure_reason
from agentic_cli.workflow.agents.runs import own_plan_and_tasks
from agentic_cli.workflow.context import (
    OVERFLOW_ERROR_CODE,
    CompactResult,
    is_context_overflow,
    subagent_overflow_message,
)
from agentic_cli.workflow.deepagents.agent_builder import Variant, build_agent_tree
from agentic_cli.workflow.deepagents.event_processor import (
    DeepAgentsEventProcessor,
    RunCallback,
    usage_event,
)
from agentic_cli.workflow.deepagents.mcp_tools import MCPSessions, without_clashes
from agentic_cli.workflow.deepagents.middleware import answer_outcome
from agentic_cli.workflow.deepagents.sessions import (
    CheckpointSessions,
    instance_run_config,
    instance_thread_id,
    job_notification,
    notice_message,
    run_config,
)
from agentic_cli.workflow.deepagents.summarization import (
    ReportingSummarizationMiddleware,
    brief_message,
    compact_cutoff,
    newest_answer,
    summary_event,
    write_summary,
)
from agentic_cli.workflow.deepagents.task_tool import SHARED_KEYS, inherited_messages
from agentic_cli.workflow.deepagents.tools import to_langchain_tools
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.langchain_models import build_chat_model
from agentic_cli.workflow.live_events import LiveEvents, merged
from agentic_cli.workflow.llm_log import LLMLogWriter
from agentic_cli.workflow.outside_content import JOB_RESULT

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from agentic_cli.models import ResolvedModel
    from agentic_cli.workflow.agents.policy import SubAgentPolicy
    from agentic_cli.workflow.sessions import SessionInfo, SessionRef

logger = Loggers.workflow()

#: Longest error text a failed turn's log line carries.
_MAX_LOGGED_ERROR = 500


def _with_instruction(prompt: str, instruction: str) -> str:
    """``prompt`` with ``instruction`` appended, joined as ADK appends one."""
    return f"{prompt}\n\n{instruction}" if prompt else instruction


class DeepAgentsWorkflowManager(BaseWorkflowManager):
    """Workflow manager on LangChain's ``create_agent`` and Deep Agents' middleware.

    A coordinator delegates through the framework's ``task`` tool, giving a
    sub-agent a brief (``agents/brief.py``) and reading back its result
    envelope; the framework's permission gate and tool-error middleware run
    in every agent's stack.

    Example:
        configs = [
            AgentConfig(name="coordinator", prompt="Coordinate.",
                        sub_agents=["researcher"]),
            AgentConfig(name="researcher", prompt="Research.",
                        description="Finds facts."),
        ]
        manager = DeepAgentsWorkflowManager(agent_configs=configs, settings=settings)

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
        *,
        subagents: SubAgentPolicy | None = None,
    ) -> None:
        """Initialize the workflow manager.

        Args:
            agent_configs: The agents; the root is the one nobody delegates to.
            settings: Application settings (``get_settings()`` if not provided).
            app_name: Application name for sessions, and what an OpenRouter
                provider is told (``settings.app_name`` if not provided).
            model: This manager's default model, a ``provider:model`` reference
                or a model id (see ``BaseWorkflowManager``).
            on_event: Optional hook to transform or drop events before they
                are yielded.
            subagents: How the sub-agents run (``subagent_policy``); None is
                ``SubAgentPolicy()`` (see ``BaseWorkflowManager``).
        """
        super().__init__(
            agent_configs=agent_configs,
            settings=settings,
            app_name=app_name,
            model=model,
            on_event=on_event,
            subagents=subagents,
        )

        # The checkpointer, and what owns its connection and the session
        # index's (an exit stack; the index's store is the session hooks').
        # A model switch keeps them (the base's _keeping), so neither its
        # release nor a rollback closes them.
        self._checkpointer: BaseCheckpointSaver | None = None
        self._checkpointer_cm: AbstractAsyncContextManager | None = None
        self._sessions: CheckpointSessions | None = None

        self._root: CompiledStateGraph | None = None
        # Every agent's graph, by name: what a sub-agent run runs on.
        self._agent_graphs: dict[str, CompiledStateGraph] = {}
        # Each sub-agent type's graph on each of its model choices, by (type,
        # choice): what a run on a choice runs on.
        self._variant_graphs: dict[tuple[str, str], CompiledStateGraph] = {}
        # The built-in general-purpose type's graph for each type that may start
        # it, by that type's name: what a run of it runs on.
        self._general_graphs: dict[str, CompiledStateGraph] = {}
        # Each agent's own tools' names as its model sees them (its tools, skill
        # tools and MCP tools), the framework's state and delegation tools left
        # out: what a task call's `tools` may name.
        self._own_tool_names: dict[str, frozenset[str]] = {}
        # The root agent's summarization middleware, which compact() uses.
        self._root_summarization: ReportingSummarizationMiddleware | None = None

        # The agents' MCP sessions, and which tool objects are theirs.
        self._mcp = MCPSessions()

        # The raw LLM log, made on the first turn that logs (it creates the file).
        self._llm_log: LLMLogWriter | None = None

        # Notices for the next turn to open with, as recoverable ERROR events
        # (an MCP server that could not start, an MCP tool skipped).
        self._pending_notices: list[WorkflowEvent] = []

        logger.debug(
            "deepagents_workflow_manager_created",
            app_name=self.app_name,
            model_override=model,
            agent_count=len(agent_configs),
        )

    @property
    def backend_type(self) -> str:
        """Return 'deepagents'."""
        return "deepagents"

    def _get_state_tools(self) -> list:
        """The LangGraph-library state tools (``Command`` / ``InjectedState``)."""
        from agentic_cli.tools.langgraph.state_tools import (
            get_plan,
            get_tasks,
            save_plan,
            save_tasks,
        )

        return [save_plan, get_plan, save_tasks, get_tasks]

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def _validate_agent_graph(self) -> AgentGraph:
        """Reject an unbuildable agent graph before anything is allocated."""
        return validate_agent_graph(
            self._agent_configs, backend=self.backend_type, policy=self.subagent_policy
        )

    def _manages_context(self) -> bool:
        """Every agent is summarized by its own context numbers."""
        return True

    def _backend_ready(self) -> bool:
        """True when the root agent and the session store are live."""
        return bool(
            self._initialized and self._root is not None and self._sessions is not None
        )

    async def _do_initialize(self) -> None:
        """Open (or keep) the checkpointer; build the agents, skills and MCP tools."""
        logger.info("initializing_services", app_name=self.app_name, backend="deepagents")

        # A preserving reinitialize keeps the live checkpointer: reuse it
        # rather than open a replacement.
        if self._checkpointer is None:
            await self._open_session_store()

        service_map = self._get_service_tool_map()

        from agentic_cli.config import SettingsContext

        # Prompt factories run inside the manager's settings context, once.
        with SettingsContext(self._settings):
            prompts = {
                config.name: config.get_prompt(self._settings)
                for config in self._agent_configs
            }
        tools: dict[str, list[BaseTool]] = {}
        for config in self._agent_configs:
            tools[config.name], prompts[config.name] = await self._agent_tools(
                config, service_map, prompts[config.name],
                model=self._resolved_models[config.name],
            )
        self._own_tool_names = {
            name: frozenset(t.name for t in own if t.name not in KEPT_TOOLS)
            for name, own in tools.items()
        }
        summarizers: dict[str, ReportingSummarizationMiddleware] = {}
        root = self._root_agent_name  # from the graph the base validated
        index_wanted = self._memory_index_wanted(root)
        roster_wanted = self._roster_wanted(root)
        graphs: dict[str, CompiledStateGraph] = {}
        variant_graphs: dict[Any, CompiledStateGraph] = {}
        variants = {
            (config.name, choice): Variant(
                config=config,
                chat_model=build_chat_model(
                    self._choice_models[(config.name, choice)],
                    settings=self._settings,
                    app_name=self.app_name,
                ),
                resolved=self._choice_models[(config.name, choice)],
                budget=self._choice_budgets.get((config.name, choice)),
            )
            for config in self._agent_configs
            if config.name != root
            for choice in config.model_choices
        }
        # The general-purpose agent of each type that may start one: its
        # spawner's tools and MCP servers (sessions of its own), its prompt,
        # its spawner's model and numbers.
        general_graphs: dict[str, CompiledStateGraph] = {}
        general: dict[str, Variant] = {}
        for spawner, config in self._delegations.general_purpose.items():
            model = self._general_model(spawner)
            # Deep Agents opens MCP sessions at initialization (ADK connects at
            # first use), so each general-purpose agent starts its spawner's
            # servers once more here: one that never answers is waited for once
            # more per spawner, and one that cannot start is a notice for it too.
            own, prompt = await self._agent_tools(
                config, service_map, config.get_prompt(self._settings), model=model,
                spawner=spawner,
            )
            general[spawner] = Variant(
                config=config,
                chat_model=build_chat_model(
                    model, settings=self._settings, app_name=self.app_name
                ),
                resolved=model,
                budget=self._budgets.get(spawner),
                tools=own,
                prompt=prompt,
            )
        self._root = build_agent_tree(
            self._agent_configs,
            chat_models={
                config.name: build_chat_model(
                    self._resolved_models[config.name],
                    settings=self._settings,
                    app_name=self.app_name,
                )
                for config in self._agent_configs
            },
            resolved=self._resolved_models,
            tools=tools,
            prompts=prompts,
            settings=self._settings,
            checkpointer=self._checkpointer,
            bound=self._identify_tool,
            budgets=self._budgets,
            summarizers=summarizers,
            open_calls=self._open_calls,
            marks=self._outside_marks,
            memory_index=self._memory_index_now if index_wanted else None,
            memory_index_cached=self._memory_index_cached if index_wanted else None,
            memory_index_forget=(
                self._forget_memory_index if index_wanted or roster_wanted else None
            ),
            roster=self._roster_now if roster_wanted else None,
            delegations=lambda: self._delegations,
            graphs=graphs,
            root=root,
            variants=variants,
            variant_graphs=variant_graphs,
            general=general,
            general_graphs=general_graphs,
        )
        self._agent_graphs = graphs
        self._variant_graphs = variant_graphs
        self._general_graphs = general_graphs
        self._root_summarization = summarizers.get(root)

        logger.info(
            "services_initialized",
            model=self.model,
            root=root,
            session_store=self.session_store,
            required_managers=list(self._required_managers),
        )

    async def _open_session_store(self) -> None:
        """Open the checkpointer ``session_store`` names and the index beside it.

        One exit stack owns both connections (``_checkpointer_cm``), so a
        release closes both, and a failure to open the second closes the
        first before it propagates. An in-memory saver is opened with its
        store, so no conversation of it predates the index: it needs no
        backfill.
        """
        kind = self.session_store
        where: dict[str, Any] = {
            "sqlite_path": (
                deepagents_sqlite_path(self._settings) if kind == "sqlite" else None
            ),
            "postgres_uri": self._settings.postgres_uri,
        }
        connections = contextlib.AsyncExitStack()
        try:
            saver, connection = await open_checkpointer(kind, **where)
            if connection is not None:
                connections.push_async_exit(connection.__aexit__)
            store, connection = await open_store(kind, **where)
            if connection is not None:
                connections.push_async_exit(connection.__aexit__)
        except BaseException:
            await self._close_connection(connections)
            raise
        self._checkpointer, self._checkpointer_cm = saver, connections
        self._sessions = CheckpointSessions(saver, store, backfill=kind != "memory")

    async def _agent_tools(
        self,
        config: AgentConfig,
        service_map: dict[str, Callable],
        prompt: str,
        *,
        model: "ResolvedModel",
        spawner: str | None = None,
    ) -> tuple[list[BaseTool], str]:
        """``config``'s tools as its model sees them, and its prompt with its skills' instruction.

        Its registered tools, then its skill tools (before the MCP tools: an
        MCP tool named like one is skipped), then its MCP servers' tools,
        opened now on sessions of this agent's own. ``model`` is the model it
        runs on (the provider its skill tools are worded for): the
        general-purpose agent's is its spawner's, and ``spawner`` names that
        type for the notices about its MCP servers.
        """
        own = to_langchain_tools(self._build_tools(config, service_map))
        if config.skills:
            skill_tools, instruction = self._skill_tools(config, own, model=model)
            own = [*own, *skill_tools]
            prompt = _with_instruction(prompt, instruction)
        if config.mcp_servers:
            own = await self._with_mcp_tools(config, own, spawner=spawner)
        return own, prompt

    def _skill_tools(
        self,
        config: AgentConfig,
        tools: list[BaseTool],
        *,
        model: "ResolvedModel | None" = None,
    ) -> tuple[list[BaseTool], str]:
        """``config``'s skills as skill tools, and the instruction its prompt gets.

        The skills resolve as on ADK: an absolute path to a skill folder, or a
        name under ``skills_dirs``. A skill that names an agent type starts it
        when loaded, as on ADK.

        Raises:
            ValueError: A skill cannot be found or loaded (``SkillStore``).
            AgentGraphError: The agent's own tools already use a skill tool's
                name; two tools of one name would leave one unreachable.
        """
        taken = [tool.name for tool in tools if tool.name in SKILL_TOOL_NAMES]
        if taken:
            *rest, last = SKILL_TOOL_NAMES
            raise AgentGraphError(
                f"Agent {config.name!r} has skills, whose tools are "
                f"{', '.join(rest)} and {last}, so its own tools cannot include "
                f"{taken[0]!r}; rename that tool."
            )
        skills = SkillStore(self._settings.skills_dirs or []).resolve(config.skills)
        skill_tools, instruction = make_skill_tools(
            skills,
            provider=(model if model is not None else self._resolved_models[config.name]).record.kind,
            # The base validated the graph before this build, so the root is set.
            agents=SkillAgents(
                config.name, lambda: self._delegations,
                root=config.name == self._root_agent_name,
            ),
        )
        logger.debug(
            "skill_tools_attached", agent=config.name, skills=[s.name for s in skills]
        )
        return skill_tools, instruction

    async def _with_mcp_tools(
        self, config: AgentConfig, tools: list[BaseTool], *, spawner: str | None = None
    ) -> list[BaseTool]:
        """``tools`` and the tools of ``config``'s MCP servers, opened now.

        A server that cannot start, and a tool whose name is taken, become
        notices for the next turn; the agent runs without them. A
        general-purpose agent's notices name its ``spawner`` too
        (``metadata["spawner"]``), since every type that may start one has an
        agent of that name.
        """
        loaded, unavailable = await self._mcp.start(config.name, config.mcp_servers)
        kept, skipped = without_clashes(config.name, tools, loaded)
        for message, code in [
            *((message, "mcp_unavailable") for message in unavailable),
            *((message, "mcp_tool_skipped") for message in skipped),
        ]:
            notice = WorkflowEvent.error(message, error_code=code, recoverable=True)
            notice.metadata["agent"] = config.name
            if spawner is not None:
                notice.metadata["spawner"] = spawner
            self._pending_notices.append(notice)
        return [*tools, *kept]

    def _identify_tool(self, obj: object) -> ToolDefinition | None:
        """What a tool object is: an open MCP session's tool, else a framework tool.

        Every agent's permission gate asks this first (``build_agent_tree``'s
        ``bound``).
        """
        definition = self._mcp.identify(obj)
        return definition if definition is not None else identify_tool(obj)

    def _detach_backend(self) -> list[Closer]:
        """The agents and their MCP sessions; the notices about them go with them.

        The MCP sessions are closed on every release, a model switch's
        included (the next initialization opens them again). They move to
        their own instance at once, so a later initialization starts its
        sessions on a fresh one whatever closing these takes;
        ``MCPSessions.aclose`` never raises, and each session's owner leaves
        it even if the close is cancelled.
        """
        self._root = None
        self._agent_graphs = {}
        self._variant_graphs = {}
        self._general_graphs = {}
        self._root_summarization = None
        self._own_tool_names = {}
        self._pending_notices = []
        sessions, self._mcp = self._mcp, MCPSessions()
        return [("mcp_sessions", sessions.aclose)]

    def _detach_store(self) -> list[Closer]:
        """The checkpointer and the session index, whose connections one exit stack owns.

        The connections are closed, not merely dropped. A model switch keeps
        them instead (``reinitialize(preserve_sessions=True)``), also through
        the rollback of a failed attempt, and ``_do_initialize`` then reuses
        the checkpointer it finds.
        """
        connection, self._checkpointer_cm = self._checkpointer_cm, None
        self._checkpointer = None
        self._sessions = None
        return [("checkpointer", functools.partial(self._close_connection, connection))]

    async def _agent_tool_names(self, agent: str) -> frozenset[str]:
        """The names a ``task`` call's ``tools`` may give for ``agent``: its built
        tools'."""
        names = self._own_tool_names.get(agent)
        return names if names is not None else await super()._agent_tool_names(agent)

    async def _close_connection(self, connection: "AbstractAsyncContextManager | None") -> None:
        """Exit what owns the checkpointer's and the index's connections.

        Never raises.
        """
        if connection is None:
            return
        try:
            await connection.__aexit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail
            logger.warning("resource_close_failed", resource="checkpointer", error=str(exc))

    # -------------------------------------------------------------------------
    # Turns
    # -------------------------------------------------------------------------

    async def process(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run one user turn through the agents.

        Holds the turn lock for the whole stream (turns serialize) and runs
        under this manager's settings, services and active turn, which reach
        every tool, sub-agents' included. A turn that fails yields one
        ``ERROR`` and ends; the checkpoint keeps the user's message and
        whatever ran before the failure, and the next turn continues after
        it.

        Args:
            message: The user's message.
            user_id: The user.
            session_id: The conversation (the manager's current one if None).

        Yields:
            The turn's ``WorkflowEvent``s, each through ``on_event``.
        """
        async with self._turn(session_id, user_id) as ref:
            logger.info("processing_message", message_length=len(message))
            for notice in self._take_notices():
                yield notice
            # What the conversation's checkpoint says of outside content.
            self._outside_marks.seed(ref, await self._sessions.outside_content(ref))
            # aclosing: a consumer that closes this generator closes the
            # run now, inside the turn, so its cancel is on record first.
            async with contextlib.aclosing(self._run_turn(
                ref, {"messages": [HumanMessage(content=message)]}
            )) as events:
                async for event in events:
                    yield event

    async def _conversation_exists(self, ref: "SessionRef") -> bool:
        """Whether ``ref``'s root thread holds at least one message, in this app.

        What ``can_resume`` and a job's resume ask (the base's templates).
        With a durable ``session_store`` (sqlite, the default, or postgres) a
        job that finishes after a restart still resumes; after the session
        was deleted, or with ``memory`` once the process restarted, it does
        not. The call id is not needed: the root agent is told in a message
        of its own, not as a second result for the call.
        """
        return self._sessions is not None and await self._sessions.exists(ref)

    async def _conversation_state(self, ref: "SessionRef") -> dict[str, Any]:
        """The state a run the user starts in ``ref``'s conversation starts from.

        The conversation's outside-content mark, seeded from its checkpoint
        and read from the manager's marks, under the key its runs share it
        by (the state a ``task`` call hands its run).
        """
        self._outside_marks.seed(ref, await self._sessions.outside_content(ref))
        mark = self._outside_marks.get(ref)
        return {"outside_content": mark} if mark else {}

    async def _keep_conversation(self, ref: "SessionRef", objective: str) -> None:
        """Write the conversation's row when it has none (``CheckpointSessions.keep``)."""
        if self._sessions is not None:
            await self._sessions.keep(ref, objective)

    async def _inherited_history(self, ref: "SessionRef") -> list[Any]:
        """``ref``'s conversation as an inheriting run the user starts reads it.

        The root's messages in its newest checkpoint, its summary applied
        (``inherited_messages``), as the root's ``task`` hands them from its
        state; none for a conversation with no checkpoint yet.
        """
        values = (await self._root.aget_state(run_config(ref))).values
        return inherited_messages(values)

    async def _deliver_job_notice(
        self, notice: str, record: Any, result: Any, ref: "SessionRef"
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run the root agent on a finished job's notice (the base's ``resume_with_job_result``).

        The turn's input is one ``HumanMessage`` (``sessions.job_notification``),
        the notice as its text, marked so ``recent_messages`` leaves it out.
        It cannot be the call's ``ToolMessage``: the call already has its
        pending result, and Anthropic and OpenAI take one result per call.
        The root receives it whichever agent started the job, since a
        sub-agent's run ended when its ``task`` returned. Otherwise it is a
        turn like any other: the pending notices open it, and the job's
        output marks the conversation as having read outside content, in its
        checkpoint too.
        """
        # The job's output is outside content: its notice marks the conversation.
        self._outside_marks.seed(ref, await self._sessions.outside_content(ref))
        self._outside_marks.mark(ref, JOB_RESULT)
        for pending in self._take_notices():
            yield pending
        async with contextlib.aclosing(self._run_turn(
            ref,
            {
                "messages": [job_notification(record, notice=notice)],
                "outside_content": JOB_RESULT,
            },
        )) as events:
            async for event in events:
                yield event

    async def _deliver_notice(
        self, text: str, ref: "SessionRef"
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run the root agent on a notice to it (the base's ``tell_root``).

        The turn's input is one ``HumanMessage`` (``sessions.notice_message``)
        marked so ``recent_messages`` and the conversation's row leave it
        out. A turn like any other: the pending notices open it, and the mark
        this manager holds for the conversation (what a run the user started
        brought) is written into the checkpoint with it, first write wins.
        """
        self._outside_marks.seed(ref, await self._sessions.outside_content(ref))
        mark = self._outside_marks.get(ref)
        for pending in self._take_notices():
            yield pending
        async with contextlib.aclosing(self._run_turn(
            ref,
            {"messages": [notice_message(text)], **({"outside_content": mark} if mark else {})},
        )) as events:
            async for event in events:
                yield event

    async def _compact_turn(
        self, session: "SessionRef", focus: str | None
    ) -> CompactResult | None:
        """Summarize the root agent's conversation now (the base's ``compact``, ``/compact``).

        With the root's own summarization middleware, as Deep Agents
        summarizes when the conversation grows: the thread's messages with
        its summary so far applied (``_apply_event_to_messages``) are cut as
        ADK's ``/compact`` cuts (``compact_cutoff``: about the agent's
        ``keep`` tokens kept, the cut where a user's message starts); the
        middleware's helper writes the summary, ``focus`` in its prompt
        (``write_summary``), under a run callback, so its start
        (``CONTEXT_SUMMARIZING``) and usage are reported to the caller's live
        events; and the summary is stored as the model node stores its own
        (``aupdate_state(..., as_node="model")``), a ``_summarization_event``
        the next request applies, replacing the summary so far. It is marked
        with the newest answer it was written after, whose reported count
        then no longer triggers a summary (``_should_summarize``).
        ``CONTEXT_TRIMMED`` (``source="compact"``) follows. The checkpoint
        keeps every message.

        The base runs it as a turn, so a turn or a job's resume never runs
        on the thread meanwhile. Nothing is written until the summary is: a
        cancel leaves the thread as it was.

        Args:
            session: The conversation.
            focus: What the summary should keep above all, if anything.

        Returns:
            What the summary did; None with no conversation yet, or nothing
            before the kept tail.

        Raises:
            ValueError: The model wrote an empty summary.
            Exception: The summary call failed (the provider's error).
        """
        config = run_config(session)
        values = (await self._root.aget_state(config)).values
        middleware = self._root_summarization
        previous = values.get(SUMMARIZATION_EVENT_KEY)
        messages = values.get("messages") or []
        sent = middleware._apply_event_to_messages(messages, previous)
        root = self._root_agent_name
        model = self._resolved_models[root]
        cut = compact_cutoff(
            sent, keep_tokens=self._budgets[root].keep, kind=model.record.kind
        )
        if cut is None:
            return None
        kept = sent[cut:]
        callback = RunCallback(
            root=root,
            agent_models=self._resolved_models,
            llm_log=self._raw_llm_log(),
            # Called outside a reporting context there is no one to
            # report to: what it reports is queued where nothing reads.
            live=live_events.current() or LiveEvents(),
            budgets=self._budgets,
            ledgers=self._usage_ledgers,
        )
        summary = await write_summary(
            middleware, sent[:cut], focus=focus, callbacks=[callback]
        )
        if not summary:
            raise ValueError(f"{model.ref} wrote an empty summary")
        newest = newest_answer(sent)
        event = summary_event(
            middleware, previous, cut, summary,
            after=newest.id if newest is not None else None,
            messages=messages,
        )
        await self._root.aupdate_state(
            config, {SUMMARIZATION_EVENT_KEY: event}, as_node="model"
        )
        # The prompt prefix changes here anyway: the conversation's
        # next request takes a new memory index.
        self._forget_memory_index(session)
        await asyncio.shield(self._recording(session))
        result = CompactResult(
            messages_summarized=cut,
            tokens_before=middleware.token_counter(sent),
            tokens_after=middleware.token_counter([event["summary_message"], *kept]),
        )
        live_events.report(WorkflowEvent.context_trimmed(
            len(sent), len(kept) + 1, "compact", root,
            messages_summarized=result.messages_summarized,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
        ))
        logger.info(
            "conversation_compacted",
            messages=result.messages_summarized,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            focus=focus is not None,
        )
        return result

    def _take_notices(self) -> list[WorkflowEvent]:
        """The pending notices, through ``on_event``; they open one turn only."""
        notices, self._pending_notices = self._pending_notices, []
        hooked = (self._apply_event_hook(notice) for notice in notices)
        return [notice for notice in hooked if notice is not None]

    async def _run_turn(
        self, ref: "SessionRef", graph_input: dict[str, Any]
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Stream the root agent on ``ref``'s thread and yield its events.

        The caller is inside the turn (the base's ``_turn``). What
        the run callback reports from inside a node (a summary that starts,
        its usage, the summary it made) is merged into the stream as it
        happens, so it shows while the node still runs, before the node's
        own output. The turn's live events are its reporter too
        (``live_events.current()``), which is how the summarization
        middleware knows the turn a failed summary belongs to. The
        conversation's row in the session index is written as the turn
        starts, the input counted, and, however the turn ends, from what it
        left (``_recording``).
        """
        processor = DeepAgentsEventProcessor(
            root=self._root_agent_name,
            agent_models=self._resolved_models,
            session_id=ref.session_id,
            budgets=self._budgets,
            ledgers=self._usage_ledgers,
        )
        live = LiveEvents()
        callback = RunCallback(
            root=self._root_agent_name,
            agent_models=self._resolved_models,
            llm_log=self._raw_llm_log(),
            live=live,
            budgets=self._budgets,
            ledgers=self._usage_ledgers,
        )
        # Each item names its namespace; the processor reads only the root's
        # (a graph a tool runs under the inherited config streams in its own).
        stream = self._root.astream(
            graph_input,
            run_config(ref, callbacks=[callback]),
            stream_mode=["updates"],
            subgraphs=True,
        )
        count = 0
        # Listed and counted before anything runs: a process killed in the
        # turn runs no finally.
        started = self._recording(ref, graph_input["messages"])
        try:
            await asyncio.shield(started)
            # aclosing: a cancelled turn closes the merge, which closes the
            # stream, and its tasks, now, in the task that iterates it.
            with live_events.reporting(live):
                async with contextlib.aclosing(merged(stream, live)) as items:
                    async for item in items:
                        if isinstance(item, WorkflowEvent):
                            events = [item]
                        else:
                            namespace, mode, payload = item
                            events = processor.process(namespace, mode, payload)
                        for event in events:
                            event = self._apply_event_hook(event)
                            if event is not None:
                                count += 1
                                yield event
        except (asyncio.CancelledError, GeneratorExit):
            # Its open calls are answered before the next turn's first model
            # call, from what the record kept (InterruptedCallsMiddleware).
            self._open_calls.cancelled(ref)
            raise
        except Exception as error:
            # The ERROR is the report; the log is for diagnosis, its traceback
            # at debug, so nothing is said twice over the CLI's prompt.
            logger.info(
                "deepagents_turn_failed",
                error_type=type(error).__name__,
                error=truncate(str(error), _MAX_LOGGED_ERROR),
            )
            logger.debug("deepagents_turn_failed", exc_info=error)
            # The failed step's writes are gone: its calls that returned keep
            # their results for the next run, which answers the others.
            self._open_calls.failed(ref)
            event = self._apply_event_hook(processor.turn_failed(error))
            if event is not None:
                yield event
            return
        finally:
            await asyncio.shield(self._recording(ref, after=started))
        # Every call of the turn has its result in the checkpoint now.
        self._open_calls.clear(ref)
        logger.info("message_processed", event_count=count)

    async def _run_agent(self, run: AgentRun) -> RunAnswer:
        """Run one sub-agent run on its agent's graph (the base's hook, ``Delegations``'s).

        A run on a model choice (``run.model``) runs on its type's graph built
        on that model (``_variant_graphs``), its usage and context the
        choice's (``_run_models``). A run of the built-in ``general_purpose``
        type (``run.based_on``) runs on the graph built for that type
        (``_general_graphs``), on that type's model.

        In the run's own task (``agents/delegate.py``), apart from its
        caller's stream: the graph is streamed with the run's own config,
        its callbacks this run's ``RunCallback``, and LangGraph's ambient
        config cleared in this task (``var_child_runnable_config``), so none
        of its chunks reach the caller's ``astream`` and the caller's
        checkpoint namespace is not the run's. The run's events go to the
        turn's live events as they are made, through a processor made for
        the run. Its input is the shared state it starts from (the
        outside-content mark alone: its plan and tasks start empty) and its
        messages: an inheriting run's history, then the task message. A
        cancel of the run alone (its control's: a limit, its parent or the
        user) closes its stream, which stops its graph, and it returns what
        it has: the last text it wrote and the shared keys it changed, as of
        its last finished step; ``Delegations`` ends it cancelled.

        The run is on its instance's own thread of the manager's checkpointer
        (``instance_run_config``): a first run starts it with the task
        message, a follow-up (``run.follow_up``) continues it, its input the
        message and the caller's mark when set. A first run's task message is
        marked as the instance's brief (``summarization.brief_message``), which
        the instance's summaries keep. The calls an earlier run of
        the instance left open are answered on that thread before its first
        model call (``InterruptedCallsMiddleware``). The shared keys it
        changed are compared with the thread's state as the run started it
        (the stream's first ``values`` chunk), so the instance's earlier
        mark is no change.

        Returns:
            Its last non-empty answer, as Deep Agents' own ``task`` read one
            (Anthropic may end with an empty ``end_turn`` after its final
            tool call), that answer's outcome mark
            (``SubAgentOutcomeMiddleware``), the shared keys it changed and its own
            plan and tasks as its graph left them;
            or, for a model call that failed, the provider's reason and its
            error's type, with the last text it wrote, as of its last
            finished step. Its answers
            are the messages after its task message (``_own_messages``): an
            inheriting run's history is not its own.

        Raises:
            RunFailed: Its request no longer fits its model's window, still
                after its summary (``subagent_overflow_message``), with the
                last text it wrote.
            RunFailed: A first run whose instance's thread the store already
                keeps: a stored transcript is never another run's.
            RuntimeError: The run has no conversation (its thread needs one).
        """
        conversation = run.conversation
        if conversation is None:
            raise RuntimeError("A sub-agent run runs in a conversation's turn.")
        # A run on a model choice runs on its type's graph built on that model;
        # a general-purpose run on the graph built for the type it is made for.
        if run.based_on is not None:
            graph = self._general_graphs[run.based_on]
        elif run.model is None:
            graph = self._agent_graphs[run.ref.agent]
        else:
            graph = self._variant_graphs[(run.ref.agent, run.model)]
        models, budgets = self._run_models(run)
        task_id = f"agentic-cli-task-{uuid.uuid4().hex}"
        # A first run's message is its instance's brief, which its summaries keep.
        task = (
            HumanMessage(content=run.message, id=task_id) if run.follow_up
            else brief_message(run.message, id=task_id)
        )
        if not run.follow_up and self._checkpointer is not None:
            # A first run never continues a thread the store already keeps
            # (as ADK refuses to make an existing session): whatever numbered
            # it, a stored transcript is never another run's.
            kept = await self._checkpointer.aget_tuple(
                {"configurable": {
                    "thread_id": instance_thread_id(conversation, run.ref.instance),
                    "checkpoint_ns": "",
                }}
            )
            if kept is not None:
                raise RunFailed(
                    f"The thread of sub-agent {run.ref.instance} already exists.",
                    code="THREAD_EXISTS",
                )
        if run.follow_up:
            inputs: dict[str, Any] = {
                **{k: v for k, v in run.state.items() if k in SHARED_KEYS and v},
                "messages": [task],
            }
        else:
            inputs = {**run.state, "messages": [*(run.history or ()), task]}
        processor = DeepAgentsEventProcessor(
            root=self._root_agent_name,
            agent_models=models,
            session_id=conversation.session_id,
            budgets=budgets,
            ledgers=self._usage_ledgers,
            run=run.ref,
        )
        callback = RunCallback(
            root=self._root_agent_name,
            agent_models=models,
            llm_log=self._raw_llm_log(),
            # Run outside a turn there is no one to report to.
            live=live_events.current() or LiveEvents(),
            budgets=budgets,
            ledgers=self._usage_ledgers,
        )
        final: dict[str, Any] = {}
        start: dict[str, Any] | None = None
        detached = var_child_runnable_config.set(None)
        try:
            stream = graph.astream(
                inputs,
                instance_run_config(conversation, run.ref.instance, callbacks=[callback]),
                stream_mode=["updates", "values"],
            )
            async with contextlib.aclosing(stream) as items:
                async for mode, payload in items:
                    if mode == "values":
                        if start is None:
                            start = payload
                        final = payload
                        continue
                    for event in processor.process((), mode, payload):
                        live_events.report(event)
        except asyncio.CancelledError:
            # Its instance's next run answers the calls this one left open
            # with what stopped it, the user or its cause (as ADK seals
            # them). The run's own cancel has stopped it; any other (its
            # turn's) goes on.
            self._open_calls.cancelled(
                conversation, instance=run.ref.instance,
                cause=self._delegations.stop_cause(run.ref, conversation=conversation),
            )
            if not run.control.own_cancel():
                raise
        except Exception as error:  # noqa: BLE001 - the run fails; its parent reads why
            # The run fails with the text it wrote before it.
            wrote = _last_text(_answers_in(final, task.id))
            if is_context_overflow(error):
                model = models.get(run.ref.agent)
                raise RunFailed(
                    subagent_overflow_message(
                        run.ref.agent, str(model.ref) if model is not None else "the model"
                    ),
                    code=OVERFLOW_ERROR_CODE,
                    text=wrote,
                ) from error
            # A model call failed: the provider's error.
            logger.debug("subagent_failed", run=run.ref.run, exc_info=error)
            plan, tasks = own_plan_and_tasks(final)
            return RunAnswer(
                text=wrote, error=failure_reason(error), error_code=type(error).__name__,
                plan=plan, tasks=tasks,
            )
        finally:
            var_child_runnable_config.reset(detached)
        # Its work has ended (nothing below waits; as ADK's, which reads its
        # state back): no cancel of the run alone stops it now.
        run.control.finish()
        answers = _answers_in(final, task.id)
        plan, tasks = own_plan_and_tasks(final)
        return RunAnswer(
            text=_last_text(answers),
            outcome=answer_outcome(answers[-1]) if answers else None,
            state=_changed(final, start if start is not None else inputs),
            plan=plan,
            tasks=tasks,
        )

    def _recording(
        self,
        ref: "SessionRef",
        adding: list[Any] | None = None,
        *,
        after: "asyncio.Future[None] | None" = None,
    ) -> "asyncio.Future[None]":
        """A task writing ``ref``'s row in the session index.

        The row is ``CheckpointSessions.record``'s. Its caller awaits it
        shielded, so a cancel landing meanwhile (Ctrl+C pressed again) leaves
        the write to finish.

        Args:
            ref: The conversation.
            adding: The input of the turn starting, counted in the row.
            after: The write the turn started with, finished first: however
                the turn is stopped, the row written last is its end's.
        """
        sessions = self._sessions

        async def write() -> None:
            if after is not None:
                await asyncio.wait([after])
            if sessions is not None:
                await sessions.record(ref, adding or ())

        return asyncio.ensure_future(write())

    def _raw_llm_log(self) -> LLMLogWriter | None:
        """The raw LLM log when ``raw_llm_logging`` is on, read for each turn.

        The same writer serves every turn; one whose directory cannot be made
        logs nothing (its records still reach the stream).
        """
        if not self._settings.raw_llm_logging:
            return None
        if self._llm_log is None:
            self._llm_log = LLMLogWriter(self._settings.app_name)
        return self._llm_log

    async def generate_simple(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a short text with the default model, outside the agents.

        Used for internal operations such as summaries. The model is built for
        the call through the LangChain adapter, with thinking off where the
        model allows it and ``max_tokens`` as its limit (``_simple_model``).
        Called by a tool during a turn, it runs in that turn's run, so the raw
        LLM log records it under the agent whose tool called it. Its usage is
        reported as a side call's (``_report_side_call``), timed around its
        request.

        Returns:
            The answer's text, without its thinking.
        """
        await self._ensure_initialized()
        resolved = self._simple_model(max_tokens)
        model = build_chat_model(resolved, settings=self._settings, app_name=self.app_name)
        started = time.perf_counter()
        response = await model.ainvoke([HumanMessage(content=prompt)])
        if response.usage_metadata:
            self._report_side_call(usage_event(
                resolved, response.usage_metadata,
                latency_ms=(time.perf_counter() - started) * 1000,
            ))
        return str(response.text)

    # -------------------------------------------------------------------------
    # Sessions (the checkpointer saves every step, and each turn's end its
    # row; nothing to flush)
    # -------------------------------------------------------------------------

    async def session_exists(self, session_id: str, *, user_id: str | None = None) -> bool:
        """True if the conversation holds at least one message.

        Args:
            session_id: The conversation.
            user_id: Its owner (defaults to ``settings.default_user``).
        """
        if self._sessions is None:
            return False
        return await self._sessions.exists(self.session_ref(session_id, user_id))

    async def list_sessions(
        self, *, user_id: str | None = None, limit: int | None = None
    ) -> list[SessionInfo]:
        """This app's conversations of one user, newest first.

        Args:
            user_id: The owner (defaults to ``settings.default_user``).
            limit: Only the newest this many (None: all of them).
        """
        if self._sessions is None:
            return []
        ref = self.session_ref(user_id=user_id)
        return await self._sessions.list(ref.app_name, ref.user_id, limit=limit)

    async def session_info(
        self, session_id: str, *, user_id: str | None = None
    ) -> SessionInfo | None:
        """One conversation's row, or None when it holds no message.

        Args:
            session_id: The conversation.
            user_id: Its owner (defaults to ``settings.default_user``).
        """
        if self._sessions is None:
            return None
        return await self._sessions.info(self.session_ref(session_id, user_id))

    async def _stored_instances(self, conversation: "SessionRef") -> dict[str, int]:
        """The highest instance number of each type ``conversation``'s checkpointer keeps.

        The base's hook (``Delegations`` asks it once per conversation per
        turn): read from the conversation's stored records and, as their
        floor, its instance threads (``CheckpointSessions.stored_instances``),
        so a manager after a restart numbers past what another ran there.
        """
        if self._sessions is None:
            return {}
        return await self._sessions.stored_instances(conversation)

    async def _load_agent_records(self, conversation: "SessionRef") -> Any | None:
        """``conversation``'s records, from the store beside the checkpointer (the
        base's store hook: ``CheckpointSessions.agent_records``).

        Raises:
            RuntimeError: The store is not open.
        """
        return await self._open_sessions().agent_records(conversation)

    async def _save_agent_records(
        self, conversation: "SessionRef", data: Mapping[str, Any]
    ) -> None:
        """Keep ``data`` as ``conversation``'s records, in the store beside the
        checkpointer (the base's store hook: ``CheckpointSessions.save_agent_records``).

        Raises:
            RuntimeError: The store is not open.
        """
        await self._open_sessions().save_agent_records(conversation, data)

    def _open_sessions(self) -> CheckpointSessions:
        sessions = self._sessions
        if sessions is None:
            raise RuntimeError("The session store is not open.")
        return sessions

    async def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        """Delete a conversation, its sub-agent instances' threads, its sub-agent
        records and its row in the session index.

        The instances' threads go first, then the records, then the
        conversation (``CheckpointSessions.delete``); then this manager
        forgets it (``_deleting``).

        Args:
            session_id: The conversation.
            user_id: Its owner (defaults to ``settings.default_user``).

        Returns:
            False if there is no such conversation.
        """
        if self._sessions is None:
            return False
        ref = self.session_ref(session_id, user_id)
        async with self._deleting(ref):
            deleted = await self._sessions.delete(ref)
        return deleted

    async def recent_messages(
        self, session_id: str, limit: int = 20, *, user_id: str | None = None
    ) -> list[dict]:
        """The conversation's recent user and assistant texts (for fact extraction).

        Args:
            session_id: The conversation.
            limit: How many messages, the most recent last.
            user_id: Its owner (defaults to ``settings.default_user``).
        """
        if self._sessions is None:
            return []
        return await self._sessions.recent(self.session_ref(session_id, user_id), limit)


def _own_messages(messages: list[Any], task_id: str | None) -> list[Any]:
    """The messages a run wrote: those after its task message, found by its id.

    An inheriting run's messages start with its caller's conversation, whose
    replies are not its answers. The task message is found by id, not by
    position: ``InterruptedCallsMiddleware.before_agent`` may answer calls
    left open in that conversation, inserting messages before it (the list
    rewritten, each message keeping its id). Were it missing (no rewrite
    drops an id), none would be the run's: an empty answer, never its
    caller's.
    """
    for index, message in enumerate(messages):
        if task_id is not None and getattr(message, "id", None) == task_id:
            return messages[index + 1:]
    return []


def _answers_in(values: dict[str, Any], task_id: str | None) -> list[AIMessage]:
    """A run's own answers in its graph's ``values``: the AI messages after its task message."""
    own = _own_messages(values.get("messages") or [], task_id)
    return [message for message in own if isinstance(message, AIMessage)]


def _last_text(answers: list[AIMessage]) -> str:
    """The text of the last of ``answers`` that has any, as Deep Agents' ``task`` reads it."""
    for message in reversed(answers):
        text = message.text.rstrip() if message.text else ""
        if text:
            return text
    return ""


def _changed(final: dict[str, Any], start: dict[str, Any]) -> dict[str, Any]:
    """The shared keys (the outside-content mark) ``final`` holds that changed.

    What the run started from is ``start``: the thread's state as the run
    began it.

    Deep Agents' ``task`` rule (``deepagents.middleware.subagents._is_unchanged``,
    private, copied): a key returned exactly as it was received is no
    change; ``is True`` takes an ambiguous comparison (an array's) as one.
    """
    changed: dict[str, Any] = {}
    for key in SHARED_KEYS:
        if key not in final:
            continue
        value = final[key]
        if key in start and (value is start[key] or (value == start[key]) is True):
            continue
        changed[key] = value
    return changed
