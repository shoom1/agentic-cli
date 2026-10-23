"""The Deep Agents workflow manager (``orchestrator = "deepagents"``).

Every ``AgentConfig`` becomes one ``create_agent`` graph on its own chat model
(``agent_builder``), and the root agent runs each turn on the thread of the
turn's ``SessionRef`` (``sessions``). The stream's node outputs become
``WorkflowEvent``s (``event_processor``). Conversations live in a LangGraph
checkpointer opened from ``session_store``, which this manager owns: its
connection is closed on cleanup and by a failed initialization, and a model
switch keeps it. The agents' MCP sessions (``mcp_tools``) are its too: they
open at initialization, and every release closes them, so a model switch
reopens them. An agent's ``skills`` become the skill tools ADK offers
(``tools/skills/langchain_tools.py``), with ADK's instruction appended to its
prompt. With ``raw_llm_logging`` on, a turn's every model call is logged
(``LLM_REQUEST``/``LLM_RESPONSE``, in the stream and in
``~/.{app_name}/logs/llm_events.jsonl``). A long-running job that finishes
resumes its conversation with a turn of its own, opened by a notification to
the root agent (``resume_with_job_result``).

Lifecycle follows ADK's contract (``workflow/adk/manager.py``): initialization,
``reinitialize`` and ``cleanup`` hold the lifecycle lock and then the turn
lock, turns enter through ``_turn_admission``, and a model switch is
transactional, carrying the conversation's services and the checkpointer
across.

Installed by the ``deepagents`` extra: ``pip install 'agentic-cli[deepagents]'``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable

from langchain_core.messages import HumanMessage

from agentic_cli.config import BaseSettings
from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers, bind_context
from agentic_cli.tools.registry import ToolDefinition, identify_tool
from agentic_cli.tools.skills import SkillStore
from agentic_cli.tools.skills.langchain_tools import SKILL_TOOL_NAMES, make_skill_tools
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.checkpointers import deepagents_sqlite_path, open_checkpointer
from agentic_cli.workflow.config import (
    AgentConfig,
    AgentGraph,
    AgentGraphError,
    validate_agent_graph,
)
from agentic_cli.workflow.context import context_budget
from agentic_cli.workflow.deepagents.agent_builder import build_agent_tree
from agentic_cli.workflow.deepagents.event_processor import (
    DeepAgentsEventProcessor,
    RunCallback,
)
from agentic_cli.workflow.deepagents.mcp_tools import MCPSessions, without_clashes
from agentic_cli.workflow.deepagents.sessions import (
    CheckpointSessions,
    job_notification,
    run_config,
)
from agentic_cli.workflow.deepagents.tools import to_langchain_tools
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.langchain_models import build_chat_model
from agentic_cli.workflow.live_events import LiveEvents, merged
from agentic_cli.workflow.llm_log import LLMLogWriter
from agentic_cli.workflow.service_registry import JOB_MANAGER

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from agentic_cli.workflow.sessions import SessionRef

logger = Loggers.workflow()

#: Longest error text a failed turn's log line carries.
_MAX_LOGGED_ERROR = 500


def _with_instruction(prompt: str, instruction: str) -> str:
    """``prompt`` with ``instruction`` appended, joined as ADK appends one."""
    return f"{prompt}\n\n{instruction}" if prompt else instruction


class DeepAgentsWorkflowManager(BaseWorkflowManager):
    """Workflow manager on LangChain's ``create_agent`` and Deep Agents' middleware.

    A coordinator delegates through Deep Agents' ``task`` tool, giving a
    sub-agent only a brief; the framework's permission gate and tool-error
    middleware run in every agent's stack.

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
    ) -> None:
        """Initialize the workflow manager.

        Args:
            agent_configs: The agents; the root is the one nobody delegates to.
            settings: Application settings (``get_settings()`` if not provided).
            app_name: Application name for sessions (``settings.app_name`` if
                not provided).
            model: This manager's default model, a ``provider:model`` reference
                or a model id (see ``BaseWorkflowManager``).
            on_event: Optional hook to transform or drop events before they
                are yielded.
        """
        super().__init__(
            agent_configs=agent_configs,
            settings=settings,
            app_name=app_name,
            model=model,
            on_event=on_event,
        )
        self.session_id = "default_session"

        # The checkpointer and the context manager owning its connection (None
        # for the in-memory saver). Pinned while reinitialize(preserve_sessions=
        # True) carries them across, so neither its release nor a rollback
        # closes them.
        self._checkpointer: BaseCheckpointSaver | None = None
        self._checkpointer_cm: AbstractAsyncContextManager | None = None
        self._checkpointer_pinned = False
        self._sessions: CheckpointSessions | None = None

        self._root: CompiledStateGraph | None = None
        self._root_name: str | None = None

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
        return validate_agent_graph(self._agent_configs, backend=self.backend_type)

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
            store = self._settings.session_store
            self._checkpointer, self._checkpointer_cm = await open_checkpointer(
                store,
                sqlite_path=(
                    deepagents_sqlite_path(self._settings) if store == "sqlite" else None
                ),
                postgres_uri=self._settings.postgres_uri,
            )
            self._sessions = CheckpointSessions(self._checkpointer)

        graph = validate_agent_graph(self._agent_configs, backend=self.backend_type)
        # Each agent's context numbers, as its summarization uses them; a
        # budget too small to work in fails here (ContextBudgetError).
        self._budgets = {
            name: context_budget(model, self._settings)
            for name, model in self._resolved_models.items()
        }
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
            own = to_langchain_tools(self._build_tools(config, service_map))
            # Skill tools before MCP tools: an MCP tool named like one is skipped.
            if config.skills:
                skill_tools, instruction = self._skill_tools(config, own)
                own = [*own, *skill_tools]
                prompts[config.name] = _with_instruction(
                    prompts[config.name], instruction
                )
            if config.mcp_servers:
                own = await self._with_mcp_tools(config, own)
            tools[config.name] = own
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
        )
        self._root_name = graph.root_name

        logger.info(
            "services_initialized",
            model=self.model,
            root=self._root_name,
            session_store=self._settings.session_store,
            required_managers=list(self._required_managers),
        )

    def _skill_tools(
        self, config: AgentConfig, tools: list[BaseTool]
    ) -> tuple[list[BaseTool], str]:
        """``config``'s skills as skill tools, and the instruction its prompt gets.

        The skills resolve as on ADK: an absolute path to a skill folder, or a
        name under ``skills_dirs``.

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
            skills, provider=self._resolved_models[config.name].record.kind
        )
        logger.debug(
            "skill_tools_attached", agent=config.name, skills=[s.name for s in skills]
        )
        return skill_tools, instruction

    async def _with_mcp_tools(
        self, config: AgentConfig, tools: list[BaseTool]
    ) -> list[BaseTool]:
        """``tools`` and the tools of ``config``'s MCP servers, opened now.

        A server that cannot start, and a tool whose name is taken, become
        notices for the next turn; the agent runs without them.
        """
        loaded, unavailable = await self._mcp.start(config.name, config.mcp_servers)
        kept, skipped = without_clashes(config.name, tools, loaded)
        for message, code in [
            *((message, "mcp_unavailable") for message in unavailable),
            *((message, "mcp_tool_skipped") for message in skipped),
        ]:
            notice = WorkflowEvent.error(message, error_code=code, recoverable=True)
            notice.metadata["agent"] = config.name
            self._pending_notices.append(notice)
        return [*tools, *kept]

    def _identify_tool(self, obj: object) -> ToolDefinition | None:
        """What a tool object is: an open MCP session's tool, else a framework tool.

        Every agent's permission gate asks this first (``build_agent_tree``'s
        ``bound``).
        """
        definition = self._mcp.identify(obj)
        return definition if definition is not None else identify_tool(obj)

    async def cleanup(self) -> None:
        """Release everything this manager owns.

        Idempotent: references are detached before they are closed, so a
        repeat call finds nothing. Takes the lifecycle and turn locks, so it
        waits for a running turn.
        """
        async with self._lifecycle_lock:
            async with self._turn_lock:
                await self._release_resources()

    async def _release_resources(self, keep_checkpointer: bool = False) -> None:
        """Release owned resources; never raises.

        The MCP sessions are closed on every path, a model switch's included,
        and the notices about them dropped. The checkpointer's connection is
        closed, not merely dropped; ``reinitialize(preserve_sessions=True)``
        keeps the checkpointer instead and reuses it, and so does the rollback
        of a failed attempt inside it (the checkpointer is pinned then).

        Args:
            keep_checkpointer: Keep the checkpointer, its connection and the
                session store open.
        """
        logger.debug("cleaning_up_workflow_manager")

        # Detach first, so a concurrent or repeat cleanup cannot close twice.
        connection = None
        if not (keep_checkpointer or self._checkpointer_pinned):
            connection, self._checkpointer_cm = self._checkpointer_cm, None
            self._checkpointer = None
            self._sessions = None
        self._root = None
        self._root_name = None
        self._budgets = {}
        self._initialized = False
        self._pending_notices = []

        # Never raises; each session's owner leaves it even if this is cancelled.
        await self._mcp.aclose()
        await self._close_connection(connection)
        await self._close_catalog()
        self._cleanup_managers()

        logger.info("workflow_manager_cleaned_up")

    async def _close_connection(self, connection: "AbstractAsyncContextManager | None") -> None:
        """Exit the context manager owning the checkpointer's connection. Never raises."""
        if connection is None:
            return
        try:
            await connection.__aexit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail
            logger.warning("resource_close_failed", resource="checkpointer", error=str(exc))

    async def reinitialize(
        self,
        model: str | None = None,
        preserve_sessions: bool = True,
    ) -> None:
        """Rebuild the agents with a new configuration (a model switch).

        Transactional: on failure the manager is left *uninitialized*, with
        nothing half-built. A preserved checkpointer survives both outcomes,
        so a retry continues the same conversations; so do the services
        scoped to them (session grants, sandbox kernels, jobs, knowledge
        bases; see ``_carrying_services``).

        Args:
            model: The new default model; None re-resolves it from settings.
            preserve_sessions: Keep the checkpointer (the same saver and
                connection) and the services, and rebuild only the agents.
                False closes them and opens fresh ones.

        Raises:
            Exception: Whatever initialization raised, after the rollback.
        """
        logger.info(
            "reinitializing_workflow_manager",
            new_model=model,
            preserve_sessions=preserve_sessions,
        )

        async with self._lifecycle_lock:
            async with self._turn_lock:
                preserved = (
                    (self._checkpointer, self._checkpointer_cm, self._sessions)
                    if preserve_sessions
                    else None
                )
                with self._carrying_services(preserve_sessions):
                    await self._release_resources(keep_checkpointer=preserve_sessions)
                    self._reset_model(model)
                    self._checkpointer_pinned = preserve_sessions
                    try:
                        # _do_initialize reuses self._checkpointer when set, so
                        # nothing replaces a preserved one.
                        await self._initialize_locked()
                    except BaseException:
                        # The rollback left the pinned checkpointer alone;
                        # restore it anyway, so a retry can use it.
                        if preserved is not None and preserved[0] is not None:
                            (
                                self._checkpointer,
                                self._checkpointer_cm,
                                self._sessions,
                            ) = preserved
                        raise
                    finally:
                        self._checkpointer_pinned = False

        logger.info(
            "workflow_manager_reinitialized",
            model=self.model,
            sessions_preserved=preserve_sessions,
        )

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
        current_session_id = session_id or self.session_id
        bind_context(session_id=current_session_id, user_id=user_id)
        logger.info("processing_message", message_length=len(message))

        async with self._turn_admission():
            with self._workflow_context(session_id=current_session_id, user_id=user_id):
                for notice in self._take_notices():
                    yield notice
                ref = self.session_ref(current_session_id, user_id)
                async for event in self._run_turn(
                    ref, {"messages": [HumanMessage(content=message)]}
                ):
                    yield event

    async def can_resume(self, record) -> bool:
        """True iff the job's conversation is still there to resume into.

        It needs the record's session and user, and that conversation in this
        app, holding a message. The call id is not needed: the root agent is
        told in a message of its own, not as a second result for the call.
        With a durable ``session_store`` (sqlite, the default, or postgres) a
        job that finishes after a restart still resumes. After the session
        was deleted, or with ``memory`` once the process restarted, it does
        not, and the CLI says the job finished while its conversation was
        unavailable.
        """
        if not (record.session_id and record.user_id) or self._sessions is None:
            return False
        return await self._sessions.exists(
            self.session_ref(record.session_id, record.user_id)
        )

    async def resume_with_job_result(
        self, record, result: Any = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Tell the root agent a long-running job finished, in a turn of its own.

        The turn's input is one ``HumanMessage`` (``sessions.job_notification``):
        ``JOB_NOTICE_HEADER``, then the payload ADK sends, as JSON, marked so
        ``recent_messages`` leaves it out. It cannot be the call's
        ``ToolMessage``: the call already has its pending result, and
        Anthropic and OpenAI take one result per call. The root receives it
        whichever agent started the job, since a sub-agent's run ended when
        its ``task`` returned. Otherwise it is a turn like any other: it waits
        for the turn lock and runs under the record's session and user. A
        record without its session or user, or whose conversation is not
        there (deleted, or another user's or app's), runs nothing and starts
        no conversation.

        Args:
            record: The finished ``JobRecord``.
            result: The job's result; fetched from the JobManager if omitted.

        Yields:
            The turn's ``WorkflowEvent``s, each through ``on_event``.
        """
        session_id, user_id = record.session_id, record.user_id
        if not session_id or not user_id:
            logger.warning(
                "job_resume_missing_ids",
                job_id=getattr(record, "job_id", None),
                have_session=bool(session_id),
                have_user=bool(user_id),
            )
            return

        bind_context(session_id=session_id, user_id=user_id)

        async with self._turn_admission():
            with self._workflow_context(session_id=session_id, user_id=user_id):
                ref = self.session_ref(session_id, user_id)
                # Streaming into a thread that is not there would start one.
                if not await self._sessions.exists(ref):
                    logger.warning(
                        "job_resume_session_missing",
                        job_id=record.job_id,
                        session_id=session_id,
                    )
                    return

                if result is None:
                    jobs = self._services.get(JOB_MANAGER)
                    if jobs is not None:
                        result = jobs.result(record.job_id)

                logger.info(
                    "job_resume_started",
                    job_id=record.job_id,
                    call_id=record.call_id,
                    state=record.state.value,
                )
                for notice in self._take_notices():
                    yield notice
                async for event in self._run_turn(
                    ref, {"messages": [job_notification(record, result)]}
                ):
                    yield event

    def _take_notices(self) -> list[WorkflowEvent]:
        """The pending notices, through ``on_event``; they open one turn only."""
        notices, self._pending_notices = self._pending_notices, []
        hooked = (self._apply_event_hook(notice) for notice in notices)
        return [notice for notice in hooked if notice is not None]

    async def _run_turn(
        self, ref: "SessionRef", graph_input: dict[str, Any]
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Stream the root agent on ``ref``'s thread and yield its events.

        The caller holds the turn and is inside ``_workflow_context``. What
        the run callback reports from inside a node (a summary that starts,
        its usage, the summary it made) is merged into the stream as it
        happens, so it shows while the node still runs, before the node's
        own output.
        """
        processor = DeepAgentsEventProcessor(
            root=self._root_name,
            agent_models=self._resolved_models,
            session_id=ref.session_id,
            budgets=self._budgets,
        )
        live = LiveEvents()
        callback = RunCallback(
            root=self._root_name,
            agent_models=self._resolved_models,
            llm_log=self._raw_llm_log(),
            live=live,
            budgets=self._budgets,
        )
        stream = self._root.astream(
            graph_input,
            run_config(ref, callbacks=[callback]),
            stream_mode=["updates", "tasks"],
            subgraphs=True,
        )
        count = 0
        try:
            # aclosing: a cancelled turn closes the merge, which closes the
            # stream, and its tasks, now, in the task that iterates it.
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
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(
                "deepagents_turn_failed",
                error_type=type(error).__name__,
                error=truncate(str(error), _MAX_LOGGED_ERROR),
                exc_info=error,
            )
            event = self._apply_event_hook(processor.turn_failed(error))
            if event is not None:
                yield event
            return
        logger.info("message_processed", event_count=count)

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
        LLM log records it under the agent whose tool called it.

        Returns:
            The answer's text, without its thinking.
        """
        await self._ensure_initialized()
        model = build_chat_model(
            self._simple_model(max_tokens), settings=self._settings, app_name=self.app_name
        )
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return str(response.text)

    # -------------------------------------------------------------------------
    # Sessions (the checkpointer saves every step; nothing to flush)
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

    async def list_sessions(self, *, user_id: str | None = None) -> list[dict]:
        """This app's conversations of one user, newest first.

        Args:
            user_id: The owner (defaults to ``settings.default_user``).

        Returns:
            ``{"session_id", "last_update", "message_count"}`` per
            conversation, ``last_update`` in epoch seconds.
        """
        if self._sessions is None:
            return []
        ref = self.session_ref(user_id=user_id)
        return await self._sessions.list(ref.app_name, ref.user_id)

    async def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        """Delete a conversation, its sub-agents' checkpoints included.

        Args:
            session_id: The conversation.
            user_id: Its owner (defaults to ``settings.default_user``).

        Returns:
            False if there is no such conversation.
        """
        if self._sessions is None:
            return False
        return await self._sessions.delete(self.session_ref(session_id, user_id))

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
