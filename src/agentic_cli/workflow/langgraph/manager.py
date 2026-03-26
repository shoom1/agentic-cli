"""LangGraph-based Workflow Manager for agentic CLI applications.

This module provides a BaseWorkflowManager implementation using LangGraph for
agent orchestration. It offers:
- Cyclical workflows (self-validation, iterative refinement)
- Native HITL support with interrupt mechanism
- Model-agnostic operation (OpenAI, Anthropic, Google)
- State checkpointing for long-running tasks

Requires the 'langgraph' optional dependency:
    pip install agentic-cli[langgraph]
"""

from __future__ import annotations

from typing import AsyncGenerator, Any, Callable, Literal, TYPE_CHECKING

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder
from agentic_cli.config import get_settings
from agentic_cli.logging import Loggers, bind_context

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()


class LangGraphWorkflowManager(BaseWorkflowManager):
    """LangGraph-based workflow manager for agentic applications.

    This manager uses LangGraph for agent orchestration, providing:
    - Explicit graph-based workflow control
    - Native cyclical workflows for iterative refinement
    - HITL support via interrupt mechanism
    - Model-agnostic operation
    - State checkpointing and time-travel debugging

    Example:
        configs = [
            AgentConfig(
                name="researcher",
                prompt="Research the topic thoroughly",
                tools=[search_web, search_papers],
            ),
            AgentConfig(
                name="analyzer",
                prompt="Analyze the research findings",
                tools=[calculate_metrics],
            ),
        ]

        manager = LangGraphWorkflowManager(
            agent_configs=configs,
            settings=my_settings,
            checkpointer="memory",  # or "postgres" for persistence
        )

        async for event in manager.process("Analyze market trends", user_id="user1"):
            print(event.content)
    """

    def __init__(
        self,
        agent_configs: list[AgentConfig],
        settings: "BaseSettings | None" = None,
        app_name: str | None = None,
        model: str | None = None,
        checkpointer: Literal["memory", "postgres", "sqlite"] | None = "memory",
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
    ) -> None:
        """Initialize the LangGraph workflow manager.

        Args:
            agent_configs: List of agent configurations.
            settings: Application settings (uses get_settings() if not provided).
            app_name: Application name for services.
            model: Model override (auto-detected from API keys if not provided).
            checkpointer: Checkpointer type for state persistence.
                         "memory" for in-memory, "postgres" for PostgreSQL,
                         "sqlite" for SQLite. None to disable checkpointing.
            on_event: Optional hook to transform/filter events before yielding.
        """
        super().__init__(
            agent_configs=agent_configs,
            settings=settings,
            app_name=app_name,
            model=model,
            on_event=on_event,
        )

        self._checkpointer_type = checkpointer

        # Graph builder (delegates graph construction + LLM factory)
        self._builder = LangGraphBuilder(self._settings)

        # LangGraph components (lazy-initialized)
        self._graph = None
        self._compiled_graph = None
        self._checkpointer = None
        self._store = None
        self._llm = None

        # Session tracking
        self.session_id = "default_session"

        logger.debug(
            "langgraph_workflow_manager_created",
            app_name=self.app_name,
            model_override=model,
            agent_count=len(agent_configs),
            checkpointer=checkpointer,
        )

    @property
    def backend_type(self) -> str:
        """Return 'langgraph'."""
        return "langgraph"

    def _get_state_tools(self) -> list:
        """Return LangGraph-native state tools using Command."""
        from agentic_cli.tools.langgraph.state_tools import (
            save_plan, get_plan, save_tasks, get_tasks,
        )
        return [save_plan, get_plan, save_tasks, get_tasks]

    async def _do_initialize(self) -> None:
        """LangGraph-specific initialization: checkpointer, graph, LLM."""
        logger.info("initializing_langgraph_services", app_name=self.app_name)

        # Create checkpointer and store
        from agentic_cli.workflow.langgraph.persistence import create_checkpointer, create_store

        # Reuse preserved checkpointer from reinitialize, or create new one
        if getattr(self, '_preserved_checkpointer', None) is not None:
            self._checkpointer = self._preserved_checkpointer
        else:
            self._checkpointer = create_checkpointer(self._checkpointer_type, self._settings)
        store_type = getattr(self._settings, "store_type", "memory")
        self._store = create_store(store_type, self._settings)

        # Build tool overrides (swap state tools for LangGraph-native versions)
        tool_overrides = {
            config.name: self._build_tools(config)
            for config in self._agent_configs
        }

        # Build and compile graph via builder
        graph = self._builder.build(
            self._agent_configs, self.model, tool_overrides=tool_overrides,
        )
        self._graph = graph

        # Compile with checkpointer and store
        compile_kwargs = {}
        if self._checkpointer:
            compile_kwargs["checkpointer"] = self._checkpointer
        if self._store:
            compile_kwargs["store"] = self._store

        if compile_kwargs:
            self._compiled_graph = graph.compile(**compile_kwargs)
        else:
            self._compiled_graph = graph.compile()

        # Initialize default LLM
        self._llm = self._builder.get_llm(self.model)

        logger.info(
            "langgraph_services_initialized",
            model=self.model,
            checkpointer=self._checkpointer_type,
            store=getattr(self._settings, "store_type", "memory"),
        )

    async def cleanup(self) -> None:
        """Clean up LangGraph resources."""
        logger.debug("cleaning_up_langgraph_workflow_manager")

        self._graph = None
        self._compiled_graph = None
        self._checkpointer = None
        self._store = None
        self._llm = None
        self._initialized = False

        # Clean up managers (sandbox, etc.)
        self._cleanup_managers()

        logger.info("langgraph_workflow_manager_cleaned_up")

    async def reinitialize(
        self,
        model: str | None = None,
        preserve_sessions: bool = True,
    ) -> None:
        """Reinitialize with new configuration."""
        logger.info(
            "reinitializing_langgraph_workflow_manager",
            new_model=model,
            preserve_sessions=preserve_sessions,
        )

        old_checkpointer = self._checkpointer if preserve_sessions else None

        await self.cleanup()

        # Update model
        self._reset_model(model)

        # Set preserved checkpointer before init so _do_initialize can reuse it
        self._preserved_checkpointer = old_checkpointer
        await self.initialize_services()
        self._preserved_checkpointer = None

        logger.info(
            "langgraph_workflow_manager_reinitialized",
            model=self.model,
            sessions_preserved=preserve_sessions,
        )

    # -------------------------------------------------------------------------
    # Main processing
    # -------------------------------------------------------------------------

    async def process(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Process user input through the LangGraph workflow.

        Args:
            message: User message.
            user_id: User identifier.
            session_id: Optional session identifier.

        Yields:
            WorkflowEvent objects representing workflow output.
        """
        if not self._initialized:
            await self.initialize_services()

        if not self._compiled_graph:
            raise RuntimeError("LangGraph workflow not initialized")

        current_session_id = session_id or self.session_id
        bind_context(session_id=current_session_id, user_id=user_id)

        logger.info("processing_message_langgraph", message_length=len(message))

        # Context setup
        with self._workflow_context():
            # Prepare initial state
            initial_state = {
                "messages": [{"role": "user", "content": message}],
                "current_agent": None,
                "session_id": current_session_id,
                "user_id": user_id,
                "metadata": {},
            }

            # Configuration for checkpointing
            config = {"configurable": {"thread_id": current_session_id}}

            # Process through graph
            full_response_parts: list[str] = []

            async for event in self._compiled_graph.astream_events(
                initial_state,
                config=config,
                version="v2",
            ):
                event_kind = event.get("event", "")
                event_name = event.get("name", "")
                event_data = event.get("data", {})

                # Convert LangGraph events to WorkflowEvents
                if event_kind == "on_chat_model_stream":
                    # Accumulate streaming tokens — yielded as complete
                    # response in on_chat_model_end to avoid fragmented output.
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        blocks = self._extract_content_blocks(chunk.content)
                        for block_type, text in blocks:
                            if text:
                                full_response_parts.append(text)

                elif event_kind == "on_chat_model_end":
                    # Yield the complete accumulated response
                    output = event_data.get("output")
                    if output and hasattr(output, "content") and output.content:
                        blocks = self._extract_content_blocks(output.content)
                        for block_type, text in blocks:
                            if text:
                                if block_type == "thinking":
                                    transformed = self._apply_event_hook(
                                        WorkflowEvent.thinking(text, current_session_id)
                                    )
                                    if transformed:
                                        yield transformed
                                else:
                                    transformed = self._apply_event_hook(
                                        WorkflowEvent.text(text, current_session_id)
                                    )
                                    if transformed:
                                        yield transformed

                    # Emit context trimming events (side-channel from agent_node)
                    for info in self._builder.drain_trim_events():
                        transformed = self._apply_event_hook(
                            WorkflowEvent.context_trimmed(
                                messages_before=info["messages_before"],
                                messages_after=info["messages_after"],
                                source="langgraph",
                                agent=info.get("agent"),
                            )
                        )
                        if transformed:
                            yield transformed

                    # Extract usage metadata
                    if output and hasattr(output, "usage_metadata") and output.usage_metadata:
                        usage = output.usage_metadata
                        cached_read = usage.get("cache_read_input_tokens") if isinstance(usage, dict) else getattr(usage, "cache_read_input_tokens", None)
                        cache_creation = usage.get("cache_creation_input_tokens") if isinstance(usage, dict) else getattr(usage, "cache_creation_input_tokens", None)
                        usage_event = WorkflowEvent.llm_usage(
                            model=self.model,
                            prompt_tokens=usage.get("input_tokens") if isinstance(usage, dict) else getattr(usage, "input_tokens", None),
                            completion_tokens=usage.get("output_tokens") if isinstance(usage, dict) else getattr(usage, "output_tokens", None),
                            total_tokens=usage.get("total_tokens") if isinstance(usage, dict) else getattr(usage, "total_tokens", None),
                            cached_tokens=cached_read,
                            cache_creation_tokens=cache_creation,
                        )
                        transformed = self._apply_event_hook(usage_event)
                        if transformed:
                            yield transformed

                elif event_kind == "on_tool_start":
                    # Tool invocation
                    transformed = self._apply_event_hook(
                        WorkflowEvent.tool_call(
                            tool_name=event_name,
                            tool_args=event_data.get("input", {}),
                        )
                    )
                    if transformed:
                        yield transformed

                elif event_kind == "on_tool_end":
                    # Tool result — ToolNode returns ToolMessage objects
                    output = event_data.get("output", "")
                    result_data: Any = None
                    success = True

                    if hasattr(output, "content"):
                        raw = output.content
                    else:
                        raw = output

                    # Preserve structured dict results for summary formatting
                    if isinstance(raw, dict):
                        result_data = raw
                        success = raw.get("success", True)
                    elif isinstance(raw, str):
                        import json as _json

                        try:
                            parsed = _json.loads(raw)
                            if isinstance(parsed, dict):
                                result_data = parsed
                                success = parsed.get("success", True)
                            else:
                                result_data = raw
                        except (ValueError, TypeError):
                            result_data = raw
                    else:
                        result_data = raw

                    transformed = self._apply_event_hook(
                        WorkflowEvent.tool_result(
                            tool_name=event_name,
                            result=result_data,
                            success=success,
                        )
                    )
                    if transformed:
                        yield transformed

                    # Emit task progress after tool results
                    progress_event = self._build_task_progress(current_session_id)
                    if progress_event:
                        transformed = self._apply_event_hook(progress_event)
                        if transformed:
                            yield transformed


            # Final progress check
            progress_event = self._build_task_progress(current_session_id)
            if progress_event:
                transformed = self._apply_event_hook(progress_event)
                if transformed:
                    yield transformed

            logger.info("message_processed_langgraph")

    def _build_task_progress(self, session_id: str) -> "WorkflowEvent | None":
        """Build task progress from graph state."""
        from agentic_cli.tools._core.tasks import task_progress_data

        if not self._compiled_graph:
            return None
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self._compiled_graph.get_state(config)
            tasks_data = state.values.get("tasks", []) if state and state.values else []
        except Exception:
            return None
        if not tasks_data:
            return None

        progress = task_progress_data(tasks_data)
        if progress is None:
            return None

        return WorkflowEvent.task_progress(
            display=progress["display"],
            progress=progress["progress"],
            current_task_id=progress.get("current_task_id"),
            current_task_description=progress.get("current_task_description"),
        )

    def _extract_content_blocks(
        self, content: Any
    ) -> list[tuple[str, str]]:
        """Extract content blocks with type information.

        Returns list of (block_type, text) tuples where block_type is
        'thinking' or 'text'.

        Args:
            content: Raw content from LLM response.

        Returns:
            List of (type, text) tuples.
        """
        blocks = []

        if isinstance(content, str):
            if content:
                blocks.append(("text", content))
            return blocks

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    block_type = item.get("type", "text")
                    text = item.get("text", "")

                    # Check for thinking blocks
                    if block_type == "thinking" or item.get("thinking"):
                        if text:
                            blocks.append(("thinking", text))
                    elif text:
                        blocks.append(("text", text))
                elif isinstance(item, str) and item:
                    blocks.append(("text", item))
                # Handle LangChain message objects
                elif hasattr(item, "type") and hasattr(item, "content"):
                    if item.type == "thinking":
                        blocks.append(("thinking", str(item.content)))
                    else:
                        blocks.append(("text", str(item.content)))

        return blocks

    async def generate_simple(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a simple text response using the current model."""
        if not self._initialized:
            await self.initialize_services()

        if not self._llm:
            self._llm = self._builder.get_llm(self.model)

        from langchain_core.messages import HumanMessage

        response = await self._llm.ainvoke(
            [HumanMessage(content=prompt)],
            max_tokens=max_tokens,
        )

        return response.content if hasattr(response, "content") else str(response)

    # -------------------------------------------------------------------------
    # Session save/resume hooks
    # -------------------------------------------------------------------------

    def _get_state_values(self, session_id: str) -> dict | None:
        """Get state values from LangGraph, or None on failure."""
        if not self._compiled_graph:
            return None
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self._compiled_graph.get_state(config)
        except Exception:
            return None
        if not state or not state.values:
            return None
        return state.values

    async def _extract_session_data(
        self, session_id: str
    ) -> tuple[list[dict], str | None]:
        """Extract normalized messages and current agent from LangGraph state."""
        values = self._get_state_values(session_id)
        if not values:
            return [], None

        current_agent = values.get("current_agent")
        raw_messages = values.get("messages", [])
        messages: list[dict] = []

        for msg in raw_messages:
            msg_type = getattr(msg, "type", "")

            if msg_type == "human":
                messages.append({"role": "user", "content": msg.content})

            elif msg_type == "ai":
                entry: dict = {"role": "assistant", "content": msg.content or ""}
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.get("id", tc.get("name", "")),
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                        }
                        for tc in tool_calls
                    ]
                messages.append(entry)

            elif msg_type == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(msg, "tool_call_id", ""),
                    "name": getattr(msg, "name", "unknown"),
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                })

        return messages, current_agent

    async def _inject_session_messages(
        self,
        session_id: str,
        messages: list[dict],
        current_agent: str | None = None,
    ) -> None:
        """Inject normalized messages into LangGraph state."""
        if not self._compiled_graph:
            raise RuntimeError("LangGraph workflow not initialized")

        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        lc_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif role == "assistant":
                kwargs: dict = {"content": msg.get("content", "")}
                if msg.get("tool_calls"):
                    kwargs["tool_calls"] = msg["tool_calls"]
                lc_messages.append(AIMessage(**kwargs))
            elif role == "tool":
                lc_messages.append(ToolMessage(
                    content=msg["content"],
                    tool_call_id=msg.get("tool_call_id", ""),
                    name=msg.get("name", "unknown"),
                ))

        config = {"configurable": {"thread_id": session_id}}
        update = {"messages": lc_messages}
        if current_agent is not None:
            update["current_agent"] = current_agent

        self._compiled_graph.update_state(config, update)


# Alias for convenience
LangGraphManager = LangGraphWorkflowManager
