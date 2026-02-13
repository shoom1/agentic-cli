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
from agentic_cli.config import (
    get_settings,
    validate_settings,
)
from agentic_cli.logging import Loggers, bind_context

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()


class LangGraphSummarizer:
    """LLM Summarizer implementation using LangGraph/LangChain.

    Uses the configured LLM (Anthropic, OpenAI, or Google) directly
    to summarize web content for the webfetch tool.
    """

    def __init__(self, manager: "LangGraphWorkflowManager") -> None:
        """Initialize the summarizer.

        Args:
            manager: The workflow manager to use for LLM calls.
        """
        self._manager = manager

    async def summarize(self, content: str, prompt: str) -> str:
        """Summarize content using the configured LLM.

        Args:
            content: The content to summarize (markdown).
            prompt: The full summarization prompt.

        Returns:
            Summarized text response.
        """
        # Use the manager's generate_simple method
        return await self._manager.generate_simple(prompt, max_tokens=2000)


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
        )

        self._checkpointer_type = checkpointer
        self._on_event = on_event

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
    def graph(self):
        """Get the compiled LangGraph graph."""
        return self._compiled_graph

    def _create_summarizer(self) -> LangGraphSummarizer:
        """Create an LLM summarizer for webfetch.

        Returns:
            LangGraphSummarizer instance that uses the configured LLM.
        """
        return LangGraphSummarizer(self)

    async def initialize_services(self, validate: bool = True) -> None:
        """Initialize LangGraph services asynchronously."""
        if self._initialized:
            logger.debug("services_already_initialized")
            return

        logger.info("initializing_langgraph_services", app_name=self.app_name)

        # Validate settings
        if validate:
            validate_settings(self._settings)

        # Export API keys
        self._settings.export_api_keys_to_env()

        # Create checkpointer and store
        from agentic_cli.workflow.langgraph.persistence import create_checkpointer, create_store

        self._checkpointer = create_checkpointer(self._checkpointer_type, self._settings)
        store_type = getattr(self._settings, "store_type", "memory")
        self._store = create_store(store_type, self._settings)

        # Build and compile graph via builder
        graph = self._builder.build(self._agent_configs, self.model)
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

        # Initialize feature managers based on tool requirements
        self._ensure_managers_initialized()

        self._initialized = True
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

        # Cancel pending input requests
        for request_id, (request, future) in self._pending_input.items():
            if not future.done():
                future.cancel()
        self._pending_input.clear()

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

        # Store checkpointer if preserving sessions
        old_checkpointer = self._checkpointer if preserve_sessions else None

        await self.cleanup()

        # Update model
        if model is not None:
            self._model = model
            self._model_resolved = True
        else:
            self._model = None
            self._model_resolved = False

        await self.initialize_services()

        # Restore checkpointer if preserving
        if old_checkpointer and preserve_sessions:
            self._checkpointer = old_checkpointer

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
                                    yield self._maybe_transform(
                                        WorkflowEvent.thinking(text, current_session_id)
                                    )
                                else:
                                    yield self._maybe_transform(
                                        WorkflowEvent.text(text, current_session_id)
                                    )

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
                        yield self._maybe_transform(usage_event)

                elif event_kind == "on_tool_start":
                    # Tool invocation
                    yield self._maybe_transform(
                        WorkflowEvent.tool_call(
                            tool_name=event_name,
                            tool_args=event_data.get("input", {}),
                        )
                    )

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

                    yield self._maybe_transform(
                        WorkflowEvent.tool_result(
                            tool_name=event_name,
                            result=result_data,
                            success=success,
                        )
                    )

                    # Emit task progress after tool results
                    progress_event = self._emit_task_progress_event()
                    if progress_event:
                        yield self._maybe_transform(progress_event)


            logger.info("message_processed_langgraph")

    def _maybe_transform(self, event: WorkflowEvent) -> WorkflowEvent:
        """Apply event transformation hook if configured."""
        if self._on_event:
            transformed = self._on_event(event)
            return transformed if transformed is not None else event
        return event

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


# Alias for convenience
LangGraphManager = LangGraphWorkflowManager
