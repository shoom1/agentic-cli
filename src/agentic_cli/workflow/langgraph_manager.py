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

import asyncio
from typing import AsyncGenerator, Any, Callable, Literal, TYPE_CHECKING

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest, EventType
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.config import (
    get_settings,
    set_context_settings,
    set_context_workflow,
    validate_settings,
)
from agentic_cli.logging import Loggers, bind_context

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()


# Lazy imports for LangGraph dependencies
def _import_langgraph():
    """Lazily import LangGraph dependencies."""
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import RetryPolicy

        return StateGraph, END, MemorySaver, RetryPolicy
    except ImportError as e:
        raise ImportError(
            "LangGraph dependencies not installed. "
            "Install with: pip install agentic-cli[langgraph]"
        ) from e


def _import_langchain_models():
    """Lazily import LangChain model integrations."""
    models = {}

    try:
        from langchain_openai import ChatOpenAI

        models["openai"] = ChatOpenAI
    except ImportError:
        pass

    try:
        from langchain_anthropic import ChatAnthropic

        models["anthropic"] = ChatAnthropic
    except ImportError:
        pass

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        models["google"] = ChatGoogleGenerativeAI
    except ImportError:
        pass

    if not models:
        raise ImportError(
            "No LangChain model integrations found. "
            "Install at least one of: langchain-openai, langchain-anthropic, langchain-google-genai"
        )

    return models


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
        checkpointer: Literal["memory", "postgres"] | None = "memory",
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
    ) -> None:
        """Initialize the LangGraph workflow manager.

        Args:
            agent_configs: List of agent configurations.
            settings: Application settings (uses get_settings() if not provided).
            app_name: Application name for services.
            model: Model override (auto-detected from API keys if not provided).
            checkpointer: Checkpointer type for state persistence.
                         "memory" for in-memory, "postgres" for PostgreSQL.
                         None to disable checkpointing.
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

        # Model resolved lazily
        self._model: str | None = model
        self._model_resolved: bool = model is not None

        # LangGraph components (lazy-initialized)
        self._graph = None
        self._compiled_graph = None
        self._checkpointer = None
        self._llm = None

        # User input handling
        self._pending_input: dict[str, tuple[UserInputRequest, asyncio.Future[str]]] = (
            {}
        )

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
    def model(self) -> str:
        """Get the model name, resolving from settings if needed."""
        if not self._model_resolved:
            self._model = self._settings.get_model()
            self._model_resolved = True
            logger.info("model_resolved", model=self._model)
        return self._model  # type: ignore[return-value]

    @property
    def graph(self):
        """Get the compiled LangGraph graph."""
        return self._compiled_graph

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
        """Request user input from the CLI."""
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
        """Provide user input for a pending request."""
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

    def _get_thinking_config(self, model: str) -> dict[str, Any] | None:
        """Get thinking configuration for the model.

        Returns provider-specific thinking configuration based on
        settings.thinking_effort and model support.

        Args:
            model: Model name to check.

        Returns:
            Provider-specific thinking config dict, or None if thinking is disabled.
        """
        thinking_effort = self._settings.thinking_effort

        if thinking_effort == "none":
            return None

        if not self._settings.supports_thinking_effort(model):
            logger.debug(
                "thinking_not_supported",
                model=model,
                effort=thinking_effort,
            )
            return None

        # Map effort levels to budget tokens for Anthropic
        # These are reasonable defaults based on Claude's capabilities
        anthropic_budgets = {
            "low": 4096,
            "medium": 10000,
            "high": 32000,
        }

        # Map effort levels for Google Gemini (uses thinking_level parameter)
        # Note: Gemini 3 Pro only supports "low" and "high", not "medium"
        google_levels = {
            "low": "low",
            "medium": "medium",
            "high": "high",
        }

        logger.debug(
            "thinking_config_created",
            model=model,
            effort=thinking_effort,
        )

        # Return provider-specific config
        if model.startswith("claude-"):
            return {
                "provider": "anthropic",
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": anthropic_budgets.get(thinking_effort, 10000),
                },
            }
        elif model.startswith("gemini-"):
            # Gemini 3 Pro only supports "low" and "high" thinking levels
            # Gemini 3 Flash supports all levels including "medium"
            is_gemini_3_pro = "gemini-3" in model and "pro" in model
            thinking_level = google_levels.get(thinking_effort, "high")

            if thinking_level == "medium" and is_gemini_3_pro:
                thinking_level = "high"
                logger.debug(
                    "thinking_level_fallback",
                    model=model,
                    requested="medium",
                    actual="high",
                    reason="Gemini 3 Pro only supports low and high",
                )

            return {
                "provider": "google",
                "include_thoughts": True,
                "thinking_level": thinking_level,
            }

        return None

    def _get_llm_for_model(self, model: str):
        """Get a LangChain LLM instance for the specified model.

        Args:
            model: Model name to use.

        Returns:
            LangChain chat model instance.

        Raises:
            ValueError: If model provider is not supported.
        """
        models = _import_langchain_models()
        thinking_config = self._get_thinking_config(model)

        # Determine provider from model name
        if model.startswith("gpt-") or model.startswith("o1"):
            if "openai" not in models:
                raise ValueError(
                    f"Model {model} requires langchain-openai. "
                    "Install with: pip install langchain-openai"
                )
            return models["openai"](model=model)

        elif model.startswith("claude-"):
            if "anthropic" not in models:
                raise ValueError(
                    f"Model {model} requires langchain-anthropic. "
                    "Install with: pip install langchain-anthropic"
                )
            # Apply Anthropic thinking config if available
            if thinking_config and thinking_config.get("provider") == "anthropic":
                return models["anthropic"](
                    model=model,
                    thinking=thinking_config["thinking"],
                )
            return models["anthropic"](model=model)

        elif model.startswith("gemini-"):
            if "google" not in models:
                raise ValueError(
                    f"Model {model} requires langchain-google-genai. "
                    "Install with: pip install langchain-google-genai"
                )
            # Apply Google thinking config if available
            if thinking_config and thinking_config.get("provider") == "google":
                return models["google"](
                    model=model,
                    include_thoughts=thinking_config.get("include_thoughts", True),
                    thinking_level=thinking_config.get("thinking_level"),
                )
            return models["google"](model=model)

        else:
            # Try to infer from available keys
            if self._settings.has_google_key and "google" in models:
                return models["google"](model=model)
            elif self._settings.has_anthropic_key and "anthropic" in models:
                return models["anthropic"](model=model)
            elif "openai" in models:
                return models["openai"](model=model)

            raise ValueError(
                f"Cannot determine provider for model: {model}. "
                "Ensure model name starts with provider prefix "
                "(gpt-, claude-, gemini-) or install appropriate langchain integration."
            )

    def _create_checkpointer(self):
        """Create the appropriate checkpointer based on configuration."""
        StateGraph, END, MemorySaver, RetryPolicy = _import_langgraph()

        if self._checkpointer_type == "memory":
            return MemorySaver()
        elif self._checkpointer_type == "postgres":
            try:
                from langgraph.checkpoint.postgres import PostgresSaver

                # PostgreSQL connection would be configured via settings
                # For now, raise a helpful error
                raise NotImplementedError(
                    "PostgreSQL checkpointer requires connection configuration. "
                    "Set up POSTGRES_URI in settings or use 'memory' checkpointer."
                )
            except ImportError:
                raise ImportError(
                    "PostgreSQL checkpointer requires langgraph-checkpoint-postgres. "
                    "Install with: pip install langgraph-checkpoint-postgres"
                )
        return None

    def _get_retry_policy(self):
        """Build RetryPolicy from settings configuration.

        Returns:
            RetryPolicy configured with settings values for retry behavior.
        """
        StateGraph, END, MemorySaver, RetryPolicy = _import_langgraph()

        return RetryPolicy(
            max_attempts=self._settings.retry_max_attempts,
            initial_interval=self._settings.retry_initial_delay,
            backoff_factor=self._settings.retry_backoff_factor,
        )

    def _build_graph(self):
        """Build the LangGraph workflow from agent configs.

        Creates a graph where each agent config becomes a node,
        with edges based on sub_agent relationships.
        """
        StateGraph, END, MemorySaver, RetryPolicy = _import_langgraph()
        from agentic_cli.workflow.langgraph_state import AgentState

        # Build config map
        config_map = {config.name: config for config in self._agent_configs}

        # Create graph
        graph = StateGraph(AgentState)

        # Get retry policy for nodes
        retry_policy = self._get_retry_policy()

        # Create nodes for each agent with retry policy
        for config in self._agent_configs:
            node_fn = self._create_agent_node(config)
            graph.add_node(config.name, node_fn, retry=retry_policy)

        # Determine entry point (root agent)
        root_agent = None
        for config in self._agent_configs:
            if config.sub_agents:
                root_agent = config.name
                break

        if root_agent is None and self._agent_configs:
            root_agent = self._agent_configs[0].name

        if root_agent is None:
            raise RuntimeError("No agents configured")

        # Set entry point
        graph.set_entry_point(root_agent)

        # Add edges based on sub_agent relationships
        for config in self._agent_configs:
            if config.sub_agents:
                # Coordinator can route to any sub-agent
                for sub_name in config.sub_agents:
                    if sub_name in config_map:
                        # Add conditional edge for routing
                        pass  # Handled by router function

                # Add router for this coordinator
                graph.add_conditional_edges(
                    config.name,
                    self._create_router(config),
                    {sub_name: sub_name for sub_name in config.sub_agents}
                    | {"__end__": END},
                )
            else:
                # Leaf agents return to coordinator or end
                # Find parent coordinator
                parent = None
                for other in self._agent_configs:
                    if config.name in other.sub_agents:
                        parent = other.name
                        break

                if parent:
                    graph.add_edge(config.name, parent)
                else:
                    graph.add_edge(config.name, END)

        self._graph = graph
        return graph

    def _create_agent_node(self, config: AgentConfig):
        """Create a LangGraph node function for an agent.

        Args:
            config: Agent configuration.

        Returns:
            Async function that processes state through the agent.
        """

        async def agent_node(state: dict) -> dict:
            """Process state through this agent."""
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

            # Get LLM for this agent (use config model or default)
            model_name = config.model or self.model
            llm = self._get_llm_for_model(model_name)

            # Bind tools if available
            if config.tools:
                llm = llm.bind_tools(config.tools)

            # Build messages
            messages = []

            # Add system prompt
            system_prompt = config.get_prompt()
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            # Add conversation history
            for msg in state.get("messages", []):
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg.get("role") == "system":
                    messages.append(SystemMessage(content=msg["content"]))

            # Invoke LLM
            try:
                response = await llm.ainvoke(messages)

                # Extract and normalize content
                raw_content = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )
                content = self._normalize_content(raw_content)

                # Handle tool calls if present
                tool_calls = []
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tc in response.tool_calls:
                        tool_calls.append(
                            {
                                "id": tc.get("id", ""),
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                            }
                        )

                return {
                    "messages": [{"role": "assistant", "content": content}],
                    "current_agent": config.name,
                    "pending_tool_calls": tool_calls,
                }

            except Exception as e:
                logger.error(
                    "agent_node_error",
                    agent=config.name,
                    error=str(e),
                )
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": f"Error in {config.name}: {str(e)}",
                        }
                    ],
                    "current_agent": config.name,
                }

        return agent_node

    def _create_router(self, config: AgentConfig):
        """Create a routing function for a coordinator agent.

        Args:
            config: Coordinator agent configuration.

        Returns:
            Function that determines next node based on state.
        """

        def router(state: dict) -> str:
            """Route to next agent or end."""
            # Check for pending tool calls that map to sub-agents
            pending_tools = state.get("pending_tool_calls", [])

            for tool_call in pending_tools:
                tool_name = tool_call.get("name", "")
                # Check if tool name matches a sub-agent
                if tool_name in config.sub_agents:
                    return tool_name

            # Check last message for delegation hints
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = last_msg.get("content", "").lower()

                for sub_agent in config.sub_agents:
                    if sub_agent.lower() in content:
                        return sub_agent

            # Default: end the workflow
            return "__end__"

        return router

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

        # Create checkpointer
        self._checkpointer = self._create_checkpointer()

        # Build and compile graph
        graph = self._build_graph()

        # Compile with checkpointer
        if self._checkpointer:
            self._compiled_graph = graph.compile(checkpointer=self._checkpointer)
        else:
            self._compiled_graph = graph.compile()

        # Initialize default LLM
        self._llm = self._get_llm_for_model(self.model)

        self._initialized = True
        logger.info(
            "langgraph_services_initialized",
            model=self.model,
            checkpointer=self._checkpointer_type,
        )

    async def cleanup(self) -> None:
        """Clean up LangGraph resources."""
        logger.debug("cleaning_up_langgraph_workflow_manager")

        self._graph = None
        self._compiled_graph = None
        self._checkpointer = None
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

        # Set up context
        set_context_settings(self._settings)
        set_context_workflow(self)

        try:
            # Prepare initial state
            initial_state = {
                "messages": [{"role": "user", "content": message}],
                "current_agent": None,
                "pending_tool_calls": [],
                "tool_results": [],
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
                if event_kind == "on_chat_model_start":
                    # Agent starting
                    yield self._maybe_transform(
                        WorkflowEvent(
                            type=EventType.TOOL_CALL,
                            content=f"Agent: {event_name}",
                            metadata={"agent": event_name},
                        )
                    )

                elif event_kind == "on_chat_model_stream":
                    # Streaming response
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # Extract content blocks with type info
                        blocks = self._extract_content_blocks(chunk.content)
                        for block_type, text in blocks:
                            if text:
                                full_response_parts.append(text)
                                if block_type == "thinking":
                                    yield self._maybe_transform(
                                        WorkflowEvent.thinking(text, current_session_id)
                                    )
                                else:
                                    yield self._maybe_transform(
                                        WorkflowEvent.text(text, current_session_id)
                                    )

                elif event_kind == "on_chat_model_end":
                    # Agent finished
                    output = event_data.get("output")
                    if output and hasattr(output, "content") and output.content:
                        blocks = self._extract_content_blocks(output.content)
                        for block_type, text in blocks:
                            if text and text not in "".join(full_response_parts):
                                full_response_parts.append(text)
                                if block_type == "thinking":
                                    yield self._maybe_transform(
                                        WorkflowEvent.thinking(text, current_session_id)
                                    )
                                else:
                                    yield self._maybe_transform(
                                        WorkflowEvent.text(text, current_session_id)
                                    )

                elif event_kind == "on_tool_start":
                    # Tool invocation
                    yield self._maybe_transform(
                        WorkflowEvent.tool_call(
                            tool_name=event_name,
                            tool_args=event_data.get("input", {}),
                        )
                    )

                elif event_kind == "on_tool_end":
                    # Tool result
                    output = event_data.get("output", "")
                    yield self._maybe_transform(
                        WorkflowEvent.tool_result(
                            tool_name=event_name,
                            result=str(output),
                            success=True,
                        )
                    )

            logger.info("message_processed_langgraph")

        finally:
            set_context_settings(None)
            set_context_workflow(None)

    def _maybe_transform(self, event: WorkflowEvent) -> WorkflowEvent:
        """Apply event transformation hook if configured."""
        if self._on_event:
            transformed = self._on_event(event)
            return transformed if transformed is not None else event
        return event

    def _normalize_content(self, content: Any) -> str:
        """Normalize LLM response content to a string.

        Handles various content formats from different providers:
        - String: returned as-is
        - List of dicts with 'text' key: extracts and joins text
        - List of strings: joins them
        - Other: converts to string

        Args:
            content: Raw content from LLM response.

        Returns:
            Normalized string content.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    # Handle {'type': 'text', 'text': '...'} format
                    if "text" in item:
                        parts.append(item["text"])
                    elif "content" in item:
                        parts.append(str(item["content"]))
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            return "".join(parts)

        return str(content) if content else ""

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
            self._llm = self._get_llm_for_model(self.model)

        from langchain_core.messages import HumanMessage

        response = await self._llm.ainvoke(
            [HumanMessage(content=prompt)],
            max_tokens=max_tokens,
        )

        return response.content if hasattr(response, "content") else str(response)


# Alias for convenience
LangGraphManager = LangGraphWorkflowManager
