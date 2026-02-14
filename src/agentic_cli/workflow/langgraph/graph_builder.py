"""LangGraph graph building and LLM provider abstraction.

Extracted from LangGraphWorkflowManager to isolate graph construction
and LLM factory logic from the runtime process loop.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

from agentic_cli.workflow.config import AgentConfig
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()


class LangGraphBuilder:
    """Builds a LangGraph StateGraph from AgentConfig declarations.

    Also handles LLM provider abstraction (needed to create agent nodes)
    and thinking configuration per-model.

    Args:
        settings: Application settings for provider detection and retry config.
    """

    # Thinking budget tokens for Anthropic models
    _THINKING_BUDGETS = {"low": 4096, "medium": 10000, "high": 32000}

    def __init__(self, settings: "BaseSettings") -> None:
        self._settings = settings
        # Side-channel for trim events: agent_node appends here, manager drains
        self._trim_events: list[dict[str, Any]] = []

    def build(self, agent_configs: list[AgentConfig], default_model: str):
        """Build the LangGraph workflow from agent configs.

        Creates a graph where each agent config becomes a node,
        with edges based on sub_agent relationships.

        Args:
            agent_configs: List of agent configurations.
            default_model: Default model name for agents without explicit model.

        Returns:
            An uncompiled StateGraph.
        """
        from langgraph.graph import StateGraph, END
        from agentic_cli.workflow.langgraph.state import AgentState

        # Build config map
        config_map = {config.name: config for config in agent_configs}

        # Create graph
        graph = StateGraph(AgentState)

        # Get retry policy for nodes
        retry_policy = self.get_retry_policy()

        # Create nodes for each agent with retry policy
        for config in agent_configs:
            node_fn = self._create_agent_node(config, default_model)
            graph.add_node(config.name, node_fn, retry=retry_policy)

        # Determine entry point (root agent)
        root_agent = None
        for config in agent_configs:
            if config.sub_agents:
                root_agent = config.name
                break

        if root_agent is None and agent_configs:
            root_agent = agent_configs[0].name

        if root_agent is None:
            raise RuntimeError("No agents configured")

        # Set entry point
        graph.set_entry_point(root_agent)

        # Add tool execution nodes for agents that have tools
        from langgraph.prebuilt import ToolNode

        agents_with_tools = {
            config.name for config in agent_configs if config.tools
        }
        tool_map = {}
        for config in agent_configs:
            if config.tools:
                tool_node_name = f"{config.name}_tools"
                tool_map[config.name] = tool_node_name
                tool_node = ToolNode(
                    config.tools, name=tool_node_name, handle_tool_errors=True,
                )
                graph.add_node(tool_node_name, tool_node)
                # After tool execution, loop back to the agent
                graph.add_edge(tool_node_name, config.name)

        # Add edges based on sub_agent relationships and tool routing
        for config in agent_configs:
            if config.sub_agents:
                # Coordinator with sub-agents: route to sub-agents or end
                destinations = {
                    sub_name: sub_name for sub_name in config.sub_agents
                } | {"__end__": END}
                if config.name in tool_map:
                    destinations[tool_map[config.name]] = tool_map[config.name]
                graph.add_conditional_edges(
                    config.name,
                    self._create_agent_router(config, tool_map.get(config.name)),
                    destinations,
                )
            elif config.name in agents_with_tools:
                # Agent with tools: route to tool node or to parent/end
                parent = None
                for other in agent_configs:
                    if config.name in other.sub_agents:
                        parent = other.name
                        break
                end_target = parent or END
                end_label = parent or "__end__"
                graph.add_conditional_edges(
                    config.name,
                    self._create_agent_router(config, tool_map[config.name]),
                    {
                        tool_map[config.name]: tool_map[config.name],
                        "__end__": end_target,
                    },
                )
            else:
                # Leaf agent without tools: return to coordinator or end
                parent = None
                for other in agent_configs:
                    if config.name in other.sub_agents:
                        parent = other.name
                        break

                if parent:
                    graph.add_edge(config.name, parent)
                else:
                    graph.add_edge(config.name, END)

        return graph

    def get_llm(self, model: str):
        """Get a LangChain LLM instance for the specified model.

        Explicitly instantiates the correct provider based on model name prefix.
        Uses GenAI (not VertexAI) for Google models.

        Args:
            model: Model identifier (e.g. "claude-sonnet-4", "gemini-2.5-pro").

        Returns:
            A LangChain BaseChatModel instance.
        """
        thinking = self.get_thinking_config(model)

        # OpenAI models
        if model.startswith("gpt-") or model.startswith("o1"):
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model=model)
            except ImportError:
                raise ImportError(
                    f"Model {model} requires langchain-openai. "
                    "Install with: pip install langchain-openai"
                )

        # Anthropic models
        if model.startswith("claude-"):
            try:
                from langchain_anthropic import ChatAnthropic
                kwargs = {"model": model}
                if thinking and thinking["provider"] == "anthropic":
                    kwargs["thinking"] = thinking["thinking"]
                return ChatAnthropic(**kwargs)
            except ImportError:
                raise ImportError(
                    f"Model {model} requires langchain-anthropic. "
                    "Install with: pip install langchain-anthropic"
                )

        # Google models - use GenAI, NOT VertexAI
        if model.startswith("gemini-"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                kwargs = {"model": model}
                if thinking and thinking["provider"] == "google":
                    kwargs["include_thoughts"] = thinking.get("include_thoughts", True)
                    kwargs["thinking_level"] = thinking.get("thinking_level")
                return ChatGoogleGenerativeAI(**kwargs)
            except ImportError:
                raise ImportError(
                    f"Model {model} requires langchain-google-genai. "
                    "Install with: pip install langchain-google-genai"
                )

        # Unknown model - try to infer from available API keys
        if self._settings.has_google_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(model=model)
            except ImportError:
                pass

        if self._settings.has_anthropic_key:
            try:
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(model=model)
            except ImportError:
                pass

        raise ValueError(
            f"Cannot determine provider for model: {model}. "
            "Ensure model name starts with a known prefix (gpt-, claude-, gemini-) "
            "or install the appropriate langchain integration."
        )

    def get_thinking_config(self, model: str) -> dict[str, Any] | None:
        """Get thinking configuration for the model.

        Returns provider-specific thinking configuration based on
        settings.thinking_effort and model support.

        Args:
            model: Model identifier.

        Returns:
            Provider-specific thinking config dict, or None.
        """
        effort = self._settings.thinking_effort
        if effort == "none" or not self._settings.supports_thinking_effort(model):
            return None

        if model.startswith("claude-"):
            return {
                "provider": "anthropic",
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self._THINKING_BUDGETS.get(effort, 10000),
                },
            }

        if model.startswith("gemini-"):
            # Gemini 3 Pro only supports "low" and "high", not "medium"
            is_gemini_3_pro = "gemini-3" in model and "pro" in model
            level = "high" if (effort == "medium" and is_gemini_3_pro) else effort
            return {
                "provider": "google",
                "include_thoughts": True,
                "thinking_level": level,
            }

        return None

    def get_retry_policy(self):
        """Build RetryPolicy from settings configuration.

        Returns:
            RetryPolicy configured with settings values for retry behavior.
        """
        from langgraph.types import RetryPolicy

        return RetryPolicy(
            max_attempts=self._settings.retry_max_attempts,
            initial_interval=self._settings.retry_initial_delay,
            backoff_factor=self._settings.retry_backoff_factor,
        )

    def _create_agent_node(self, config: AgentConfig, default_model: str) -> Callable:
        """Create a LangGraph node function for an agent.

        Args:
            config: Agent configuration.
            default_model: Fallback model name.

        Returns:
            Async function that processes state through the agent.
        """

        async def agent_node(state: dict) -> dict:
            """Process state through this agent."""
            from langchain_core.messages import SystemMessage

            # Get LLM for this agent (use config model or default)
            model_name = config.model or default_model
            llm = self.get_llm(model_name)

            # Bind tools if available
            if config.tools:
                llm = llm.bind_tools(config.tools)

            # Build messages: system prompt + conversation history
            # State messages are already LangChain objects (add_messages reducer
            # auto-converts dicts), so we can use them directly.
            messages = []

            system_prompt = config.get_prompt()
            if system_prompt:
                if self._settings.prompt_caching_enabled and model_name.startswith("claude-"):
                    messages.append(
                        SystemMessage(
                            content=[{
                                "type": "text",
                                "text": system_prompt,
                                "cache_control": {"type": "ephemeral"},
                            }]
                        )
                    )
                else:
                    messages.append(SystemMessage(content=system_prompt))

            conversation = state.get("messages", [])

            # Apply context window trimming if enabled
            if self._settings.context_window_enabled:
                from langchain_core.messages import trim_messages

                pre_trim_count = len(conversation)
                conversation = trim_messages(
                    conversation,
                    max_tokens=self._settings.context_window_target_tokens,
                    strategy="last",
                    token_counter="approximate",
                    start_on="human",
                    include_system=False,  # System message added separately above
                )
                post_trim_count = len(conversation)
                if post_trim_count < pre_trim_count:
                    logger.debug(
                        "context_window_trimmed",
                        agent=config.name,
                        messages_before=pre_trim_count,
                        messages_after=post_trim_count,
                        messages_removed=pre_trim_count - post_trim_count,
                    )
                    self._trim_events.append({
                        "messages_before": pre_trim_count,
                        "messages_after": post_trim_count,
                        "agent": config.name,
                    })

            messages.extend(conversation)

            # Invoke LLM
            try:
                response = await llm.ainvoke(messages)

                return {
                    "messages": [response],
                    "current_agent": config.name,
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

    def _create_agent_router(
        self, config: AgentConfig, tool_node_name: str | None
    ) -> Callable:
        """Create a routing function for an agent node.

        Routes to tool execution if pending tool calls exist, to sub-agents
        if the agent is a coordinator, or to __end__ otherwise.

        Args:
            config: Agent configuration.
            tool_node_name: Name of the tool execution node (if agent has tools).

        Returns:
            Function that determines next node based on state.
        """

        def router(state: dict) -> str:
            """Route based on tool calls on last message and sub-agent delegation."""
            messages = state.get("messages", [])
            if not messages:
                return "__end__"

            last_msg = messages[-1]
            tool_calls = getattr(last_msg, "tool_calls", [])

            if tool_calls:
                # Check if any tool call maps to a sub-agent
                for tool_call in tool_calls:
                    name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                    if config.sub_agents and name in config.sub_agents:
                        return name

                # Otherwise route to tool execution node
                if tool_node_name:
                    return tool_node_name

            # Check last message for delegation hints (sub-agents only)
            if config.sub_agents:
                content = getattr(last_msg, "content", "")
                if isinstance(content, str):
                    content_lower = content.lower()
                    for sub_agent in config.sub_agents:
                        if sub_agent.lower() in content_lower:
                            return sub_agent

            return "__end__"

        return router

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
