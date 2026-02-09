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

        # Add edges based on sub_agent relationships
        for config in agent_configs:
            if config.sub_agents:
                # Add router for this coordinator
                graph.add_conditional_edges(
                    config.name,
                    self._create_router(config),
                    {sub_name: sub_name for sub_name in config.sub_agents}
                    | {"__end__": END},
                )
            else:
                # Leaf agents return to coordinator or end
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
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

            # Get LLM for this agent (use config model or default)
            model_name = config.model or default_model
            llm = self.get_llm(model_name)

            # Bind tools if available
            if config.tools:
                llm = llm.bind_tools(config.tools)

            # Build messages
            messages = []

            # Add system prompt
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

    def _create_router(self, config: AgentConfig) -> Callable:
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
