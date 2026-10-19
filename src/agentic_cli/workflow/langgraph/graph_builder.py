"""LangGraph graph building.

Extracted from LangGraphWorkflowManager to isolate graph construction from the
runtime process loop. Each agent node runs on a LangChain chat model built
once, when the graph is built, from the agent's resolved model
(``workflow/langchain_models.py``).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Mapping, TYPE_CHECKING

from agentic_cli.models import CacheMechanism, ResolvedModel
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()


class LangGraphBuilder:
    """Builds a LangGraph StateGraph from AgentConfig declarations.

    Args:
        settings: Application settings: the retry, timeout, caching and
            context-window settings.
        app_name: The application's name, which some providers are told
            (default: ``settings.app_name``).
    """

    def __init__(self, settings: "BaseSettings", app_name: str | None = None) -> None:
        self._settings = settings
        self._app_name = app_name or settings.app_name
        # Side-channel for trim events: agent_node appends here, manager drains
        self._trim_events: deque[dict[str, Any]] = deque()
        #: Each agent's chat model, by agent name, from the last ``build``.
        self.chat_models: dict[str, BaseChatModel] = {}

    def drain_trim_events(self) -> list[dict[str, Any]]:
        """Drain and return all pending context-trim events.

        Returns:
            List of trim event info dicts (empties the internal deque).
        """
        events = list(self._trim_events)
        self._trim_events.clear()
        return events

    def build(
        self,
        agent_configs: list[AgentConfig],
        models: Mapping[str, ResolvedModel],
        tool_overrides: dict[str, list] | None = None,
    ):
        """Build the LangGraph workflow from agent configs.

        Creates a graph where each agent config becomes a node,
        with edges based on sub_agent relationships. Each agent's chat model
        is built here, once (``chat_models``).

        Args:
            agent_configs: List of agent configurations.
            models: Each agent's resolved model, by agent name (the manager
                    passes the models it resolved at startup).
            tool_overrides: Optional mapping of agent name to pre-built tool lists.
                           When provided, these replace ``config.tools`` for tool
                           node creation and LLM binding.

        Returns:
            An uncompiled StateGraph.

        Raises:
            ValueError: If an agent has no resolved model.
        """
        from langgraph.graph import StateGraph, END
        from agentic_cli.workflow import langchain_models
        from agentic_cli.workflow.langgraph.state import AgentState

        # Build config map
        config_map = {config.name: config for config in agent_configs}

        # Create graph
        graph = StateGraph(AgentState)

        # Get retry policy for nodes
        retry_policy = self.get_retry_policy()

        _overrides = tool_overrides or {}

        def _get_tools(config: AgentConfig) -> list:
            return _overrides.get(config.name, config.tools or [])

        # Create nodes for each agent with retry policy, each on its own model
        chat_models = {}
        for config in agent_configs:
            if config.name not in models:
                raise ValueError(f"No resolved model for agent {config.name!r}.")
            model = models[config.name]
            chat_models[config.name] = langchain_models.build_chat_model(
                model, settings=self._settings, app_name=self._app_name
            )
            node_fn = self._create_agent_node(
                config, model, chat_models[config.name], tools=_get_tools(config) or None,
            )
            graph.add_node(config.name, node_fn, retry_policy=retry_policy)
        self.chat_models = chat_models

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
            config.name for config in agent_configs if _get_tools(config)
        }
        tool_map = {}
        for config in agent_configs:
            tools = _get_tools(config)
            if tools:
                tool_node_name = f"{config.name}_tools"
                tool_map[config.name] = tool_node_name
                wrapped_tools = [wrap_tool_for_permission(t) for t in tools]
                tool_node = ToolNode(
                    wrapped_tools, name=tool_node_name, handle_tool_errors=True,
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

    def _create_agent_node(
        self,
        config: AgentConfig,
        model: ResolvedModel,
        chat_model: "BaseChatModel",
        tools: list | None = None,
    ) -> Callable:
        """Create a LangGraph node function for an agent.

        Args:
            config: Agent configuration.
            model: The agent's resolved model.
            chat_model: The chat model built from it, which the node calls.
            tools: The agent's tools (default: ``config.tools``).

        Returns:
            Async function that processes state through the agent.
        """
        from agentic_cli.workflow import langchain_models

        agent_tools = tools or config.tools or []
        llm = chat_model.bind_tools(agent_tools) if agent_tools else chat_model
        # Claude on Anthropic's API caches the system prompt it is told to; a
        # model reached another way (OpenRouter, the user's server) does not.
        cache_system_prompt = (
            self._settings.prompt_caching_enabled
            and model.record.cache is CacheMechanism.ANTHROPIC
        )

        async def agent_node(state: dict) -> dict:
            """Process state through this agent."""
            from langchain_core.messages import SystemMessage

            # Build messages: system prompt + conversation history
            # State messages are already LangChain objects (add_messages reducer
            # auto-converts dicts), so we can use them directly.
            messages = []

            system_prompt = config.get_prompt()
            if system_prompt:
                if cache_system_prompt:
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

            # Another provider's reasoning (a model switch, a mixed tree) is
            # not this model's to read; the checkpoint keeps it.
            messages.extend(
                langchain_models.without_foreign_reasoning(conversation, model.record.kind)
            )

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
