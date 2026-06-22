"""Configuration classes for workflow management."""

from dataclasses import dataclass, field
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_cli.workflow.model_settings import ModelSettings
    from agentic_cli.workflow.mcp import MCPServerConfig


@dataclass
class AgentConfig:
    """Configuration for an agent in the workflow.

    Agents are defined declaratively using this config, and the workflow manager
    creates the actual agent instances from these configs.

    Attributes:
        name: Unique identifier for the agent
        prompt: System instruction - either a string or a callable that returns one
        tools: Tools the agent can use. Each entry is a callable, a registered
            tool name (e.g. "kb_search"), or a dotted import path
            (e.g. "my_pkg.tools.my_tool"). String refs are resolved to callables
            when the workflow manager is constructed.
        sub_agents: Names of agents that this agent can delegate to
        description: Short description for routing/logging
        model: Optional model override (defaults to manager's model)
        model_settings: Optional per-agent generation parameters (temperature,
            thinking, etc.). Currently consumed by the ADK backend only.
        mcp_servers: Optional MCP servers whose tools are exposed to this agent.
            Currently consumed by the ADK backend only.
        include_state_tools: Whether to auto-inject plan/task state tools (default True)
    """

    name: str
    prompt: str | Callable[[], str]
    tools: list[Callable[..., Any] | str] = field(default_factory=list)
    sub_agents: list[str] = field(default_factory=list)
    description: str = ""
    model: str | None = None
    model_settings: "ModelSettings | None" = None
    mcp_servers: "list[MCPServerConfig]" = field(default_factory=list)
    include_state_tools: bool = True

    def get_prompt(self) -> str:
        """Get the prompt string, calling the getter if needed."""
        if callable(self.prompt):
            return self.prompt()
        return self.prompt
