"""Configuration classes for workflow management."""

from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class AgentConfig:
    """Configuration for an agent in the workflow.

    Agents are defined declaratively using this config, and the WorkflowManager
    creates the actual ADK Agent instances from these configs.

    Attributes:
        name: Unique identifier for the agent
        prompt: System instruction - either a string or a callable that returns one
        tools: List of tool functions the agent can use
        sub_agents: Names of agents that this agent can delegate to
        description: Short description for routing/logging
        model: Optional model override (defaults to manager's model)
    """

    name: str
    prompt: str | Callable[[], str]
    tools: list[Callable[..., Any]] = field(default_factory=list)
    sub_agents: list[str] = field(default_factory=list)
    description: str = ""
    model: str | None = None

    def get_prompt(self) -> str:
        """Get the prompt string, calling the getter if needed."""
        if callable(self.prompt):
            return self.prompt()
        return self.prompt
