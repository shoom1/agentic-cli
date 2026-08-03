"""Configuration classes for workflow management."""

import inspect
from dataclasses import dataclass, field
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.model_settings import ModelSettings
    from agentic_cli.workflow.mcp import MCPServerConfig


@dataclass
class AgentConfig:
    """Configuration for an agent in the workflow.

    Agents are defined declaratively using this config, and the workflow manager
    creates the actual agent instances from these configs.

    Attributes:
        name: Unique identifier for the agent
        prompt: System instruction — either a string or a callable returning one.
            The callable may take no arguments, or a single ``settings``
            argument, which receives the manager's settings instance (see
            ``get_prompt``).
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
        skills: Optional skill references (Agent Skills / SKILL.md folders) —
            each a path to a skill directory or a name resolved under
            ``settings.skills_dirs``. Currently consumed by the ADK backend only.
        include_state_tools: Whether to auto-inject plan/task state tools (default True)
    """

    name: str
    prompt: str | Callable[..., str]
    tools: list[Callable[..., Any] | str] = field(default_factory=list)
    sub_agents: list[str] = field(default_factory=list)
    description: str = ""
    model: str | None = None
    model_settings: "ModelSettings | None" = None
    mcp_servers: "list[MCPServerConfig]" = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    include_state_tools: bool = True

    def get_prompt(self, settings: "BaseSettings | None" = None) -> str:
        """Get the prompt string, calling the factory if the prompt is callable.

        Supported factory shapes, in the order they are tried:

        1. **Callable with no arguments** — including one whose parameters all
           have defaults (``lambda prefix="x": ...``). Called as-is; it should
           read ``get_settings()``, which the manager binds to its own instance
           while building agents.
        2. **Callable taking exactly one settings argument** — passed the
           manager's settings explicitly.

        Anything else (two required parameters, a required parameter that is
        not settings) is rejected: guessing would either drop the caller's
        intent or pass settings into an unrelated slot.

        Args:
            settings: The manager's settings, when available.

        Returns:
            The resolved system instruction.

        Raises:
            AgentGraphError: If the factory's signature is unsupported, it is
                async, or it does not return a string. The message names the
                agent.
        """
        if not callable(self.prompt):
            return self.prompt

        if inspect.iscoroutinefunction(self.prompt):
            raise AgentGraphError(
                f"Agent {self.name!r}: async prompt factories are not supported "
                "— the instruction is resolved synchronously while agents are "
                "built. Use a plain function."
            )

        result = self._call_prompt_factory(settings)
        if inspect.isawaitable(result):
            raise AgentGraphError(
                f"Agent {self.name!r}: the prompt factory returned an awaitable; "
                "it must return a string."
            )
        if not isinstance(result, str):
            raise AgentGraphError(
                f"Agent {self.name!r}: the prompt factory returned "
                f"{type(result).__name__}, not a string."
            )
        return result

    def _call_prompt_factory(self, settings: "BaseSettings | None"):
        """Invoke the factory with the arity it actually supports."""
        try:
            signature = inspect.signature(self.prompt)
        except (TypeError, ValueError):  # builtins / C callables
            return self.prompt()

        # Preferred and backward-compatible: if it binds with no arguments
        # (including all-defaulted parameters), call it with none.
        try:
            signature.bind()
        except TypeError:
            pass
        else:
            return self.prompt()

        if settings is None:
            raise AgentGraphError(
                f"Agent {self.name!r}: the prompt factory requires an argument "
                f"{signature}, but this caller resolved the prompt without "
                "settings. Use a zero-argument factory here."
            )

        try:
            signature.bind(settings)
        except TypeError:
            raise AgentGraphError(
                f"Agent {self.name!r}: unsupported prompt factory signature "
                f"{signature}. A prompt factory must take no arguments, or "
                "exactly one argument that receives the settings instance."
            ) from None
        return self.prompt(settings)


class AgentGraphError(ValueError):
    """A configured agent graph cannot be built.

    Raised before any backend object is constructed, so the message names the
    offending agents rather than surfacing as a partially-built hierarchy.
    """


@dataclass(frozen=True)
class AgentGraph:
    """A validated agent graph ready to construct, in dependency order.

    Attributes:
        config_map: Agent name → config.
        build_order: Names ordered so every agent follows its sub-agents.
        root_name: The agent the runner starts from.
    """

    config_map: dict[str, AgentConfig]
    build_order: tuple[str, ...]
    root_name: str


def validate_agent_graph(
    configs: list[AgentConfig], backend: str = "adk"
) -> AgentGraph:
    """Validate an agent graph and return it in dependency (topological) order.

    Checks, in order, so the first failure is the most fundamental:

    1. at least one agent;
    2. no duplicate agent names;
    3. every ``sub_agents`` entry resolves to a configured agent;
    4. no agent lists itself as a sub-agent;
    5. no delegation cycle;
    6. no agent is a sub-agent of two parents — ADK agents hold a single
       ``parent_agent``, so a shared child is a tree the backend cannot build.

    Args:
        configs: The declared agents.
        backend: Backend name, used only in error messages.

    Returns:
        The validated graph plus a build order in which every agent comes after
        the agents it delegates to.

    Raises:
        AgentGraphError: With the offending agent name(s) in the message.
    """
    if not configs:
        raise AgentGraphError("No agents configured: at least one AgentConfig is required.")

    config_map: dict[str, AgentConfig] = {}
    duplicates: list[str] = []
    for config in configs:
        if config.name in config_map:
            duplicates.append(config.name)
        config_map[config.name] = config
    if duplicates:
        raise AgentGraphError(
            f"Duplicate agent name(s): {', '.join(sorted(set(duplicates)))}. "
            "Agent names must be unique."
        )

    missing = [
        f"{config.name} -> {sub}"
        for config in configs
        for sub in config.sub_agents
        if sub not in config_map
    ]
    if missing:
        raise AgentGraphError(
            f"Unknown sub_agents reference(s): {', '.join(missing)}. "
            f"Known agents: {', '.join(sorted(config_map))}."
        )

    self_refs = [c.name for c in configs if c.name in c.sub_agents]
    if self_refs:
        raise AgentGraphError(
            f"Agent(s) list themselves as sub_agents: {', '.join(sorted(self_refs))}."
        )

    parents: dict[str, str] = {}
    shared: list[str] = []
    for config in configs:
        for sub in config.sub_agents:
            if sub in parents:
                shared.append(f"{sub} (of {parents[sub]} and {config.name})")
            else:
                parents[sub] = config.name
    if shared:
        raise AgentGraphError(
            f"The {backend} backend requires a tree: agent(s) with more than one "
            f"parent: {', '.join(sorted(shared))}."
        )

    order = _topological_order(config_map)
    # The root is the agent nobody delegates to. Selecting "first config with
    # sub_agents" made the root depend on declaration order (listing a
    # sub-coordinator before its parent promoted the child to root) and
    # silently accepted a forest: only one root is ever run, so every other
    # tree — and every agent under it — was unreachable.
    roots = [c.name for c in configs if c.name not in parents]
    if len(roots) > 1:
        raise AgentGraphError(
            f"Agent graph has {len(roots)} roots: {', '.join(sorted(roots))}. "
            "Exactly one agent may be unreferenced — the runner starts from a "
            "single root, so agents under any other root are unreachable. Add "
            "the extra root(s) to a coordinator's sub_agents."
        )
    return AgentGraph(config_map=config_map, build_order=order, root_name=roots[0])


def _topological_order(config_map: dict[str, AgentConfig]) -> tuple[str, ...]:
    """Order agent names so each follows its sub-agents.

    Raises:
        AgentGraphError: If a delegation cycle is found (names the cycle).
    """
    order: list[str] = []
    done: set[str] = set()
    visiting: list[str] = []

    def _visit(name: str) -> None:
        if name in done:
            return
        if name in visiting:
            cycle = visiting[visiting.index(name):] + [name]
            raise AgentGraphError(
                f"Delegation cycle in sub_agents: {' -> '.join(cycle)}."
            )
        visiting.append(name)
        for sub in config_map[name].sub_agents:
            _visit(sub)
        visiting.pop()
        done.add(name)
        order.append(name)

    for name in config_map:
        _visit(name)
    return tuple(order)
