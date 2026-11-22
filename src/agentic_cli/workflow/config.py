"""Configuration classes for workflow management."""

import inspect
from dataclasses import dataclass, field
from typing import Callable, Any, TYPE_CHECKING

from agentic_cli.workflow.agents.brief import BRIEF_FIELDS
from agentic_cli.logging import Loggers
from agentic_cli.workflow.agents.policy import (
    SPAWN_SOURCES,
    AgentLimits,
    SubAgentPolicy,
    as_policy,
)

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.model_settings import ModelSettings
    from agentic_cli.workflow.mcp import MCPServerConfig

logger = Loggers.workflow()

#: The brief fields a delegation must fill unless an agent asks for more.
_DEFAULT_BRIEF_REQUIRED = ("objective",)

#: What ``brief_required`` may name: the title, or a brief field.
_BRIEF_NAMES = ("title", *BRIEF_FIELDS)


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
        model: The agent's model: a ``provider:model`` reference, a bare
            model id, or a role (``fast``, ``strong``, or the app's own).
            None runs it on the ``default`` role (the manager's model).
        model_settings: Optional per-agent generation parameters (temperature,
            thinking, etc.), applied on both backends.
        mcp_servers: Optional MCP servers whose tools are exposed to this agent.
            Consumed by the ADK and Deep Agents backends; LangGraph ignores them.
        skills: Optional skill references (Agent Skills / SKILL.md folders) —
            each an absolute path to a skill directory or a name resolved under
            ``settings.skills_dirs`` (never in the working directory). Consumed
            by the ADK and Deep Agents backends; LangGraph ignores them.
        include_state_tools: Whether to auto-inject plan/task state tools (default True)
        inherit_context: Whether this agent also sees its parent's
            conversation when its parent delegates to it. False (the
            default): only the request it is given. True: also the parent's
            conversation as it stands, after any summarizing and without the
            parent's reasoning. A sub-agent works and returns its answer to
            its parent in the same turn on both backends; only the root
            replies to the user. True only on sub-agents, and only on the
            root's own sub-agents without sub-agents of their own
            (``validate_agent_graph`` checks it).
        brief_required: The brief fields a delegation to this agent must
            fill (``workflow/agents/brief.py``: ``objective``,
            ``deliverable``, ``constraints``, ``known``, ``done_when``, or
            ``title``); a brief always needs its title and its objective,
            so the default asks for nothing more. Only on sub-agents: the
            root is never delegated to.
        limits: This agent's own run limits (``AgentLimits``), over the
            policy's (``SubAgentPolicy.limits``) field by field: a field
            left None takes the policy's. None (the default) takes the
            policy's as they are. Only on sub-agents.
        continuable: Whether an instance of this agent takes follow-ups
            after its run. True (the default) keeps it ``idle`` for the
            next; False closes it when its run ends. Only on sub-agents.
        spawnable_by: Who may start this type at run time (a subset of
            ``SPAWN_SOURCES``: ``agent``, ``skill``, ``user``; a list or
            tuple is read as one), which the app's ``SubAgentPolicy``
            must allow too. None (the default): only its parent in
            ``sub_agents`` starts it. A type that sets it and is in no
            ``sub_agents`` list is a free type: no root, started at run time
            only. Only on sub-agents.
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
    inherit_context: bool = False
    brief_required: tuple[str, ...] = _DEFAULT_BRIEF_REQUIRED
    limits: AgentLimits | None = None
    continuable: bool = True
    spawnable_by: frozenset[str] | None = None

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
        free_types: The agent types that set ``spawnable_by`` and are in no
            ``sub_agents`` list, in config order: built like any sub-agent,
            started at run time only, never the root.
    """

    config_map: dict[str, AgentConfig]
    build_order: tuple[str, ...]
    root_name: str
    free_types: tuple[str, ...] = ()


def validate_agent_graph(
    configs: list[AgentConfig],
    backend: str = "adk",
    *,
    policy: SubAgentPolicy | None = None,
) -> AgentGraph:
    """Validate an agent graph and return it in dependency (topological) order.

    Checks, in order, so the first failure is the most fundamental:

    1. at least one agent;
    2. no duplicate agent names;
    3. every ``sub_agents`` entry resolves to a configured agent;
    4. no agent lists itself as a sub-agent;
    5. no delegation cycle;
    6. no agent is a sub-agent of two parents — ADK agents hold a single
       ``parent_agent``, so a shared child is a tree the backend cannot build;
    7. exactly one root among the agents that are not free types: each
       agent's ``spawnable_by`` is checked first (None, or a non-empty set,
       list or tuple of ``SPAWN_SOURCES``), a free type is one that sets it
       and is in no ``sub_agents`` list, and the root may not set it;
    8. each agent's ``inherit_context`` is True or False;
    9. the root does not set ``inherit_context``: it has no parent's
       conversation;
    10. ``inherit_context`` only on an agent without sub-agents, and only on
        the root's own sub-agents;
    11. each agent's ``brief_required`` is a tuple (or list) of brief field
        names: ``title``, or one of ``BRIEF_FIELDS``;
    12. each agent's ``limits`` is None or an ``AgentLimits``;
    13. each agent's ``continuable`` is True or False;
    14. the root sets none of ``brief_required`` (beyond its default, the
        objective), ``limits`` and ``continuable`` (beyond True): it is never
        delegated to.

    A free type that no source ``policy`` enables can start is warned of
    (``subagent_type_unreachable``), once for each validation.

    Args:
        configs: The declared agents.
        backend: Backend name, used only in error messages.
        policy: The app's sub-agent policy (None: ``SubAgentPolicy()``);
            only the free types' warning reads it.

    Returns:
        The validated graph plus a build order in which every agent comes after
        the agents it delegates to.

    Raises:
        AgentGraphError: With the offending agent name(s) in the message.
        TypeError: ``policy`` is neither a ``SubAgentPolicy`` nor None.
    """
    policy = as_policy(policy)
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
    _check_spawnable_by(configs)
    free = tuple(
        c.name for c in configs if c.spawnable_by is not None and c.name not in parents
    )
    # The root is the agent nobody delegates to that is no free type.
    # Selecting "first config with
    # sub_agents" made the root depend on declaration order (listing a
    # sub-coordinator before its parent promoted the child to root) and
    # silently accepted a forest: only one root is ever run, so every other
    # tree — and every agent under it — was unreachable.
    roots = [c.name for c in configs if c.name not in parents and c.name not in free]
    if not roots:
        if len(free) == 1:
            raise AgentGraphError(
                f"Agent {free[0]!r} is the root agent and cannot set spawnable_by: "
                "only a sub-agent type is started at run time."
            )
        raise AgentGraphError(
            f"Agent graph has no root: every agent nobody delegates to ({', '.join(free)}) "
            "sets spawnable_by, which makes it a type started at run time. Leave "
            "spawnable_by off the agent the runner starts from."
        )
    if len(roots) > 1:
        raise AgentGraphError(
            f"Agent graph has {len(roots)} roots: {', '.join(sorted(roots))}. "
            "Exactly one agent may be unreferenced — the runner starts from a "
            "single root, so agents under any other root are unreachable. Add "
            "the extra root(s) to a coordinator's sub_agents."
        )
    _check_inherit_context(config_map, roots[0])
    _check_delegation_fields(config_map, roots[0])
    _warn_unreachable([config_map[name] for name in free], policy)
    return AgentGraph(
        config_map=config_map, build_order=order, root_name=roots[0], free_types=free
    )


def _check_spawnable_by(configs: list[AgentConfig]) -> None:
    """Each agent's ``spawnable_by`` is None, or a non-empty set, list or tuple of sources.

    Raises:
        AgentGraphError: Naming the agent, the value and what to change.
    """
    allowed = ", ".join(repr(source) for source in SPAWN_SOURCES)
    for config in configs:
        value = config.spawnable_by
        if value is None:
            continue
        if isinstance(value, (set, frozenset, list, tuple)) and not value:
            raise AgentGraphError(
                f"Agent {config.name!r} has an empty spawnable_by: name at least one "
                "source, or leave it out for a type only its parent starts."
            )
        if not isinstance(value, (set, frozenset, list, tuple)) or not all(
            isinstance(item, str) and item in SPAWN_SOURCES for item in value
        ):
            raise AgentGraphError(
                f"Agent {config.name!r} has spawnable_by {value!r}: each source must be "
                f"one of {allowed}."
            )


def _warn_unreachable(free: list[AgentConfig], policy: SubAgentPolicy) -> None:
    """Log each free type that no source the policy enables can start.

    An app may define types it enables per install, so this is a warning.
    """
    for config in free:
        if not set(config.spawnable_by or ()) & policy.spawn_sources:
            logger.warning(
                "subagent_type_unreachable",
                agent=config.name,
                spawnable_by=sorted(config.spawnable_by or ()),
                spawn_sources=sorted(policy.spawn_sources),
            )


def _check_inherit_context(config_map: dict[str, AgentConfig], root: str) -> None:
    """Each agent's ``inherit_context`` fits its place in the tree.

    Raises:
        AgentGraphError: Naming the agent(s) and what to change.
    """
    # Not coerced: "fresh", a value of this field's earlier string form, is
    # truthy, and would show a sub-agent the whole conversation.
    unknown = [
        f"{name!r} ({config.inherit_context!r})"
        for name, config in config_map.items()
        if not isinstance(config.inherit_context, bool)
    ]
    if unknown:
        raise AgentGraphError(
            f"inherit_context must be True or False: {', '.join(unknown)}. "
            "True shows a sub-agent its parent's conversation too; False (the "
            "default) the request alone."
        )

    if config_map[root].inherit_context:
        raise AgentGraphError(
            f"Agent {root!r} is the root agent, but sets inherit_context=True: "
            "inherit_context applies to sub-agents only. Remove it from the root."
        )

    inheriting = sorted(
        name
        for name, config in config_map.items()
        if config.inherit_context and config.sub_agents
    )
    if inheriting:
        raise AgentGraphError(
            f"Agent(s) {', '.join(repr(n) for n in inheriting)} set "
            "inherit_context=True and have sub-agents of their own: "
            "inherit_context is for agents without sub-agents. Leave it False "
            "on a coordinator."
        )

    # Inheriting is for the main agent's own sub-agents (rule 10).
    nested = sorted({
        sub
        for name, config in config_map.items()
        if name != root
        for sub in config.sub_agents
        if config_map[sub].inherit_context
    })
    if nested:
        raise AgentGraphError(
            f"Agent(s) {', '.join(repr(n) for n in nested)} set inherit_context=True "
            f"below another sub-agent: inherit_context is for the main "
            f"agent's ({root!r}) own sub-agents. Leave it False, or make it a "
            f"sub-agent of {root!r}."
        )


def _check_delegation_fields(config_map: dict[str, AgentConfig], root: str) -> None:
    """Each agent's ``brief_required``, ``limits`` and ``continuable`` are what they say, and the root sets none.

    Raises:
        AgentGraphError: Naming the agent(s), the value and what to change.
    """
    unknown = [
        f"{name!r} ({config.brief_required!r})"
        for name, config in config_map.items()
        if not _names_brief_fields(config.brief_required)
    ]
    if unknown:
        raise AgentGraphError(
            f"brief_required must be a tuple or list of brief field names "
            f"({', '.join(_BRIEF_NAMES)}): {', '.join(unknown)}."
        )

    not_limits = [
        f"{name!r} ({config.limits!r})"
        for name, config in config_map.items()
        if config.limits is not None and not isinstance(config.limits, AgentLimits)
    ]
    if not_limits:
        raise AgentGraphError(
            f"limits must be an AgentLimits or None: {', '.join(not_limits)}. "
            "Write limits=AgentLimits(max_model_calls=..., timeout=..., max_tokens=...)."
        )

    not_bool = [
        f"{name!r} ({config.continuable!r})"
        for name, config in config_map.items()
        if not isinstance(config.continuable, bool)
    ]
    if not_bool:
        raise AgentGraphError(
            f"continuable must be True or False: {', '.join(not_bool)}. "
            "True (the default) keeps a sub-agent's instance for follow-ups; "
            "False closes it when its run ends."
        )

    config = config_map[root]
    on_root = [
        name
        for name, is_set in (
            ("brief_required", tuple(config.brief_required) != _DEFAULT_BRIEF_REQUIRED),
            ("limits", config.limits is not None),
            ("continuable", config.continuable is False),
        )
        if is_set
    ]
    if on_root:
        raise AgentGraphError(
            f"Agent {root!r} is the root agent, but sets {' and '.join(on_root)}: "
            "they apply to a delegation, and the root is never delegated to. "
            "Remove them from the root."
        )


def _names_brief_fields(value: object) -> bool:
    """Whether ``value`` is a tuple or list of brief field names (a string is neither)."""
    return isinstance(value, (tuple, list)) and all(
        isinstance(name, str) and name in _BRIEF_NAMES for name in value
    )


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
