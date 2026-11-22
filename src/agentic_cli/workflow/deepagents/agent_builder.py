"""The agents: one ``create_agent`` graph per ``AgentConfig``, each standalone.

Every agent is its own graph, built apart from the others, on the manager's
checkpointer (a sub-agent's runs are on a thread of its instance's own, which
its follow-ups continue); a tree built without a checkpointer compiles its
sub-agents with none. An agent with sub-agents, its own or types it may start
at run time (``spawn.can_have_children``), gets the framework's ``task`` tool
(``task_tool.build_task_tool``) among its tools (the root's offer fixed when
it is built, a sub-agent's rewritten for each model call by
``RunControlMiddleware``; with ``SubAgentPolicy.follow_ups``, also the
parent's other tools, ``task_tool.build_follow_up_tools``): a call hands the
brief to the manager's ``Delegations``, which runs the sub-agent's graph apart
from the caller's stream (``DeepAgentsWorkflowManager._run_agent``) and
answers with the result envelope. Only the configured sub-agents are offered:
there is no automatic general-purpose sub-agent, and none of Deep Agents' file
tools. ``graphs``, when given, is filled with every agent's graph by name:
what the manager runs a sub-agent on. A sub-agent type's model choices are
built once more each (``Variant``, ``variants=``): the type's graph on
another model, filled into ``variant_graphs`` by key.

Every stack, outermost first: ``InterruptedCallsMiddleware`` (answers a tool
call a cancelled turn or a crash left without a result, before each run, as
``workflow/interrupts.py`` says, and, the outermost model wrapper, gives a
call whose id the agent's state already holds an id of its own; Deep
Agents' own ``PatchToolCallsMiddleware`` without the manager's record), then
the model side and the tool side.

- Model side, outermost first: a sub-agent's refused or cut final answer is
  said so (``SubAgentOutcomeMiddleware``, every agent but the root), a
  sub-agent run's limits are kept at its model calls (``RunControlMiddleware``,
  every agent but the root, given the manager's delegations), other
  providers' and other agents' reasoning
  is left out of the request (``ForeignReasoningMiddleware``), Deep Agents'
  summarization counts
  exactly what will be sent (``summarization.py``), every provider's context
  overflow reaches it as one it summarizes on (``ContextOverflowMiddleware``),
  the roster of its sub-agents, for the main agent that continues them
  (``RosterMiddleware``), then the memory index, for the main agent with a
  memory tool (``MemoryIndexMiddleware``), go first in the request (the
  index ahead of the roster), and on Claude
  with ``prompt_caching_enabled`` the cache marks land on the
  final request (``AnthropicPromptCachingMiddleware``: the system prompt, the
  last tool and the conversation, for 5 minutes, as on ADK).
- Tool side, outermost first: the permission gate (``task`` among the tools
  it gates), the tool result cap
  (``ToolResultCapMiddleware``: a result over ``tool_result_max_chars`` is
  cut, its full output saved; an error result too), the tool-error
  middleware, and the active tool call (a job a tool starts is linked to its
  call), so only a call the gate let through runs as one.

The model side is built from the agent's resolved model and the settings,
and the cap from the settings; a tree built without settings (a unit test's)
has neither.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from langchain.agents import create_agent
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

from agentic_cli.models import CacheMechanism
from agentic_cli.tools._core.delegation import FOLLOW_UP_CLASH_ERROR, FOLLOW_UP_TOOLS, TASK_TOOL
from agentic_cli.tools.registry import ToolDefinition, identify_tool
from agentic_cli.workflow.agents.spawn import can_have_children
from agentic_cli.workflow.config import AgentConfig, AgentGraphError
from agentic_cli.workflow.deepagents.middleware import (
    ActiveToolCallMiddleware,
    ContextOverflowMiddleware,
    ForeignReasoningMiddleware,
    InterruptedCallsMiddleware,
    MemoryIndexMiddleware,
    PermissionMiddleware,
    RosterMiddleware,
    RunControlMiddleware,
    SubAgentOutcomeMiddleware,
    ToolCallErrorMiddleware,
    ToolResultCapMiddleware,
)
from agentic_cli.workflow.deepagents.state import AgentCliState
from agentic_cli.workflow.deepagents.task_tool import build_follow_up_tools, build_task_tool
from agentic_cli.workflow.context import ContextBudget
from agentic_cli.workflow.interrupts import OpenCalls
from agentic_cli.workflow.outside_content import OutsideContentMarks
from agentic_cli.workflow.deepagents.summarization import (
    ReportingSummarizationMiddleware,
    build_summarization,
)
from agentic_cli.workflow.tool_output import tool_output_dir

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from agentic_cli.config import BaseSettings
    from agentic_cli.models import ResolvedModel
    from agentic_cli.workflow.agents.delegate import Delegations


@dataclass(frozen=True)
class Variant:
    """One more graph of a sub-agent type, on another model (a model choice's).

    Built as the type's own is, with its tools and prompt, but its model
    side from ``resolved`` and its summarization from ``budget``.

    Attributes:
        config: The type.
        chat_model: The chat model of the choice.
        resolved: The choice's resolved model (None: a unit test's, with no model side).
        budget: The choice's context numbers; None: computed from ``resolved``.
    """

    config: AgentConfig
    chat_model: "BaseChatModel"
    resolved: "ResolvedModel | None"
    budget: ContextBudget | None = None


def build_agent_tree(
    configs: list[AgentConfig],
    *,
    chat_models: Mapping[str, "BaseChatModel"],
    resolved: Mapping[str, "ResolvedModel"],
    tools: Mapping[str, list["BaseTool"]],
    prompts: Mapping[str, str],
    settings: "BaseSettings | None",
    checkpointer: "BaseCheckpointSaver | None",
    bound: Callable[[object], ToolDefinition | None] = identify_tool,
    budgets: Mapping[str, ContextBudget] | None = None,
    summarizers: dict[str, ReportingSummarizationMiddleware] | None = None,
    open_calls: OpenCalls | None = None,
    marks: OutsideContentMarks | None = None,
    memory_index: Callable[[], str] | None = None,
    memory_index_cached: Callable[[], str] | None = None,
    memory_index_forget: Callable[[], None] | None = None,
    delegations: "Callable[[], Delegations] | None" = None,
    graphs: "dict[str, CompiledStateGraph] | None" = None,
    roster: Callable[[], str] | None = None,
    root: str | None = None,
    variants: "Mapping[Hashable, Variant] | None" = None,
    variant_graphs: "dict[Hashable, CompiledStateGraph] | None" = None,
) -> "CompiledStateGraph":
    """Build every agent, each on its own, and return the root.

    Validates nothing: the manager runs ``validate_agent_graph`` first, so
    ``configs`` holds one root (the agent nobody delegates to that is no
    free type) and the free types, which are built as any sub-agent is.

    Args:
        configs: The agents.
        chat_models: Each agent's chat model, by name.
        resolved: Each agent's resolved model, by name. With ``settings``,
            this is what the model-side middleware (foreign reasoning,
            summarization and prompt caching) is configured from.
        tools: Each agent's LangChain tools, by name (none if absent).
        prompts: Each agent's system prompt, already resolved.
        settings: The manager's settings: the model side and the tool result
            cap are built from them.
        checkpointer: The checkpointer every agent is compiled on: a
            sub-agent's runs are on threads of their own, one per instance
            (``deepagents/sessions.py::instance_run_config``). None: the
            root has none, and the other agents are compiled with none
            (``checkpointer=False``), not left to take a caller's.
        bound: What every agent's permission gate asks first, for a tool
            object the framework built and bound. Defaults to the framework
            registry's identity map; a manager with tool objects of its own
            (its MCP tools) passes its map chained before it.
        budgets: Each agent's context numbers, by name, as the manager computed
            them (``BaseWorkflowManager._compute_budgets``): its summarization
            runs on them. An agent without one (or no mapping) gets them
            computed from its resolved model and the settings.
        summarizers: Filled, when given, with each agent's summarization
            middleware, by name (none for an agent built without settings):
            the manager keeps the root's to summarize on demand
            (``compact``).
        open_calls: The manager's record of its calls, for answering those
            a cancelled turn leaves open (``InterruptedCallsMiddleware``).
        marks: The manager's record of each conversation's mark, which every
            agent's permission gate marks when it allows a call that brings
            outside content (``workflow/outside_content.py``).
        memory_index: The root's memory index for its conversation (``""``
            for none): given, the root's stack has ``MemoryIndexMiddleware``
            (the manager passes it only for a root with a memory tool). No
            other agent gets one.
        memory_index_cached: The same index, which reads no file: the root's
            summarization counts it in its requests' size.
        memory_index_forget: Drops the conversation's index and roster
            snapshots: the root's summarization calls it when it writes a
            summary, before the request that carries the summary is sent.
        delegations: The manager's delegations (``BaseWorkflowManager._delegations``):
            given, an agent that can have sub-agents (its own, or types it may
            start at run time) gets the ``task`` tool, which hands each call
            to them, and every agent but the root
            ``RunControlMiddleware``, which keeps a run's limits. Without it
            no agent delegates (a tree a unit test builds alone).
        graphs: Filled, when given, with every agent's graph, by name.
        roster: The root's roster of its open sub-agents for its conversation
            (``""`` for none): given, the root's stack has ``RosterMiddleware``
            and its summarization counts it (the manager passes it only for a
            root that continues its sub-agents). No other agent gets one.
        root: The root's name (the validated graph's ``AgentGraph.root_name``).
            None: the first agent nobody delegates to that sets no
            ``spawnable_by``.
        variants: Sub-agent types on other models (a type's model choices),
            by key: each is built as the type's graph is, from its own chat
            model, resolved model and budget, with the type's tools and
            prompt, and filled into ``variant_graphs``.
        variant_graphs: Filled, when given, with each variant's graph, by key.

    Raises:
        AgentGraphError: A delegating agent's own tools include one named
            ``task``, the name of its delegation tool.
    """
    config_map = {config.name: config for config in configs}
    delegated = {sub for config in configs for sub in config.sub_agents}
    if root is None:
        root = next(
            config.name for config in configs
            if config.name not in delegated and config.spawnable_by is None
        )

    def build(name: str) -> "CompiledStateGraph":
        is_root = name == root
        return build_agent(
            config_map[name], root=is_root, configs=configs, chat_model=chat_models[name],
            resolved=resolved.get(name), tools=tools.get(name, []), prompt=prompts[name],
            settings=settings, checkpointer=checkpointer, bound=bound,
            budget=budgets.get(name) if budgets is not None else None,
            summarizers=summarizers, open_calls=open_calls, marks=marks,
            memory_index=memory_index if is_root else None,
            memory_index_cached=memory_index_cached if is_root else None,
            memory_index_forget=memory_index_forget if is_root else None,
            delegations=delegations, roster=roster if is_root else None,
        )

    built = {name: build(name) for name in config_map}
    if graphs is not None:
        graphs.update(built)
    for key, variant in (variants or {}).items():
        name = variant.config.name
        graph = build_agent(
            variant.config, root=False, configs=configs, chat_model=variant.chat_model,
            resolved=variant.resolved, tools=tools.get(name, []), prompt=prompts[name],
            settings=settings, checkpointer=checkpointer, bound=bound, budget=variant.budget,
            open_calls=open_calls, marks=marks, delegations=delegations,
        )
        if variant_graphs is not None:
            variant_graphs[key] = graph
    return built[root]


def build_agent(
    config: AgentConfig,
    *,
    root: bool,
    configs: list[AgentConfig],
    chat_model: "BaseChatModel",
    resolved: "ResolvedModel | None",
    tools: list["BaseTool"],
    prompt: str,
    settings: "BaseSettings | None",
    checkpointer: "BaseCheckpointSaver | None",
    bound: Callable[[object], ToolDefinition | None] = identify_tool,
    budget: ContextBudget | None = None,
    summarizers: dict[str, ReportingSummarizationMiddleware] | None = None,
    open_calls: OpenCalls | None = None,
    marks: OutsideContentMarks | None = None,
    memory_index: Callable[[], str] | None = None,
    memory_index_cached: Callable[[], str] | None = None,
    memory_index_forget: Callable[[], None] | None = None,
    delegations: "Callable[[], Delegations] | None" = None,
    roster: Callable[[], str] | None = None,
) -> "CompiledStateGraph":
    """Build one agent's graph: ``config`` on ``chat_model``, with its tools and prompt.

    What ``build_agent_tree`` builds each agent with (its arguments, for one
    agent), and builds a model choice's variant of a sub-agent type with.
    ``root`` says whether this is the root: only it has the memory index and
    the roster (passed only for it), and every other agent has the run
    control and the outcome middleware; a sub-agent's graph has no
    checkpointer of its own unless the tree has one.

    Raises:
        AgentGraphError: A delegating agent's own tools include one named
            ``task``, the name of its delegation tool.
    """
    name = config.name
    model_side = _model_side(
        chat_model, resolved, settings, budget=budget,
        memory_index=memory_index, memory_index_cached=memory_index_cached,
        memory_index_forget=memory_index_forget, roster=roster,
    )
    if not root:
        # Inside the outcome middleware, which reads the answer whose
        # calls it dropped as final; outside the summarization, which
        # counts its note.
        if delegations is not None:
            model_side = [RunControlMiddleware(delegations), *model_side]
        # Outermost: it sees the answer every other model-side
        # middleware let through, as the run's answer is read.
        model_side = [SubAgentOutcomeMiddleware(), *model_side]
    if summarizers is not None:
        summarizers.update(
            (name, middleware) for middleware in model_side
            if isinstance(middleware, ReportingSummarizationMiddleware)
        )
    delegates = delegations is not None and can_have_children(
        config, configs, delegations().policy, root=root
    )
    follows = delegates and delegations is not None and delegations().policy.follow_ups
    own = _own_tools(config, tools, delegates=delegates, follows=follows)
    if delegates and delegations is not None:
        own = [*own, build_task_tool(name, delegations, root=root)]
        if follows:
            own = [*own, *build_follow_up_tools(delegations)]
    return create_agent(
        model=chat_model,
        tools=own,
        system_prompt=prompt,
        middleware=[
            PatchToolCallsMiddleware() if open_calls is None
            else InterruptedCallsMiddleware(open_calls, root=root),
            *model_side,
            *_tool_side(config, settings, bound=bound, open_calls=open_calls, marks=marks),
        ],
        state_schema=AgentCliState,
        name=name,
        checkpointer=checkpointer if root or checkpointer is not None else False,
    )


def _own_tools(
    config: AgentConfig, tools: list["BaseTool"], *, delegates: bool, follows: bool = False
) -> list["BaseTool"]:
    """The agent's tools; a delegating agent's must leave ``task`` to delegation.

    A model is offered one tool per name, so a delegating agent's own tool
    named ``task`` would take its delegation tool's place, or lose it; so
    would one named like a tool of the parent's that ``follows`` (the policy's
    ``follow_ups``) gives it.
    """
    if delegates and any(tool.name == TASK_TOOL for tool in tools):
        raise AgentGraphError(
            f"Agent {config.name!r} delegates to sub-agents through a tool named "
            f"{TASK_TOOL!r}, so its own tools cannot include one; rename that tool."
        )
    if follows:
        own_names = {tool.name for tool in tools}
        # The first in ``FOLLOW_UP_TOOLS`` order, as on ADK.
        clash = next((name for name in FOLLOW_UP_TOOLS if name in own_names), None)
        if clash is not None:
            raise AgentGraphError(FOLLOW_UP_CLASH_ERROR.format(agent=config.name, tool=clash))
    return list(tools)


def _tool_side(
    config: AgentConfig,
    settings: "BaseSettings | None",
    *,
    bound: Callable[[object], ToolDefinition | None],
    open_calls: OpenCalls | None = None,
    marks: OutsideContentMarks | None = None,
) -> list["AgentMiddleware"]:
    """What runs each of the agent's tool calls, outermost first."""
    stack: list[AgentMiddleware] = [
        PermissionMiddleware(config.name, bound=bound, open_calls=open_calls, marks=marks)
    ]
    if settings is not None:
        # Inside the gate, whose denials are short; outside the tool-error
        # middleware, so an error result is capped too.
        stack.append(
            ToolResultCapMiddleware(
                settings.tool_result_max_chars, tool_output_dir(settings.app_name)
            )
        )
    stack += [ToolCallErrorMiddleware(), ActiveToolCallMiddleware()]
    return stack


def _model_side(
    model: "BaseChatModel",
    resolved: "ResolvedModel | None",
    settings: "BaseSettings | None",
    *,
    budget: ContextBudget | None = None,
    memory_index: Callable[[], str] | None = None,
    memory_index_cached: Callable[[], str] | None = None,
    memory_index_forget: Callable[[], None] | None = None,
    roster: Callable[[], str] | None = None,
) -> list["AgentMiddleware"]:
    """What shapes each of the agent's model requests, outermost first."""
    if resolved is None or settings is None:
        return []
    stack: list[AgentMiddleware] = [
        ForeignReasoningMiddleware(resolved.record.kind, agent=resolved.agent),
        build_summarization(
            model, resolved, settings, budget=budget,
            index=memory_index_cached, roster=roster, on_written=memory_index_forget,
        ),
        # Inside the summarization: it summarizes on the overflows it raises.
        ContextOverflowMiddleware(),
    ]
    if roster is not None:
        # Outside the memory index's, which then goes ahead of it.
        stack.append(RosterMiddleware(roster))
    if memory_index is not None:
        # Inside the summarization, which counts it, and before the caching.
        stack.append(MemoryIndexMiddleware(memory_index))
    if resolved.record.cache is CacheMechanism.ANTHROPIC and settings.prompt_caching_enabled:
        # The system prompt is a plain string: the middleware marks it.
        stack.append(
            AnthropicPromptCachingMiddleware(ttl="5m", unsupported_model_behavior="ignore")
        )
    return stack
