"""The agent tree: one ``create_agent`` graph per ``AgentConfig``.

Agents are built bottom-up, so each coordinator is built after the sub-agents
it delegates to. A coordinator gets Deep Agents' ``SubAgentMiddleware`` with
one ``CompiledSubAgent`` per sub-agent, which gives it a ``task`` tool
(``description``: the brief, ``subagent_type``: the sub-agent's name). Only
the configured sub-agents are offered: there is no automatic general-purpose
sub-agent, and none of Deep Agents' file tools.

What a sub-agent sees follows its ``context``. A ``fresh`` one (the default)
is Deep Agents' isolated sub-agent: one ``HumanMessage`` holding the brief,
not the conversation. An ``inherit`` one is a fork (``mode="fork"``): the
coordinator's effective conversation (after any summary), without the
message that called ``task``, then Deep Agents' preamble and the brief; its
own summarization compacts an inherited history too large for its window,
and its stack sends none of the coordinator's reasoning on. Either way it
gets the coordinator's ``plan`` and ``tasks``, keeps its own prompt, tools,
model and middleware, runs its own tool loop, and its last non-empty text
comes back as the ``task`` call's result, with the ``plan``/``tasks`` it
changed. Deep Agents marks forks beta with a ``LangChainBetaWarning``
LangChain would print over the UI; that one warning is filtered, and a
debug log (``deepagents_fork_beta``) says which agents fork.

Every stack, outermost first: ``InterruptedCallsMiddleware`` (answers a
tool call a cancelled turn or a crash left without a result, before each
run, as ``workflow/interrupts.py`` says; Deep Agents' own
``PatchToolCallsMiddleware`` without the manager's record), the
coordinator's ``SubAgentMiddleware`` (so its ``task`` tool is gated like any
other), then the model side and the tool side.

- Model side, outermost first: other providers' and other agents' reasoning
  is left out of the request (``ForeignReasoningMiddleware``), Deep Agents'
  summarization counts
  exactly what will be sent (``summarization.py``), every provider's context
  overflow reaches it as one it summarizes on (``ContextOverflowMiddleware``),
  and on Claude with ``prompt_caching_enabled`` the cache marks land on the
  final request (``AnthropicPromptCachingMiddleware``: the system prompt, the
  last tool and the conversation, for 5 minutes, as on ADK).
- Tool side, outermost first: the permission gate, the tool result cap
  (``ToolResultCapMiddleware``: a result over ``tool_result_max_chars`` is
  cut, its full output saved; an error result too), the tool-error
  middleware, and the active tool call (a job a tool starts is linked to its
  call), so only a call the gate let through runs as one.

The model side is built from the agent's resolved model and the settings,
and the cap from the settings; a tree built without settings (a unit test's)
has neither.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from deepagents import CompiledSubAgent, SubAgentMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from langchain.agents import create_agent
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core._api import LangChainBetaWarning

from agentic_cli.logging import Loggers
from agentic_cli.models import CacheMechanism
from agentic_cli.tools._core.delegation import TASK_TOOL
from agentic_cli.tools.registry import (
    ToolDefinition,
    bind_tool_identity,
    get_registry,
    identify_tool,
)
from agentic_cli.workflow.config import AgentConfig, AgentGraphError
from agentic_cli.workflow.deepagents.middleware import (
    ActiveToolCallMiddleware,
    ContextOverflowMiddleware,
    ForeignReasoningMiddleware,
    InterruptedCallsMiddleware,
    PermissionMiddleware,
    ToolCallErrorMiddleware,
    ToolResultCapMiddleware,
)
from agentic_cli.workflow.deepagents.state import AgentCliState
from agentic_cli.workflow.interrupts import OpenCalls
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

logger = Loggers.workflow()

#: What Deep Agents warns when a coordinator forks a sub-agent.
_FORK_BETA = r"The feature `forked subagents` is in beta"


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
    summarizers: dict[str, ReportingSummarizationMiddleware] | None = None,
    open_calls: OpenCalls | None = None,
) -> "CompiledStateGraph":
    """Build every agent, bottom-up, and return the root.

    Validates nothing: the manager runs ``validate_agent_graph`` first, so
    ``configs`` is a tree with one root (the agent nobody delegates to).

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
        checkpointer: The root's checkpointer. Sub-agents get none: they
            inherit it through LangGraph's nested-graph rules, so their
            checkpoints sit in the root's thread under their task's
            namespace.
        bound: What every agent's permission gate asks first, for a tool
            object the framework built and bound. Defaults to the framework
            registry's identity map; a manager with tool objects of its own
            (its MCP tools) passes its map chained before it.
        summarizers: Filled, when given, with each agent's summarization
            middleware, by name (none for an agent built without settings):
            the manager keeps the root's to summarize on demand
            (``compact``).
        open_calls: The manager's record of its calls, for answering those
            a cancelled turn leaves open (``InterruptedCallsMiddleware``).

    Raises:
        AgentGraphError: A coordinator's own tools include one named
            ``task``, the name of its delegation tool.
    """
    config_map = {config.name: config for config in configs}
    delegated = {sub for config in configs for sub in config.sub_agents}
    root = next(config.name for config in configs if config.name not in delegated)

    def build(name: str) -> "CompiledStateGraph":
        config = config_map[name]
        subagents = [
            _subagent(config_map[sub], build(sub)) for sub in config.sub_agents
        ]
        model = chat_models[name]
        model_side = _model_side(model, resolved.get(name), settings)
        if summarizers is not None:
            summarizers.update(
                (name, middleware) for middleware in model_side
                if isinstance(middleware, ReportingSummarizationMiddleware)
            )
        return create_agent(
            model=model,
            tools=_own_tools(config, tools.get(name, []), delegates=bool(subagents)),
            system_prompt=prompts[name],
            middleware=_middleware(
                subagents,
                first=(
                    PatchToolCallsMiddleware() if open_calls is None
                    else InterruptedCallsMiddleware(open_calls, root=name == root)
                ),
                model_side=model_side,
                tool_side=_tool_side(config, settings, bound=bound, open_calls=open_calls),
            ),
            state_schema=AgentCliState,
            name=name,
            checkpointer=checkpointer if name == root else None,
        )

    return build(root)


def _own_tools(
    config: AgentConfig, tools: list["BaseTool"], *, delegates: bool
) -> list["BaseTool"]:
    """The agent's tools; a coordinator's must leave ``task`` to delegation.

    The tool node keeps one tool per name, the agent's own after the
    middleware's, so a coordinator's tool named ``task`` would silently take
    delegation over.
    """
    if delegates and any(tool.name == TASK_TOOL for tool in tools):
        raise AgentGraphError(
            f"Agent {config.name!r} delegates to sub-agents through a tool named "
            f"{TASK_TOOL!r}, so its own tools cannot include one; rename that tool."
        )
    return list(tools)


def _middleware(
    subagents: list[CompiledSubAgent],
    *,
    first: "AgentMiddleware",
    model_side: list["AgentMiddleware"],
    tool_side: list["AgentMiddleware"],
) -> list["AgentMiddleware"]:
    stack: list[AgentMiddleware] = [first]
    if subagents:
        stack.append(_delegation(subagents))
    return stack + model_side + tool_side


def _tool_side(
    config: AgentConfig,
    settings: "BaseSettings | None",
    *,
    bound: Callable[[object], ToolDefinition | None],
    open_calls: OpenCalls | None = None,
) -> list["AgentMiddleware"]:
    """What runs each of the agent's tool calls, outermost first."""
    stack: list[AgentMiddleware] = [
        PermissionMiddleware(config.name, bound=bound, open_calls=open_calls)
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
) -> list["AgentMiddleware"]:
    """What shapes each of the agent's model requests, outermost first."""
    if resolved is None or settings is None:
        return []
    stack: list[AgentMiddleware] = [
        ForeignReasoningMiddleware(resolved.record.kind, agent=resolved.agent),
        build_summarization(model, resolved, settings),
        # Inside the summarization: it summarizes on the overflows it raises.
        ContextOverflowMiddleware(),
    ]
    if resolved.record.cache is CacheMechanism.ANTHROPIC and settings.prompt_caching_enabled:
        # The system prompt is a plain string: the middleware marks it.
        stack.append(
            AnthropicPromptCachingMiddleware(ttl="5m", unsupported_model_behavior="ignore")
        )
    return stack


def _delegation(subagents: list[CompiledSubAgent]) -> SubAgentMiddleware:
    """The coordinator's ``task`` tool, bound to the framework's declaration.

    The framework built this object, so it binds it: the gate identifies it
    by identity, never by its name.
    """
    forks = [spec["name"] for spec in subagents if spec.get("mode") == "fork"]
    if forks:
        # Deep Agents warns that forks are beta whenever a coordinator has
        # one, and LangChain shows its beta warnings: over the UI, once.
        warnings.filterwarnings("ignore", message=_FORK_BETA, category=LangChainBetaWarning)
        logger.debug("deepagents_fork_beta", agents=forks)
    middleware = SubAgentMiddleware(
        # Only stored: compiled sub-agents never read it.
        backend=None,
        subagents=subagents,
        state_schema=AgentCliState,
    )
    for task_tool in middleware.tools:
        bind_tool_identity(task_tool, get_registry().get(TASK_TOOL))
    return middleware


def _subagent(config: AgentConfig, runnable: "CompiledStateGraph") -> CompiledSubAgent:
    """How the coordinator's ``task`` tool offers ``config``.

    The coordinator chooses by the description, so an agent without one is
    still offered, under a placeholder, and the omission is logged. An
    ``inherit`` agent is forked: it continues the coordinator's conversation.
    """
    description = config.description.strip()
    if not description:
        logger.warning("deepagents_subagent_without_description", agent=config.name)
        description = f"The {config.name} agent."
    spec = CompiledSubAgent(name=config.name, description=description, runnable=runnable)
    if config.context == "inherit":
        spec["mode"] = "fork"
    return spec
