"""Framework tools as LangChain tools, for ``create_agent``.

``BaseWorkflowManager._build_tools`` hands back plain callables: framework
tools, their service-bound variants, the LangGraph-library state tools, and
whatever the application passes. ``create_agent`` runs ``BaseTool`` objects,
so each callable becomes a ``StructuredTool`` whose ``.func`` (sync) or
``.coroutine`` (async) *is* that callable. The permission gate unwraps exactly
that type, so it identifies the tool the registry issued, and nothing else.

A sync tool has only ``.func``: under ``ainvoke`` LangChain runs it on its
executor thread with a copy of the context, so the service registry, the
active turn and the input callback (all ContextVars) still reach it.
"""

from __future__ import annotations

import inspect
from typing import Annotated, Any, Callable

from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from langchain_core.tools.base import FILTERED_ARGS, create_schema_from_function
from pydantic import create_model

from agentic_cli.tools.registry import ToolRegistry, get_registry

#: Appended to a long-running tool's description: ADK's
#: ``LongRunningFunctionTool`` note, word for word.
LONG_RUNNING_NOTE = (
    "\n\nNOTE: This is a long-running operation. Do not call this tool again if"
    " it has already returned some intermediate or pending status."
)

#: The argument ADK injects into a tool. A tool written for ADK may declare it.
_TOOL_CONTEXT = "tool_context"


def to_langchain_tools(
    tools: list, *, registry: ToolRegistry | None = None
) -> list[BaseTool]:
    """``tools`` as LangChain tools, in order.

    Args:
        tools: Callables and ``BaseTool`` objects. A ``BaseTool`` (an MCP or
            skill tool, or one an application built) is returned unchanged.
        registry: Which registry says what a callable is (long-running or
            not). Defaults to the framework's.
    """
    # Not `registry or ...`: an empty ToolRegistry is falsy (it has __len__).
    registry = get_registry() if registry is None else registry
    return [
        tool if isinstance(tool, BaseTool) else _from_callable(tool, registry)
        for tool in tools
    ]


def _from_callable(func: Callable[..., Any], registry: ToolRegistry) -> StructuredTool:
    definition = registry.identify(func)
    name = func.__name__
    description = (
        inspect.getdoc(func) or (definition.description if definition else "") or name
    )
    if definition is not None and definition.long_running:
        description += LONG_RUNNING_NOTE
    runs = {"coroutine": func} if inspect.iscoroutinefunction(func) else {"func": func}

    if _TOOL_CONTEXT not in inspect.signature(func).parameters:
        return StructuredTool.from_function(name=name, description=description, **runs)
    return StructuredTool.from_function(
        name=name,
        description=description,
        args_schema=_schema_without_tool_context(name, func),
        infer_schema=False,
        **runs,
    )


def _schema_without_tool_context(name: str, func: Callable[..., Any]) -> type:
    """``func``'s arguments, with ``tool_context`` hidden and always ``None``.

    The field is an injected argument, so the model's schema leaves it out,
    and its default is what LangChain passes when the call does not name it.
    Its type admits only ``None``: a value the model sends anyway fails
    validation instead of reaching the tool.
    """
    schema = create_schema_from_function(
        name, func, filter_args=[*FILTERED_ARGS, _TOOL_CONTEXT]
    )
    return create_model(
        schema.__name__,
        __base__=schema,
        tool_context=(Annotated[None, InjectedToolArg], None),
    )
