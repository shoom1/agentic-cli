"""The error results a failed tool call gets, on every backend.

A tool that raises, or a call to a tool name the agent does not have, costs
that one call, not the turn: the model reads ``{"success": False, "error":
...}`` and can recover. Framework tools return that shape themselves; these
helpers give it to anything that still raises (a bug, or an application's own
tool), so ADK's ``ToolErrorPlugin`` and the Deep Agents middleware answer the
model in the same words.
"""

from __future__ import annotations

from collections.abc import Iterable

from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers

logger = Loggers.workflow()

#: Longest error text a result carries (``truncate`` adds an ellipsis).
MAX_TOOL_ERROR = 500


def tool_error_result(tool_name: str, error: BaseException) -> dict:
    """The result for a tool call that raised ``error``; logs ``tool_raised``."""
    logger.warning(
        "tool_raised",
        tool=tool_name,
        error_type=type(error).__name__,
        error=truncate(str(error), MAX_TOOL_ERROR),
        exc_info=error,
    )
    return {
        "success": False,
        "error": truncate(
            f"{tool_name} failed: {type(error).__name__}: {error}", MAX_TOOL_ERROR
        ),
    }


def unknown_tool_result(tool_name: str, available: Iterable[str]) -> dict:
    """The result for a call to a tool the agent does not have.

    ADK's text for an unknown function (its first two lines), as ADK's
    ``ToolErrorPlugin`` reports it, with the agent's tools in name order.
    """
    return {
        "success": False,
        "error": truncate(
            f"{tool_name} failed: ValueError: Tool '{tool_name}' not found.\n"
            f"Available tools: {', '.join(sorted(available))}",
            MAX_TOOL_ERROR,
        ),
    }
