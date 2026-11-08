"""Outside content: what marks a conversation, and the mark each one keeps (spec §9).

A conversation is marked once a tool that brings content from outside the
user and the project (a web page, a document, a program's output, an MCP
server's answer) has been allowed to run in it. Content read stays in the
model's context after its turn, so the mark never clears: from then on a
memory write in that conversation is approved per call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.tools.registry import ToolDefinition

logger = Loggers.workflow()


def brings_outside_content(definition: "ToolDefinition | None", args: Mapping[str, Any]) -> bool:
    """Whether a call of ``definition`` with ``args`` brings outside content.

    A flag that is a function decides from the arguments; one that raises
    counts as outside content (the safe answer).
    """
    flag = getattr(definition, "outside_content", False)
    if callable(flag):
        try:
            return bool(flag(dict(args)))
        except Exception:
            logger.debug("outside_content_check_failed", exc_info=True)
            return True
    return bool(flag)
