"""ADK plugin: a tool result too big for the context is cut.

Every tool result passes ``workflow/tool_output.py::cap_tool_result`` in the
``after_tool_callback``, whatever the tool object (a function tool, an MCP or
skill tool): a result over ``tool_result_max_chars`` reaches the model cut,
with a note pointing at its full output in ``./.<app_name>/tool-output/``.
Deep Agents' cap does the same
(``workflow/deepagents/middleware.py::ToolResultCapMiddleware``).

ADK stops at the first plugin whose ``after_tool_callback`` returns a value,
so the manager registers this one after ``TaskProgressPlugin``, which reads
the task list after every call and returns nothing. For the same reason a
native ADK YAML app gets none: its agents' own ``after_tool_callback``s would
be skipped for every cut result.

ADK's Claude and OpenAI Responses adapters send a model only part of a
result: its ``content`` when that is a non-empty string or list (as
``read_file`` and MCP tools return), and otherwise, on OpenAI Responses, its
``result`` when that is set (a list result, ``job_result``, the fallback).
So the note also goes where they read it (:func:`_note_where_sent`):
otherwise the model would read the cut text without learning where the rest
is. Gemini and the chat adapters send the whole result, ``truncated``
included.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.adk.plugins.base_plugin import BasePlugin

from agentic_cli.logging import Loggers
from agentic_cli.paths import resolve_path
from agentic_cli.tools.registry import get_registry
from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin
from agentic_cli.workflow.tool_output import cap_tool_result, is_saved_output

if TYPE_CHECKING:
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext

logger = Loggers.workflow()


class ToolResultCapPlugin(BasePlugin):
    """Cut every tool result over ``max_chars``, saving its full output in ``folder``.

    Args:
        max_chars: ``tool_result_max_chars``.
        folder: Where full outputs are saved (``tool_output_dir``).
    """

    def __init__(self, max_chars: int, folder: Path) -> None:
        super().__init__(name="tool_result_cap")
        self.max_chars = max_chars
        self.folder = folder

    async def after_tool_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext | None",
        result: Any,
    ) -> dict | None:
        """The cut result, or None to leave it as it is."""
        try:
            capped = cap_tool_result(
                result,
                tool_name=tool.name,
                max_chars=self.max_chars,
                folder=self.folder,
                source=self._saved_output_read(tool, tool_args),
            )
        except Exception as e:
            # The cap never raises by design; a bug in it must not end the
            # turn, which a plugin's exception does. The result goes uncut,
            # and the traceback goes to the debug log, not over the UI.
            logger.warning(
                "tool_result_cap_failed",
                tool=tool.name,
                error_type=type(e).__name__,
                error=str(e),
            )
            logger.debug("tool_result_cap_failed", tool=tool.name, exc_info=e)
            return None
        if capped is result:
            return None
        if not isinstance(capped, dict):
            capped = {"result": capped}
        return _note_where_sent(capped)

    def _saved_output_read(self, tool: "BaseTool", tool_args: dict[str, Any]) -> Path | None:
        """The saved output this call reads, when it is the framework's ``read_file``.

        The tool is identified as the permission plugin identifies it, by the
        object the registry issued, never by its name.
        """
        read_file = get_registry().get("read_file")
        if read_file is None or PermissionPlugin._resolve_definition(tool) is not read_file:
            return None
        try:
            path = resolve_path(tool_args["path"])
        except (KeyError, TypeError, ValueError):
            return None
        return path if is_saved_output(path, self.folder) else None


def _note_where_sent(capped: dict) -> dict:
    """``capped`` with its note also where ADK's adapters read a result.

    In ``content`` when that is sent alone (a non-empty string or list),
    else in ``result``: appended to a string, as the last item of a list,
    under ``truncated`` in a dict. A ``result`` of another type (a number)
    is left as it is.
    """
    note = capped.get("truncated")
    if not isinstance(note, str):
        return capped
    content = capped.get("content")
    if isinstance(content, str) and content:
        return {**capped, "content": f"{content}\n\n{note}"}
    if isinstance(content, list) and content:
        return {**capped, "content": [*content, {"type": "text", "text": note}]}
    result = capped.get("result")
    if isinstance(result, str):
        return {**capped, "result": f"{result}\n\n{note}"}
    if isinstance(result, list):
        return {**capped, "result": [*result, note]}
    if isinstance(result, dict):
        return {**capped, "result": {**result, "truncated": note}}
    return capped
