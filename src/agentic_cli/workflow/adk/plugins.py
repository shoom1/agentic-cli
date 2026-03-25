"""ADK Plugins for agentic-cli.

Provides framework-level cross-cutting concerns as ADK Plugins:
- ConfirmationPlugin: HITL confirmation for DANGEROUS tools
"""

from __future__ import annotations

import uuid
from typing import Any, Optional, TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin

from agentic_cli.tools.registry import get_registry, PermissionLevel
from agentic_cli.workflow.context import get_context_workflow
from agentic_cli.workflow.events import UserInputRequest, InputType
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext

logger = Loggers.workflow()

# Cache of tool names -> is_dangerous, built lazily
_dangerous_cache: dict[str, bool] | None = None


def is_dangerous(tool_name: str) -> bool:
    """Check if a tool is registered as DANGEROUS permission level."""
    global _dangerous_cache
    if _dangerous_cache is None:
        registry = get_registry()
        _dangerous_cache = {
            defn.name: defn.permission_level == PermissionLevel.DANGEROUS
            for defn in registry.list_tools()
        }
    return _dangerous_cache.get(tool_name, False)


class ConfirmationPlugin(BasePlugin):
    """ADK Plugin that requires user confirmation for DANGEROUS tools.

    Uses the workflow manager's request_user_input callback to prompt
    the user before executing any tool with PermissionLevel.DANGEROUS.

    Replaces the old _wrap_dangerous decorator pattern with a single
    framework-level hook that applies to all agents globally.
    """

    def __init__(self) -> None:
        super().__init__(name="confirmation")

    async def before_tool_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext",
    ) -> Optional[dict]:
        """Intercept DANGEROUS tool calls and request user confirmation."""
        if not is_dangerous(tool.name):
            return None

        workflow = get_context_workflow()
        if workflow is None:
            logger.warning("confirmation_plugin.no_workflow", tool=tool.name)
            return None

        arg_summary = ", ".join(
            f"{k}={repr(v)[:50]}" for k, v in list(tool_args.items())[:3]
        )
        prompt = (
            f"Tool requires approval: {tool.name}({arg_summary})\n\n"
            f"Allow this operation? (yes/no)"
        )

        request = UserInputRequest(
            request_id=str(uuid.uuid4())[:8],
            tool_name=tool.name,
            prompt=prompt,
            input_type=InputType.CONFIRM,
            default="no",
        )

        try:
            response = await workflow.request_user_input(request)
        except RuntimeError:
            logger.warning("confirmation_plugin.no_callback", tool=tool.name)
            return None

        approved = response.strip().lower() in ("yes", "y", "approve", "true")

        if approved:
            logger.debug("confirmation_plugin.approved", tool=tool.name)
            return None

        logger.info("confirmation_plugin.denied", tool=tool.name)
        return {
            "success": False,
            "error": f"User denied approval for {tool.name}",
        }
