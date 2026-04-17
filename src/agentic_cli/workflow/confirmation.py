"""Backend-neutral HITL confirmation for DANGEROUS tools.

Both the ADK ConfirmationPlugin and the LangGraph tool wrapper call into
this module to decide whether a tool call needs user approval and, if so,
to request it through the active workflow manager. Keeping these helpers
here (rather than under workflow/adk/) prevents LangGraph from having to
import ADK internals.
"""

from __future__ import annotations

import uuid
from typing import Any

from agentic_cli.tools.registry import PermissionLevel, get_registry
from agentic_cli.workflow.events import InputType, UserInputRequest
from agentic_cli.workflow.service_registry import WORKFLOW, get_service

_APPROVED_RESPONSES = ("yes", "y", "approve", "true")


def is_dangerous(tool_name: str) -> bool:
    """Return True if the named tool is registered as DANGEROUS."""
    defn = get_registry().get(tool_name)
    if defn is None:
        return False
    return defn.permission_level == PermissionLevel.DANGEROUS


async def request_tool_confirmation(
    tool_name: str, tool_args: dict[str, Any]
) -> bool | None:
    """Prompt the user for confirmation of a dangerous tool call.

    Returns:
        True if approved, False if denied, None if no workflow/callback
        is available to prompt the user.
    """
    workflow = get_service(WORKFLOW)
    if workflow is None:
        return None

    arg_summary = ", ".join(
        f"{k}={repr(v)[:50]}" for k, v in list(tool_args.items())[:3]
    )
    request = UserInputRequest(
        request_id=str(uuid.uuid4())[:8],
        tool_name=tool_name,
        prompt=(
            f"Tool requires approval: {tool_name}({arg_summary})\n\n"
            f"Allow this operation? (yes/no)"
        ),
        input_type=InputType.CONFIRM,
        default="no",
    )

    try:
        response = await workflow.request_user_input(request)
    except RuntimeError:
        return None

    return response.strip().lower() in _APPROVED_RESPONSES
