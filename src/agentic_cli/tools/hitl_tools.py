"""Human-in-the-loop tools for agentic workflows.

Two blocking tools: request_approval and create_checkpoint.
Both use the workflow manager's user-input-request pattern to block
until the user responds.

Example:
    from agentic_cli.tools import hitl_tools

    # In agent config
    AgentConfig(
        tools=[hitl_tools.request_approval, hitl_tools.create_checkpoint],
    )
"""

import json
import uuid
from typing import Any

from agentic_cli.tools import requires
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.config import get_context_workflow
from agentic_cli.workflow.events import UserInputRequest, InputType


@register_tool(
    category=ToolCategory.INTERACTION,
    permission_level=PermissionLevel.SAFE,
    description="Request user approval before a risky or consequential action. Blocks execution until the user approves or rejects. Use this for destructive operations, external API calls, or anything that can't be easily undone.",
)
@requires("approval_manager")
async def request_approval(
    action: str,
    details: str,
    risk_level: str = "medium",
) -> dict[str, Any]:
    """Request user approval before proceeding.

    Blocks until the user approves or rejects. Use this for actions
    that could have significant consequences.

    Args:
        action: Short description of the action (e.g., "Delete old files").
        details: Detailed description of what will happen.
        risk_level: Risk level ("low", "medium", "high").

    Returns:
        A dict with {"approved": bool, "reason": str | None}.
    """
    workflow = get_context_workflow()
    if workflow is None:
        return {"success": False, "error": "Workflow manager not available"}

    request_id = str(uuid.uuid4())[:8]
    prompt = f"[{risk_level.upper()}] Approval requested: {action}\n\nDetails: {details}\n\nApprove? (yes/no)"

    request = UserInputRequest(
        request_id=request_id,
        tool_name="request_approval",
        prompt=prompt,
        input_type=InputType.CONFIRM,
        default="no",
    )

    response = await workflow.request_user_input(request)

    approved = response.strip().lower() in ("yes", "y", "approve", "true")

    return {
        "success": True,
        "approved": approved,
        "reason": None if approved else response,
    }


@register_tool(
    category=ToolCategory.INTERACTION,
    permission_level=PermissionLevel.SAFE,
    description="Create a review checkpoint presenting draft content to the user. Blocks until the user reviews and responds (continue/edit/abort). Use this for draft summaries, reports, or code that should be reviewed before finalizing.",
)
@requires("checkpoint_manager")
async def create_checkpoint(
    name: str,
    content: str,
    content_type: str = "markdown",
    allow_edit: bool = True,
) -> dict[str, Any]:
    """Create a review checkpoint for the user.

    Blocks until the user reviews and responds. Use this for draft
    content that should be reviewed before proceeding.

    Args:
        name: Name of the checkpoint (e.g., "draft_summary").
        content: Content to review.
        content_type: Type of content ("markdown", "text", "code", "json").
        allow_edit: Whether user can edit the content.

    Returns:
        A dict with {"action": "continue"|"edit"|"abort",
                      "edited_content": str | None,
                      "feedback": str | None}.
    """
    workflow = get_context_workflow()
    if workflow is None:
        return {"success": False, "error": "Workflow manager not available"}

    request_id = str(uuid.uuid4())[:8]

    edit_hint = " You can edit the content or provide feedback." if allow_edit else ""
    prompt = (
        f"Checkpoint: {name} ({content_type})\n\n"
        f"{content}\n\n"
        f"Actions: continue / edit / abort.{edit_hint}"
    )

    request = UserInputRequest(
        request_id=request_id,
        tool_name="create_checkpoint",
        prompt=prompt,
        input_type=InputType.CHOICE,
        choices=["continue", "edit", "abort"],
        default="continue",
    )

    response = await workflow.request_user_input(request)

    # Parse the response — it may be a simple action or JSON with edit data
    action = "continue"
    edited_content = None
    feedback = None

    try:
        parsed = json.loads(response)
        action = parsed.get("action", "continue")
        edited_content = parsed.get("edited_content")
        feedback = parsed.get("feedback")
    except (json.JSONDecodeError, TypeError):
        # Simple string response — treat as action
        response_lower = response.strip().lower()
        if response_lower in ("continue", "edit", "abort"):
            action = response_lower
        elif response_lower in ("yes", "y", "ok"):
            action = "continue"
        elif response_lower in ("no", "n", "stop"):
            action = "abort"
        else:
            # Treat as feedback
            action = "continue"
            feedback = response.strip()

    return {
        "success": True,
        "action": action,
        "edited_content": edited_content,
        "feedback": feedback,
    }
