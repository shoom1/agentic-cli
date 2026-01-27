"""Human-in-the-loop tools for agentic workflows.

These tools provide approval gates and checkpoints for human oversight.
The ApprovalManager and CheckpointManager are auto-created by the workflow
manager when these tools are used.

Example:
    from agentic_cli.tools import hitl_tools

    # In agent config
    AgentConfig(
        tools=[hitl_tools.create_checkpoint, hitl_tools.request_approval, ...],
    )
"""

from typing import Any

from agentic_cli.tools import requires
from agentic_cli.workflow.context import (
    get_context_approval_manager,
    get_context_checkpoint_manager,
)


@requires("checkpoint_manager")
def create_checkpoint(
    name: str,
    content: str,
    content_type: str = "markdown",
    allow_edit: bool = True,
    allow_regenerate: bool = True,
) -> dict[str, Any]:
    """Create a checkpoint for user review before continuing.

    Use this when you have draft content that should be reviewed
    before proceeding (e.g., research summaries, code drafts, plans).

    Args:
        name: Name of the checkpoint (e.g., "draft_summary").
        content: Content to review.
        content_type: Type of content ("markdown", "text", "code", "json").
        allow_edit: Whether user can edit the content.
        allow_regenerate: Whether user can request regeneration.

    Returns:
        A dict with the checkpoint ID.
    """
    manager = get_context_checkpoint_manager()
    if manager is None:
        return {"success": False, "error": "Checkpoint manager not available"}

    checkpoint = manager.create_checkpoint(
        name=name,
        content=content,
        content_type=content_type,
        allow_edit=allow_edit,
        allow_regenerate=allow_regenerate,
    )

    return {
        "success": True,
        "checkpoint_id": checkpoint.id,
        "name": name,
        "content_type": content_type,
        "message": f"Created checkpoint '{name}' for review",
    }


@requires("checkpoint_manager")
def get_checkpoint_result(checkpoint_id: str) -> dict[str, Any]:
    """Get the result of a checkpoint review.

    Args:
        checkpoint_id: The checkpoint ID to check.

    Returns:
        A dict with the review result if resolved, or pending status.
    """
    manager = get_context_checkpoint_manager()
    if manager is None:
        return {"success": False, "error": "Checkpoint manager not available"}

    if not manager.is_resolved(checkpoint_id):
        return {
            "success": True,
            "resolved": False,
            "message": "Checkpoint is pending user review",
        }

    result = manager.get_result(checkpoint_id)
    if result is None:
        return {"success": False, "error": f"Checkpoint '{checkpoint_id}' not found"}

    return {
        "success": True,
        "resolved": True,
        "action": result.action.value,
        "edited_content": result.edited_content,
        "feedback": result.feedback,
    }


@requires("approval_manager")
def request_approval(
    action: str,
    details: str,
    tool_name: str | None = None,
    risk_level: str = "medium",
) -> dict[str, Any]:
    """Request explicit user approval before proceeding.

    Use this for actions that could have significant consequences
    (file writes, shell commands, external API calls, etc.).

    Args:
        action: Short description of the action (e.g., "write to config.json").
        details: Detailed description of what will happen.
        tool_name: Optional tool name for categorization.
        risk_level: Risk level ("low", "medium", "high").

    Returns:
        A dict with the approval request ID and status.
    """
    manager = get_context_approval_manager()
    if manager is None:
        return {"success": False, "error": "Approval manager not available"}

    request = manager.request_approval(
        tool=tool_name or "agent_action",
        operation="execute",
        description=action,
        details={"description": details},
        risk_level=risk_level,
    )

    return {
        "success": True,
        "request_id": request.id,
        "action": action,
        "risk_level": risk_level,
        "status": "pending",
        "message": f"Approval requested for: {action}",
    }


@requires("approval_manager")
def check_approval(request_id: str) -> dict[str, Any]:
    """Check the status of an approval request.

    Args:
        request_id: The approval request ID to check.

    Returns:
        A dict with the approval status.
    """
    manager = get_context_approval_manager()
    if manager is None:
        return {"success": False, "error": "Approval manager not available"}

    # Check if still pending
    pending = manager.get_pending_request(request_id)
    if pending is not None:
        return {
            "success": True,
            "status": "pending",
            "message": "Awaiting user decision",
        }

    # Check result
    result = manager.get_result(request_id)
    if result is None:
        return {"success": False, "error": f"Request '{request_id}' not found"}

    return {
        "success": True,
        "status": result.status.value,
        "approved": result.status.value == "approved",
        "reason": result.reason,
        "modified_args": result.modified_args,
    }


@requires("approval_manager")
def check_requires_approval(
    tool: str,
    operation: str,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check if a specific action requires approval.

    This allows agents to proactively check before taking action.

    Args:
        tool: Tool name.
        operation: Operation being performed.
        args: Arguments to the operation.

    Returns:
        A dict indicating whether approval is required.
    """
    manager = get_context_approval_manager()
    if manager is None:
        return {"success": False, "error": "Approval manager not available"}

    requires = manager.requires_approval(tool, operation, args or {})

    return {
        "success": True,
        "requires_approval": requires,
        "tool": tool,
        "operation": operation,
    }
