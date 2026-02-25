"""Human-in-the-loop tools for agentic workflows.

One blocking tool (request_approval) plus backing dataclasses and
ApprovalManager for audit history. Uses the workflow manager's
user-input-request pattern to block until the user responds.

Example:
    from agentic_cli.tools import hitl_tools

    # In agent config
    AgentConfig(
        tools=[hitl_tools.request_approval],
    )
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentic_cli.tools import requires, require_context
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.context import get_context_workflow
from agentic_cli.workflow.events import UserInputRequest, InputType


# ---------------------------------------------------------------------------
# Config & data classes
# ---------------------------------------------------------------------------


@dataclass
class ApprovalRule:
    """Rule defining when approval is required.

    Attributes:
        tool: Tool name (e.g., "shell_executor")
        operations: Specific operations requiring approval (None = all)
        auto_approve_patterns: Patterns that skip approval (glob-style)
    """

    tool: str
    operations: list[str] | None = None
    auto_approve_patterns: list[str] = field(default_factory=list)

    def matches_tool(self, tool_name: str) -> bool:
        """Check if rule applies to a tool."""
        return self.tool == tool_name

    def matches_operation(self, operation: str) -> bool:
        """Check if rule applies to an operation."""
        if self.operations is None:
            return True  # All operations
        return operation in self.operations


@dataclass
class HITLConfig:
    """Configuration for Human-in-the-Loop features.

    Attributes:
        approval_rules: Rules defining when approval is required
        feedback_enabled: Enable inline feedback
        confidence_threshold: Threshold below which to ask for help
        confidence_visible: Show confidence scores to user
    """

    approval_rules: list[ApprovalRule] = field(default_factory=list)
    feedback_enabled: bool = True
    confidence_threshold: float = 0.75
    confidence_visible: bool = True


@dataclass
class ApprovalRequest:
    """Record of an approval request."""

    id: str
    action: str
    details: str
    risk_level: str = "medium"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ApprovalResult:
    """Result of an approval decision."""

    request_id: str
    approved: bool
    reason: str | None = None
    decided_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# ApprovalManager
# ---------------------------------------------------------------------------


class ApprovalManager:
    """Manages approval request records.

    The actual blocking is handled by the tool via workflow manager's
    user-input-request pattern. This manager records decisions for
    audit/history purposes.

    Example:
        manager = ApprovalManager(config)
        result = manager.record(request_id, action, approved=True)
    """

    def __init__(self, config: HITLConfig | None = None):
        """Initialize approval manager.

        Args:
            config: Optional HITL configuration.
        """
        self._config = config or HITLConfig()
        self._history: list[ApprovalResult] = []

    def record(
        self,
        request_id: str,
        approved: bool,
        reason: str | None = None,
    ) -> ApprovalResult:
        """Record an approval decision.

        Args:
            request_id: The request ID.
            approved: Whether the request was approved.
            reason: Optional reason for the decision.

        Returns:
            ApprovalResult with the recorded decision.
        """
        result = ApprovalResult(
            request_id=request_id,
            approved=approved,
            reason=reason,
        )
        self._history.append(result)
        return result

    @property
    def history(self) -> list[ApprovalResult]:
        """Get all recorded approval decisions."""
        return list(self._history)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@register_tool(
    category=ToolCategory.INTERACTION,
    permission_level=PermissionLevel.SAFE,
    description="Request user approval before a risky or consequential action. Blocks execution until the user approves or rejects. Use this for destructive operations, external API calls, or anything that can't be easily undone.",
)
@requires("approval_manager")
@require_context("Workflow manager", get_context_workflow)
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
