"""Approval management for critical agent actions."""

import fnmatch
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from agentic_cli.hitl.config import ApprovalRule, HITLConfig


class ApprovalStatus(Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ApprovalRequest:
    """Request for user approval."""

    id: str
    tool: str
    operation: str
    description: str
    details: dict[str, Any]
    risk_level: str = "medium"  # low, medium, high
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ApprovalResult:
    """Result of an approval decision."""

    request_id: str
    status: ApprovalStatus
    modified_args: dict[str, Any] | None = None
    reason: str | None = None
    decided_at: datetime = field(default_factory=datetime.now)


class ApprovalManager:
    """Manages approval requests for agent actions.

    Tracks which actions require approval, manages pending requests,
    and records approval decisions.

    Example:
        config = HITLConfig(approval_rules=[
            ApprovalRule(tool="shell_executor"),
        ])
        manager = ApprovalManager(config)

        if manager.requires_approval("shell_executor", "run", args):
            request = manager.request_approval(...)
            # Wait for user decision
            manager.approve(request.id)
    """

    def __init__(self, config: HITLConfig):
        """Initialize approval manager.

        Args:
            config: HITL configuration with approval rules
        """
        self._config = config
        self._pending: dict[str, ApprovalRequest] = {}
        self._results: dict[str, ApprovalResult] = {}

    def requires_approval(
        self,
        tool: str,
        operation: str,
        args: dict[str, Any],
    ) -> bool:
        """Check if an action requires approval.

        Args:
            tool: Tool name
            operation: Operation being performed
            args: Arguments to the operation

        Returns:
            True if approval is required
        """
        for rule in self._config.approval_rules:
            if not rule.matches_tool(tool):
                continue

            if not rule.matches_operation(operation):
                continue

            # Check auto-approve patterns
            if self._matches_auto_approve(rule, args):
                return False

            return True

        return False

    def _matches_auto_approve(
        self,
        rule: ApprovalRule,
        args: dict[str, Any],
    ) -> bool:
        """Check if args match auto-approve patterns."""
        if not rule.auto_approve_patterns:
            return False

        # Check command argument (common for shell executor)
        command = args.get("command", "")

        for pattern in rule.auto_approve_patterns:
            if fnmatch.fnmatch(command, pattern):
                return True

        return False

    def request_approval(
        self,
        tool: str,
        operation: str,
        description: str,
        details: dict[str, Any],
        risk_level: str = "medium",
    ) -> ApprovalRequest:
        """Create an approval request.

        Args:
            tool: Tool name
            operation: Operation being performed
            description: Human-readable description
            details: Details about the action
            risk_level: Risk level (low, medium, high)

        Returns:
            ApprovalRequest instance
        """
        request = ApprovalRequest(
            id=str(uuid.uuid4())[:8],
            tool=tool,
            operation=operation,
            description=description,
            details=details,
            risk_level=risk_level,
        )

        self._pending[request.id] = request
        return request

    def get_pending_request(self, request_id: str) -> ApprovalRequest | None:
        """Get a pending request by ID."""
        return self._pending.get(request_id)

    def approve(self, request_id: str) -> None:
        """Approve a request.

        Args:
            request_id: Request ID to approve
        """
        if request_id not in self._pending:
            return

        self._results[request_id] = ApprovalResult(
            request_id=request_id,
            status=ApprovalStatus.APPROVED,
        )

        del self._pending[request_id]

    def reject(self, request_id: str, reason: str | None = None) -> None:
        """Reject a request.

        Args:
            request_id: Request ID to reject
            reason: Optional rejection reason
        """
        if request_id not in self._pending:
            return

        self._results[request_id] = ApprovalResult(
            request_id=request_id,
            status=ApprovalStatus.REJECTED,
            reason=reason,
        )

        del self._pending[request_id]

    def modify_and_approve(
        self,
        request_id: str,
        modified_args: dict[str, Any],
    ) -> None:
        """Modify request arguments and approve.

        Args:
            request_id: Request ID
            modified_args: Modified arguments to use
        """
        if request_id not in self._pending:
            return

        self._results[request_id] = ApprovalResult(
            request_id=request_id,
            status=ApprovalStatus.APPROVED,
            modified_args=modified_args,
        )

        del self._pending[request_id]

    def is_approved(self, request_id: str) -> bool:
        """Check if request was approved."""
        result = self._results.get(request_id)
        return result is not None and result.status == ApprovalStatus.APPROVED

    def is_rejected(self, request_id: str) -> bool:
        """Check if request was rejected."""
        result = self._results.get(request_id)
        return result is not None and result.status == ApprovalStatus.REJECTED

    def get_result(self, request_id: str) -> ApprovalResult | None:
        """Get the result for a request."""
        return self._results.get(request_id)
