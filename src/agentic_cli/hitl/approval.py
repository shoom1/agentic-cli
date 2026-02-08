"""Approval management for critical agent actions.

Simplified: the approval tool now blocks via workflow manager's
user-input-request pattern. This module provides the backing
manager that the workflow manager auto-creates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentic_cli.hitl.config import HITLConfig


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
