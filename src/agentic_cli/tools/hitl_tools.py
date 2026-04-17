"""Human-in-the-loop tools for agentic workflows.

Provides ApprovalManager for audit history of HITL confirmation decisions.
Actual confirmation is handled at the framework level by ADK ConfirmationPlugin
and LangGraph's _wrap_for_confirmation wrapper.

Example:
    from agentic_cli.tools.hitl_tools import ApprovalManager, ApprovalResult

    manager = ApprovalManager()
    result = manager.record("req-1", approved=True)
"""

from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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

    The actual blocking is handled by framework-level confirmation hooks
    (ADK ConfirmationPlugin / LangGraph _wrap_for_confirmation). This manager
    records decisions for audit/history purposes.

    Example:
        manager = ApprovalManager()
        result = manager.record(request_id, approved=True)
    """

    def __init__(self) -> None:
        """Initialize approval manager."""
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
