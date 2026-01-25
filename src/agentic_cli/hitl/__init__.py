"""Human-in-the-Loop module for agentic CLI applications.

Provides approval gates, review checkpoints, feedback capture,
and confidence estimation for human oversight of agent actions.
"""

from agentic_cli.hitl.config import HITLConfig, ApprovalRule
from agentic_cli.hitl.approval import ApprovalManager, ApprovalRequest, ApprovalResult

__all__ = [
    "HITLConfig",
    "ApprovalRule",
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalResult",
]
