"""Human-in-the-Loop module for agentic CLI applications.

Provides approval gates and review checkpoints for human oversight
of agent actions. Tools block via the workflow manager's user-input
pattern until the user responds.
"""

from agentic_cli.hitl.config import HITLConfig, ApprovalRule
from agentic_cli.hitl.approval import ApprovalManager, ApprovalRequest, ApprovalResult
from agentic_cli.hitl.checkpoints import (
    CheckpointManager,
    CheckpointResult,
)

__all__ = [
    # Config
    "HITLConfig",
    "ApprovalRule",
    # Approval
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalResult",
    # Checkpoints
    "CheckpointManager",
    "CheckpointResult",
]
