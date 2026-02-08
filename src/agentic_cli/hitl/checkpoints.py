"""Checkpoint management for review points in workflows.

Simplified: the checkpoint tool now blocks via workflow manager's
user-input-request pattern. This module provides the backing
manager that the workflow manager auto-creates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CheckpointResult:
    """Result of a checkpoint review."""

    checkpoint_id: str
    action: str  # "continue", "edit", "abort"
    edited_content: str | None = None
    feedback: str | None = None
    resolved_at: datetime = field(default_factory=datetime.now)


class CheckpointManager:
    """Manages checkpoint review records.

    The actual blocking is handled by the tool via workflow manager's
    user-input-request pattern. This manager records results for
    audit/history purposes.

    Example:
        manager = CheckpointManager()
        result = manager.record(checkpoint_id, action="continue")
    """

    def __init__(self):
        """Initialize checkpoint manager."""
        self._history: list[CheckpointResult] = []

    def record(
        self,
        checkpoint_id: str,
        action: str,
        edited_content: str | None = None,
        feedback: str | None = None,
    ) -> CheckpointResult:
        """Record a checkpoint review result.

        Args:
            checkpoint_id: The checkpoint ID.
            action: User action ("continue", "edit", "abort").
            edited_content: Edited content if action is "edit".
            feedback: User feedback.

        Returns:
            CheckpointResult with the recorded decision.
        """
        result = CheckpointResult(
            checkpoint_id=checkpoint_id,
            action=action,
            edited_content=edited_content,
            feedback=feedback,
        )
        self._history.append(result)
        return result

    @property
    def history(self) -> list[CheckpointResult]:
        """Get all recorded checkpoint results."""
        return list(self._history)
