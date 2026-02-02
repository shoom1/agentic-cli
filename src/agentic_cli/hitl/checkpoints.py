"""Checkpoint management for review points in workflows.

Checkpoints allow agents to pause for user review at key points,
enabling approval, editing, or regeneration of outputs.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CheckpointAction(Enum):
    """Actions user can take at a checkpoint."""

    CONTINUE = "continue"
    EDIT = "edit"
    REGENERATE = "regenerate"
    ABORT = "abort"


@dataclass
class Checkpoint:
    """A review checkpoint."""

    id: str
    name: str
    content: Any
    content_type: str = "markdown"  # markdown, json, text, code
    allow_edit: bool = True
    allow_regenerate: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointResult:
    """Result of a checkpoint review."""

    checkpoint_id: str
    action: CheckpointAction
    edited_content: Any = None
    feedback: str | None = None
    resolved_at: datetime = field(default_factory=datetime.now)


class CheckpointManager:
    """Manages review checkpoints in workflows.

    Provides a way for agents to pause for user review at key points.
    Users can continue, edit content, request regeneration, or abort.

    Example:
        manager = CheckpointManager()

        checkpoint = manager.create_checkpoint(
            name="draft_review",
            content="## Summary\n\nFindings...",
            content_type="markdown",
        )

        # Present to user and get action
        result = manager.resolve_checkpoint(
            checkpoint.id,
            action=CheckpointAction.EDIT,
            edited_content="## Summary\n\nRevised findings...",
        )
    """

    def __init__(self):
        """Initialize checkpoint manager."""
        self._checkpoints: dict[str, Checkpoint] = {}
        self._results: dict[str, CheckpointResult] = {}

    def create_checkpoint(
        self,
        name: str,
        content: Any,
        content_type: str = "markdown",
        allow_edit: bool = True,
        allow_regenerate: bool = True,
        **metadata,
    ) -> Checkpoint:
        """Create a review checkpoint.

        Args:
            name: Checkpoint name (e.g., "draft_review")
            content: Content to review
            content_type: Type of content (markdown, json, text, code)
            allow_edit: Whether user can edit the content
            allow_regenerate: Whether user can request regeneration
            **metadata: Additional metadata

        Returns:
            Checkpoint instance
        """
        checkpoint = Checkpoint(
            id=str(uuid.uuid4())[:8],
            name=name,
            content=content,
            content_type=content_type,
            allow_edit=allow_edit,
            allow_regenerate=allow_regenerate,
            metadata=metadata,
        )

        self._checkpoints[checkpoint.id] = checkpoint
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    def resolve_checkpoint(
        self,
        checkpoint_id: str,
        action: CheckpointAction,
        edited_content: Any = None,
        feedback: str | None = None,
    ) -> CheckpointResult:
        """Resolve a checkpoint with user action.

        Args:
            checkpoint_id: Checkpoint ID
            action: Action taken by user
            edited_content: Modified content (if action is EDIT)
            feedback: User feedback (optional)

        Returns:
            CheckpointResult with the resolution
        """
        result = CheckpointResult(
            checkpoint_id=checkpoint_id,
            action=action,
            edited_content=edited_content,
            feedback=feedback,
        )

        self._results[checkpoint_id] = result
        return result

    def get_result(self, checkpoint_id: str) -> CheckpointResult | None:
        """Get the result for a checkpoint."""
        return self._results.get(checkpoint_id)

    def is_resolved(self, checkpoint_id: str) -> bool:
        """Check if a checkpoint has been resolved."""
        return checkpoint_id in self._results

    def get_unresolved(self) -> list[Checkpoint]:
        """Get checkpoints awaiting review.

        Returns:
            List of Checkpoint objects that haven't been resolved yet.
        """
        return [
            cp for cp in self._checkpoints.values()
            if cp.id not in self._results
        ]

    def get_results(self) -> dict[str, CheckpointResult]:
        """Get all checkpoint results.

        Returns:
            Dictionary mapping checkpoint IDs to their results.
        """
        return dict(self._results)
