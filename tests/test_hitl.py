"""Tests for Human-in-the-Loop module."""

import pytest


class TestHITLConfig:
    """Tests for HITL configuration."""

    def test_default_config(self):
        """Test default HITL configuration."""
        from agentic_cli.hitl import HITLConfig

        config = HITLConfig()

        assert config.checkpoint_enabled is True
        assert config.confidence_threshold == 0.75

    def test_config_with_approval_rules(self):
        """Test configuration with approval rules."""
        from agentic_cli.hitl import HITLConfig, ApprovalRule

        rules = [
            ApprovalRule(tool="shell_executor"),
            ApprovalRule(tool="write_file"),
        ]

        config = HITLConfig(approval_rules=rules)

        assert len(config.approval_rules) == 2


class TestApprovalManager:
    """Tests for simplified ApprovalManager class."""

    def test_empty_history(self):
        """Test manager starts with empty history."""
        from agentic_cli.hitl import ApprovalManager

        manager = ApprovalManager()
        assert manager.history == []

    def test_record_approval(self):
        """Test recording an approval."""
        from agentic_cli.hitl import ApprovalManager

        manager = ApprovalManager()
        result = manager.record("req-1", approved=True)

        assert result.request_id == "req-1"
        assert result.approved is True
        assert result.reason is None
        assert len(manager.history) == 1

    def test_record_rejection(self):
        """Test recording a rejection with reason."""
        from agentic_cli.hitl import ApprovalManager

        manager = ApprovalManager()
        result = manager.record("req-1", approved=False, reason="Too risky")

        assert result.approved is False
        assert result.reason == "Too risky"

    def test_history_accumulates(self):
        """Test that history accumulates multiple records."""
        from agentic_cli.hitl import ApprovalManager

        manager = ApprovalManager()
        manager.record("req-1", approved=True)
        manager.record("req-2", approved=False, reason="No")
        manager.record("req-3", approved=True)

        assert len(manager.history) == 3


class TestCheckpointManager:
    """Tests for simplified CheckpointManager class."""

    def test_empty_history(self):
        """Test manager starts with empty history."""
        from agentic_cli.hitl import CheckpointManager

        manager = CheckpointManager()
        assert manager.history == []

    def test_record_continue(self):
        """Test recording a continue action."""
        from agentic_cli.hitl import CheckpointManager

        manager = CheckpointManager()
        result = manager.record("cp-1", action="continue")

        assert result.checkpoint_id == "cp-1"
        assert result.action == "continue"
        assert result.edited_content is None
        assert result.feedback is None

    def test_record_edit(self):
        """Test recording an edit action with content."""
        from agentic_cli.hitl import CheckpointManager

        manager = CheckpointManager()
        result = manager.record(
            "cp-1",
            action="edit",
            edited_content="Modified content",
        )

        assert result.action == "edit"
        assert result.edited_content == "Modified content"

    def test_record_abort(self):
        """Test recording an abort action with feedback."""
        from agentic_cli.hitl import CheckpointManager

        manager = CheckpointManager()
        result = manager.record(
            "cp-1",
            action="abort",
            feedback="Wrong approach",
        )

        assert result.action == "abort"
        assert result.feedback == "Wrong approach"

    def test_history_accumulates(self):
        """Test that history accumulates multiple records."""
        from agentic_cli.hitl import CheckpointManager

        manager = CheckpointManager()
        manager.record("cp-1", action="continue")
        manager.record("cp-2", action="edit", edited_content="new")
        manager.record("cp-3", action="abort", feedback="stop")

        assert len(manager.history) == 3


class TestHITLImports:
    """Tests for HITL module exports."""

    def test_import_all(self):
        """Test all expected exports are available."""
        from agentic_cli.hitl import (
            HITLConfig,
            ApprovalRule,
            ApprovalManager,
            ApprovalRequest,
            ApprovalResult,
            CheckpointManager,
            CheckpointResult,
        )

        assert HITLConfig is not None
        assert ApprovalRule is not None
        assert ApprovalManager is not None
        assert ApprovalRequest is not None
        assert ApprovalResult is not None
        assert CheckpointManager is not None
        assert CheckpointResult is not None
