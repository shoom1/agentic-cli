"""Tests for Human-in-the-Loop module."""

import pytest


class TestApprovalManager:
    """Tests for simplified ApprovalManager class."""

    def test_empty_history(self):
        """Test manager starts with empty history."""
        from agentic_cli.tools.hitl_tools import ApprovalManager

        manager = ApprovalManager()
        assert manager.history == []

    def test_record_approval(self):
        """Test recording an approval."""
        from agentic_cli.tools.hitl_tools import ApprovalManager

        manager = ApprovalManager()
        result = manager.record("req-1", approved=True)

        assert result.request_id == "req-1"
        assert result.approved is True
        assert result.reason is None
        assert len(manager.history) == 1

    def test_record_rejection(self):
        """Test recording a rejection with reason."""
        from agentic_cli.tools.hitl_tools import ApprovalManager

        manager = ApprovalManager()
        result = manager.record("req-1", approved=False, reason="Too risky")

        assert result.approved is False
        assert result.reason == "Too risky"

    def test_history_accumulates(self):
        """Test that history accumulates multiple records."""
        from agentic_cli.tools.hitl_tools import ApprovalManager

        manager = ApprovalManager()
        manager.record("req-1", approved=True)
        manager.record("req-2", approved=False, reason="No")
        manager.record("req-3", approved=True)

        assert len(manager.history) == 3


class TestHITLImports:
    """Tests for hitl_tools module exports."""

    def test_import_core_classes(self):
        """Test core exports are available."""
        from agentic_cli.tools.hitl_tools import (
            ApprovalManager,
            ApprovalResult,
        )

        assert ApprovalManager is not None
        assert ApprovalResult is not None
