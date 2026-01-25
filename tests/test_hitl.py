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
            ApprovalRule(tool="file_manager", operations=["delete", "write"]),
        ]

        config = HITLConfig(approval_rules=rules)

        assert len(config.approval_rules) == 2


class TestApprovalManager:
    """Tests for ApprovalManager class."""

    def test_requires_approval_for_configured_tool(self):
        """Test that configured tools require approval."""
        from agentic_cli.hitl import HITLConfig, ApprovalRule, ApprovalManager

        config = HITLConfig(
            approval_rules=[ApprovalRule(tool="shell_executor")]
        )
        manager = ApprovalManager(config)

        assert manager.requires_approval("shell_executor", "run", {}) is True
        assert manager.requires_approval("search_kb", "search", {}) is False

    def test_requires_approval_for_specific_operation(self):
        """Test that specific operations require approval."""
        from agentic_cli.hitl import HITLConfig, ApprovalRule, ApprovalManager

        config = HITLConfig(
            approval_rules=[
                ApprovalRule(tool="file_manager", operations=["delete", "write"])
            ]
        )
        manager = ApprovalManager(config)

        assert manager.requires_approval("file_manager", "delete", {}) is True
        assert manager.requires_approval("file_manager", "read", {}) is False

    def test_auto_approve_patterns(self):
        """Test auto-approval patterns."""
        from agentic_cli.hitl import HITLConfig, ApprovalRule, ApprovalManager

        config = HITLConfig(
            approval_rules=[
                ApprovalRule(
                    tool="shell_executor",
                    auto_approve_patterns=["ls *", "pwd", "cat *"],
                )
            ]
        )
        manager = ApprovalManager(config)

        # Should be auto-approved
        assert manager.requires_approval("shell_executor", "run", {"command": "ls -la"}) is False
        assert manager.requires_approval("shell_executor", "run", {"command": "pwd"}) is False

        # Should require approval
        assert manager.requires_approval("shell_executor", "run", {"command": "rm file.txt"}) is True

    def test_create_and_process_approval_request(self):
        """Test creating and processing approval requests."""
        from agentic_cli.hitl import HITLConfig, ApprovalRule, ApprovalManager

        config = HITLConfig(
            approval_rules=[ApprovalRule(tool="shell_executor")]
        )
        manager = ApprovalManager(config)

        # Create request
        request = manager.request_approval(
            tool="shell_executor",
            operation="run",
            description="Execute shell command",
            details={"command": "pytest tests/"},
        )

        assert request.id is not None
        assert request.tool == "shell_executor"
        assert manager.get_pending_request(request.id) is not None

        # Approve it
        manager.approve(request.id)

        assert manager.is_approved(request.id) is True
        assert manager.get_pending_request(request.id) is None

    def test_reject_request(self):
        """Test rejecting an approval request."""
        from agentic_cli.hitl import HITLConfig, ApprovalRule, ApprovalManager

        config = HITLConfig(
            approval_rules=[ApprovalRule(tool="shell_executor")]
        )
        manager = ApprovalManager(config)

        request = manager.request_approval(
            tool="shell_executor",
            operation="run",
            description="Run command",
            details={},
        )

        manager.reject(request.id, reason="Too risky")

        assert manager.is_rejected(request.id) is True

    def test_modify_and_approve(self):
        """Test modifying request args and approving."""
        from agentic_cli.hitl import HITLConfig, ApprovalRule, ApprovalManager

        config = HITLConfig(
            approval_rules=[ApprovalRule(tool="shell_executor")]
        )
        manager = ApprovalManager(config)

        request = manager.request_approval(
            tool="shell_executor",
            operation="run",
            description="Run command",
            details={"command": "rm -rf /tmp/test"},
        )

        # Modify the command before approving
        manager.modify_and_approve(
            request.id,
            modified_args={"command": "rm /tmp/test/specific_file.txt"}
        )

        result = manager.get_result(request.id)
        assert result.modified_args["command"] == "rm /tmp/test/specific_file.txt"


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        from agentic_cli.hitl import CheckpointManager, Checkpoint

        manager = CheckpointManager()

        checkpoint = manager.create_checkpoint(
            name="draft_review",
            content="## Draft Summary\n\nKey findings...",
            content_type="markdown",
        )

        assert checkpoint.name == "draft_review"
        assert checkpoint.content_type == "markdown"

    def test_checkpoint_allows_edit(self):
        """Test checkpoint with edit allowed."""
        from agentic_cli.hitl import CheckpointManager

        manager = CheckpointManager()

        checkpoint = manager.create_checkpoint(
            name="review",
            content="Content to review",
            allow_edit=True,
            allow_regenerate=False,
        )

        assert checkpoint.allow_edit is True
        assert checkpoint.allow_regenerate is False

    @pytest.mark.asyncio
    async def test_checkpoint_continue_action(self):
        """Test continuing from a checkpoint."""
        from agentic_cli.hitl import CheckpointManager, CheckpointAction

        manager = CheckpointManager()

        checkpoint = manager.create_checkpoint(
            name="review",
            content="Content",
        )

        # Simulate user choosing to continue
        result = manager.resolve_checkpoint(
            checkpoint.id,
            action=CheckpointAction.CONTINUE,
        )

        assert result.action == CheckpointAction.CONTINUE
        assert result.edited_content is None

    @pytest.mark.asyncio
    async def test_checkpoint_edit_action(self):
        """Test editing at a checkpoint."""
        from agentic_cli.hitl import CheckpointManager, CheckpointAction

        manager = CheckpointManager()

        checkpoint = manager.create_checkpoint(
            name="review",
            content="Original content",
            allow_edit=True,
        )

        result = manager.resolve_checkpoint(
            checkpoint.id,
            action=CheckpointAction.EDIT,
            edited_content="Modified content",
        )

        assert result.action == CheckpointAction.EDIT
        assert result.edited_content == "Modified content"

    def test_checkpoint_abort_action(self):
        """Test aborting at a checkpoint."""
        from agentic_cli.hitl import CheckpointManager, CheckpointAction

        manager = CheckpointManager()

        checkpoint = manager.create_checkpoint(
            name="review",
            content="Content",
        )

        result = manager.resolve_checkpoint(
            checkpoint.id,
            action=CheckpointAction.ABORT,
            feedback="Don't proceed with this approach",
        )

        assert result.action == CheckpointAction.ABORT
        assert result.feedback == "Don't proceed with this approach"
