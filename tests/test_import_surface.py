"""The framework-facing contracts are importable from their natural packages.

These types are what an embedding application programs against — a turn's
outcome, a controller's state, a conversation's identity, a job's delivery
state, and the graph error it must handle at startup. Each was reachable only
from a private module path.
"""

from __future__ import annotations

import pytest


class TestTopLevelExports:
    @pytest.mark.parametrize(
        "name",
        ["SessionRef", "AgentGraphError", "TurnResult", "TurnStatus", "WorkflowState"],
    )
    def test_exported_from_package_root(self, name: str):
        import agentic_cli

        assert hasattr(agentic_cli, name), f"agentic_cli.{name} is not importable"
        assert name in agentic_cli.__all__, f"{name} is missing from __all__"

    def test_root_exports_are_the_defining_objects(self):
        """No shadow copies: the export is the class the framework uses."""
        import agentic_cli
        from agentic_cli.cli.message_processor import TurnResult, TurnStatus
        from agentic_cli.cli.workflow_controller import WorkflowState
        from agentic_cli.workflow.config import AgentGraphError
        from agentic_cli.workflow.sessions import SessionRef

        assert agentic_cli.SessionRef is SessionRef
        assert agentic_cli.AgentGraphError is AgentGraphError
        assert agentic_cli.TurnResult is TurnResult
        assert agentic_cli.TurnStatus is TurnStatus
        assert agentic_cli.WorkflowState is WorkflowState


class TestToolsExports:
    """``declare_tool`` is how an application declares a tool it implements
    per-backend; it sits next to ``register_tool`` in the tools package."""

    def test_declare_tool_is_exported(self):
        from agentic_cli import tools

        assert hasattr(tools, "declare_tool")
        assert "declare_tool" in tools.__all__

    def test_declare_tool_is_the_defining_object(self):
        from agentic_cli import tools
        from agentic_cli.tools.registry import declare_tool

        assert tools.declare_tool is declare_tool

    def test_minimal_usage(self):
        """Declare a contract, register a backend variant against it."""
        from agentic_cli.tools import ToolCategory, declare_tool, register_tool
        from agentic_cli.tools.registry import ToolRegistry
        from agentic_cli.workflow.permissions import EXEMPT

        registry = ToolRegistry()
        declare_tool(
            "doc_probe_tool",
            description="A tool each backend implements natively.",
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
            registry=registry,
        )

        def _native(content: str) -> dict:
            """Backend-native implementation."""
            return {"success": True}

        returned = registry.register(
            _native,
            variant_of="doc_probe_tool",
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
        )

        assert returned.__name__ == "doc_probe_tool"
        assert registry.identify(_native) is registry.get("doc_probe_tool")
        assert register_tool is not None  # exported alongside


class TestJobsExports:
    def test_resume_lifecycle_types_are_exported(self):
        from agentic_cli.tools import jobs

        assert hasattr(jobs, "ResumeState")
        assert hasattr(jobs, "ResumeStateError")
        assert "ResumeState" in jobs.__all__
        assert "ResumeStateError" in jobs.__all__

    def test_resume_state_values(self):
        from agentic_cli.tools.jobs import ResumeState

        assert [s.value for s in ResumeState] == [
            "pending",
            "resuming",
            "delivered",
            "failed",
        ]


class TestSessionRefShape:
    def test_is_a_frozen_triple(self):
        from agentic_cli import SessionRef

        ref = SessionRef(app_name="app", user_id="u", session_id="s")
        assert (ref.app_name, ref.user_id, ref.session_id) == ("app", "u", "s")
        with pytest.raises(Exception):
            ref.user_id = "other"  # frozen
