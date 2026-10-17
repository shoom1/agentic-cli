"""The framework-facing contracts are importable from their natural packages.

These types are what an embedding application programs against — a turn's
outcome, a controller's state, a conversation's identity, a job's delivery
state, and the graph error it must handle at startup. Each was reachable only
from a private module path.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import agentic_cli


def _run_fresh(code: str) -> str:
    """Run ``code`` in a fresh interpreter, which has imported nothing yet."""
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return run.stdout.strip()


class TestLazyPackageRoot:
    """``agentic_cli`` loads what it exports on first use, so importing one
    part of the framework (``agentic_cli.memory``) does not load the CLI and
    the workflow with it."""

    def test_importing_the_root_loads_nothing_else(self):
        loaded = _run_fresh(
            "import sys, agentic_cli; "
            "print(sorted(m for m in sys.modules if m.startswith('agentic_cli')))"
        )
        assert loaded == "['agentic_cli']"

    @pytest.mark.parametrize("name", agentic_cli.__all__)
    def test_every_exported_name_resolves(self, name: str):
        if name == "LangGraphWorkflowManager":
            pytest.importorskip("langgraph")
        assert getattr(agentic_cli, name) is not None

    def test_subpackages_stay_reachable_as_attributes(self):
        """``import agentic_cli`` used to load these, so code may reach them
        through the root without importing them."""
        found = _run_fresh(
            "import agentic_cli; "
            "print(agentic_cli.cli.app.BaseCLIApp.__name__, "
            "agentic_cli.workflow.factory.__name__, agentic_cli.config.get_settings.__name__)"
        )
        assert found == "BaseCLIApp agentic_cli.workflow.factory get_settings"

    def test_an_unknown_name_is_an_attribute_error(self):
        with pytest.raises(AttributeError, match="no_such_name"):
            agentic_cli.no_such_name  # noqa: B018


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
