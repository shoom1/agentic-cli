"""Tests for sandbox execution tools.

Tests cover:
- ExecutionResult model
- SandboxManager with mock backend
- sandbox_execute / sandbox_reset tool functions
- Manager auto-detection via @requires
- JupyterLocalBackend integration (guarded by jupyter_client availability)
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentic_cli.tools.sandbox.models import ExecutionResult
from agentic_cli.tools.sandbox.backends.base import SandboxBackend
from agentic_cli.tools.sandbox.manager import SandboxManager, SandboxSession
from tests.conftest import MockContext


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class MockSandboxBackend(SandboxBackend):
    """Test backend that returns configurable results."""

    def __init__(self, result: ExecutionResult | None = None) -> None:
        self._result = result or ExecutionResult(success=True, stdout="ok\n", result="42")
        self._sessions: set[str] = set()
        self.execute_calls: list[dict] = []
        self.reset_calls: list[str] = []

    def execute(self, code, session_id, timeout_seconds=120, working_dir=None):
        self._sessions.add(session_id)
        self.execute_calls.append({
            "code": code,
            "session_id": session_id,
            "timeout_seconds": timeout_seconds,
            "working_dir": working_dir,
        })
        return self._result

    def reset_session(self, session_id):
        self._sessions.discard(session_id)
        self.reset_calls.append(session_id)

    def cleanup(self):
        self._sessions.clear()

    def has_session(self, session_id):
        return session_id in self._sessions


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

class TestExecutionResult:
    def test_defaults(self):
        r = ExecutionResult()
        assert r.success is True
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.result is None
        assert r.artifacts == []
        assert r.execution_time == 0.0
        assert r.error == ""

    def test_full_result(self):
        r = ExecutionResult(
            success=False,
            stdout="hello",
            stderr="warning",
            result="3.14",
            artifacts=["/tmp/plot.png"],
            execution_time=1.5,
            error="NameError: x",
        )
        assert r.success is False
        assert r.stdout == "hello"
        assert r.result == "3.14"
        assert r.artifacts == ["/tmp/plot.png"]
        assert r.error == "NameError: x"


# ---------------------------------------------------------------------------
# SandboxManager
# ---------------------------------------------------------------------------

class TestSandboxManager:
    def test_execute_creates_session(self, tmp_path):
        with MockContext() as ctx:
            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)

            result = mgr.execute("x = 1", session_id="s1")
            assert result.success is True
            assert len(backend.execute_calls) == 1
            assert backend.execute_calls[0]["session_id"] == "s1"

            sessions = mgr.list_sessions()
            assert len(sessions) == 1
            assert sessions[0]["session_id"] == "s1"
            assert sessions[0]["execution_count"] == 1
            mgr.cleanup()

    def test_default_session(self, tmp_path):
        with MockContext() as ctx:
            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)

            result = mgr.execute("1 + 1")
            assert result.success is True
            assert backend.execute_calls[0]["session_id"] == "default"
            mgr.cleanup()

    def test_max_sessions_enforced(self, tmp_path):
        with MockContext(sandbox_max_sessions=2) as ctx:
            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)

            mgr.execute("1", session_id="s1")
            mgr.execute("2", session_id="s2")
            result = mgr.execute("3", session_id="s3")

            assert result.success is False
            assert "Maximum sessions" in result.error
            assert len(mgr.list_sessions()) == 2
            mgr.cleanup()

    def test_reset_active_session(self, tmp_path):
        with MockContext() as ctx:
            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)

            mgr.execute("1", session_id="s1")
            was_active = mgr.reset_session("s1")
            assert was_active is True
            assert len(mgr.list_sessions()) == 0
            assert "s1" in backend.reset_calls
            mgr.cleanup()

    def test_reset_inactive_session(self, tmp_path):
        with MockContext() as ctx:
            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)

            was_active = mgr.reset_session("nonexistent")
            assert was_active is False
            mgr.cleanup()

    def test_list_sessions_empty(self, tmp_path):
        with MockContext() as ctx:
            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)
            assert mgr.list_sessions() == []
            mgr.cleanup()

    def test_cleanup_clears_sessions(self, tmp_path):
        with MockContext() as ctx:
            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)

            mgr.execute("1", session_id="s1")
            mgr.execute("2", session_id="s2")
            mgr.cleanup()
            assert mgr.list_sessions() == []

    def test_execution_count_increments(self, tmp_path):
        with MockContext() as ctx:
            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)

            mgr.execute("1", session_id="s1")
            mgr.execute("2", session_id="s1")
            mgr.execute("3", session_id="s1")

            sessions = mgr.list_sessions()
            assert sessions[0]["execution_count"] == 3
            mgr.cleanup()

    def test_working_dir_created(self, tmp_path):
        with MockContext() as ctx:
            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)

            mgr.execute("1", session_id="test_sess")
            call = backend.execute_calls[0]
            working_dir = call["working_dir"]
            assert working_dir is not None
            assert Path(working_dir).exists()
            assert "test_sess" in str(working_dir)
            mgr.cleanup()


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

class TestSandboxTools:
    def test_sandbox_execute_success(self, tmp_path):
        with MockContext() as ctx:
            from agentic_cli.workflow.context import set_context_sandbox_manager

            backend = MockSandboxBackend(
                ExecutionResult(success=True, stdout="hi\n", result="42", execution_time=0.123)
            )
            mgr = SandboxManager(ctx.settings, backend=backend)
            token = set_context_sandbox_manager(mgr)

            try:
                from agentic_cli.tools.sandbox import sandbox_execute
                result = sandbox_execute("print('hi')")
                assert result["success"] is True
                assert result["stdout"] == "hi\n"
                assert result["result"] == "42"
                assert result["execution_time"] == 0.123
            finally:
                token.var.reset(token)
                mgr.cleanup()

    def test_sandbox_execute_no_manager(self, tmp_path):
        with MockContext():
            from agentic_cli.workflow.context import set_context_sandbox_manager

            token = set_context_sandbox_manager(None)
            try:
                from agentic_cli.tools.sandbox import sandbox_execute
                result = sandbox_execute("1 + 1")
                assert result["success"] is False
                assert "not available" in result["error"]
            finally:
                token.var.reset(token)

    def test_sandbox_reset_active(self, tmp_path):
        with MockContext() as ctx:
            from agentic_cli.workflow.context import set_context_sandbox_manager

            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)
            mgr.execute("1", session_id="s1")
            token = set_context_sandbox_manager(mgr)

            try:
                from agentic_cli.tools.sandbox import sandbox_reset
                result = sandbox_reset(session_id="s1")
                assert result["success"] is True
                assert "reset" in result["message"]
            finally:
                token.var.reset(token)
                mgr.cleanup()

    def test_sandbox_reset_inactive(self, tmp_path):
        with MockContext() as ctx:
            from agentic_cli.workflow.context import set_context_sandbox_manager

            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)
            token = set_context_sandbox_manager(mgr)

            try:
                from agentic_cli.tools.sandbox import sandbox_reset
                result = sandbox_reset(session_id="nope")
                assert result["success"] is True
                assert "was not active" in result["message"]
            finally:
                token.var.reset(token)
                mgr.cleanup()

    def test_sandbox_reset_no_manager(self, tmp_path):
        with MockContext():
            from agentic_cli.workflow.context import set_context_sandbox_manager

            token = set_context_sandbox_manager(None)
            try:
                from agentic_cli.tools.sandbox import sandbox_reset
                result = sandbox_reset()
                assert result["success"] is False
                assert "not available" in result["error"]
            finally:
                token.var.reset(token)


# ---------------------------------------------------------------------------
# Manager auto-detection
# ---------------------------------------------------------------------------

class TestManagerAutoDetection:
    def test_requires_sandbox_manager_detected(self):
        """Verify @requires('sandbox_manager') is detected by _detect_required_managers."""
        from agentic_cli.tools.sandbox import sandbox_execute

        assert hasattr(sandbox_execute, "requires")
        assert "sandbox_manager" in sandbox_execute.requires

    def test_base_manager_detects_sandbox(self, tmp_path):
        """BaseWorkflowManager picks up sandbox_manager from tool configs."""
        from agentic_cli.tools.sandbox import sandbox_execute
        from agentic_cli.workflow.config import AgentConfig

        config = AgentConfig(
            name="test",
            prompt="test",
            tools=[sandbox_execute],
        )

        with MockContext(google_api_key="test-key") as ctx:
            # Use ADK manager for concrete implementation
            from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

            mgr = GoogleADKWorkflowManager(
                agent_configs=[config],
                settings=ctx.settings,
            )
            assert "sandbox_manager" in mgr.required_managers


# ---------------------------------------------------------------------------
# JupyterLocalBackend integration tests
# ---------------------------------------------------------------------------

_has_jupyter = pytest.importorskip.__module__ is not None  # always True, placeholder

try:
    import jupyter_client as _jc
    _has_jupyter = True
except ImportError:
    _has_jupyter = False


@pytest.mark.skipif(not _has_jupyter, reason="jupyter_client not installed")
class TestJupyterLocalBackend:
    """Integration tests requiring jupyter_client + ipykernel."""

    def test_simple_execution(self, tmp_path):
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            result = backend.execute("print('hello')", session_id="test", working_dir=tmp_path)
            assert result.success is True
            assert "hello" in result.stdout
        finally:
            backend.cleanup()

    def test_state_persistence(self, tmp_path):
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            backend.execute("x = 42", session_id="test", working_dir=tmp_path)
            result = backend.execute("print(x)", session_id="test", working_dir=tmp_path)
            assert result.success is True
            assert "42" in result.stdout
        finally:
            backend.cleanup()

    def test_error_handling(self, tmp_path):
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            result = backend.execute("1 / 0", session_id="test", working_dir=tmp_path)
            assert result.success is False
            assert "ZeroDivisionError" in result.error
        finally:
            backend.cleanup()

    def test_reset_clears_state(self, tmp_path):
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            backend.execute("x = 99", session_id="test", working_dir=tmp_path)
            backend.reset_session("test")
            result = backend.execute("print(x)", session_id="test", working_dir=tmp_path)
            assert result.success is False
            assert "NameError" in result.error
        finally:
            backend.cleanup()

    def test_execute_result_value(self, tmp_path):
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            result = backend.execute("2 + 3", session_id="test", working_dir=tmp_path)
            assert result.success is True
            assert result.result == "5"
        finally:
            backend.cleanup()

    def test_has_session(self, tmp_path):
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            assert backend.has_session("test") is False
            backend.execute("1", session_id="test", working_dir=tmp_path)
            assert backend.has_session("test") is True
            backend.reset_session("test")
            assert backend.has_session("test") is False
        finally:
            backend.cleanup()
