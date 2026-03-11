"""Tests for sandbox execution tools.

Tests cover:
- ExecutionResult model
- SandboxManager with mock backend
- sandbox_execute tool function
- SandboxCommand CLI command
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



# ---------------------------------------------------------------------------
# SandboxCommand
# ---------------------------------------------------------------------------

class TestSandboxCommand:
    @pytest.fixture()
    def mock_app(self, tmp_path):
        """Create a mock app with session and workflow."""
        app = MagicMock()
        app.session = MagicMock()
        return app

    def _make_manager(self, ctx, sessions=None):
        """Create a SandboxManager with optional pre-created sessions."""
        backend = MockSandboxBackend()
        mgr = SandboxManager(ctx.settings, backend=backend)
        for sid in (sessions or []):
            mgr.execute("1", session_id=sid)
        return mgr

    @pytest.mark.asyncio
    async def test_list_sessions(self, mock_app):
        with MockContext() as ctx:
            from agentic_cli.cli.builtin_commands import SandboxCommand

            mgr = self._make_manager(ctx, sessions=["s1", "s2"])
            mock_app.workflow.sandbox_manager = mgr
            cmd = SandboxCommand()

            await cmd.execute("", mock_app)
            mock_app.session.add_rich.assert_called_once()
            mgr.cleanup()

    @pytest.mark.asyncio
    async def test_list_empty(self, mock_app):
        with MockContext() as ctx:
            from agentic_cli.cli.builtin_commands import SandboxCommand

            mgr = self._make_manager(ctx)
            mock_app.workflow.sandbox_manager = mgr
            cmd = SandboxCommand()

            await cmd.execute("", mock_app)
            mock_app.session.add_message.assert_called_once_with(
                "system", "No active sandbox sessions."
            )
            mgr.cleanup()

    @pytest.mark.asyncio
    async def test_reset_session(self, mock_app):
        with MockContext() as ctx:
            from agentic_cli.cli.builtin_commands import SandboxCommand

            mgr = self._make_manager(ctx, sessions=["s1"])
            mock_app.workflow.sandbox_manager = mgr
            cmd = SandboxCommand()

            await cmd.execute("reset s1", mock_app)
            mock_app.session.add_success.assert_called_once()
            assert len(mgr.list_sessions()) == 0
            mgr.cleanup()

    @pytest.mark.asyncio
    async def test_reset_default_session(self, mock_app):
        with MockContext() as ctx:
            from agentic_cli.cli.builtin_commands import SandboxCommand

            mgr = self._make_manager(ctx, sessions=["default"])
            mock_app.workflow.sandbox_manager = mgr
            cmd = SandboxCommand()

            await cmd.execute("reset", mock_app)
            mock_app.session.add_success.assert_called_once()
            assert len(mgr.list_sessions()) == 0
            mgr.cleanup()

    @pytest.mark.asyncio
    async def test_reset_inactive_session(self, mock_app):
        with MockContext() as ctx:
            from agentic_cli.cli.builtin_commands import SandboxCommand

            mgr = self._make_manager(ctx)
            mock_app.workflow.sandbox_manager = mgr
            cmd = SandboxCommand()

            await cmd.execute("reset nope", mock_app)
            mock_app.session.add_warning.assert_called_once()
            mgr.cleanup()

    @pytest.mark.asyncio
    async def test_reset_all(self, mock_app):
        with MockContext() as ctx:
            from agentic_cli.cli.builtin_commands import SandboxCommand

            mgr = self._make_manager(ctx, sessions=["s1", "s2", "s3"])
            mock_app.workflow.sandbox_manager = mgr
            cmd = SandboxCommand()

            await cmd.execute("reset --all", mock_app)
            mock_app.session.add_success.assert_called_once()
            assert "3" in mock_app.session.add_success.call_args[0][0]
            assert len(mgr.list_sessions()) == 0
            mgr.cleanup()

    @pytest.mark.asyncio
    async def test_no_sandbox_manager(self, mock_app):
        from agentic_cli.cli.builtin_commands import SandboxCommand

        mock_app.workflow.sandbox_manager = None
        cmd = SandboxCommand()

        await cmd.execute("", mock_app)
        mock_app.session.add_warning.assert_called_once()
        assert "not available" in mock_app.session.add_warning.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_no_workflow(self, mock_app):
        from agentic_cli.cli.builtin_commands import SandboxCommand

        mock_app.workflow = property(lambda self: (_ for _ in ()).throw(RuntimeError))
        type(mock_app).workflow = property(lambda self: (_ for _ in ()).throw(RuntimeError("not ready")))
        cmd = SandboxCommand()

        await cmd.execute("", mock_app)
        mock_app.session.add_warning.assert_called_once()


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

class TestSandboxRestrictions:
    """Tests for sandbox network and shell restrictions."""

    def test_blocked_import_requests(self, tmp_path):
        """Sandbox blocks import of requests module."""
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            result = backend.execute(
                "import requests", session_id="test", working_dir=tmp_path,
            )
            assert result.success is False
            assert "not available in the sandbox" in result.error
        finally:
            backend.cleanup()

    def test_blocked_import_urllib(self, tmp_path):
        """Sandbox blocks import of urllib module."""
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            result = backend.execute(
                "import urllib.request", session_id="test", working_dir=tmp_path,
            )
            assert result.success is False
            assert "not available in the sandbox" in result.error
        finally:
            backend.cleanup()

    def test_blocked_import_socket(self, tmp_path):
        """Sandbox blocks import of socket module."""
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            result = backend.execute(
                "import socket", session_id="test", working_dir=tmp_path,
            )
            assert result.success is False
            assert "not available in the sandbox" in result.error
        finally:
            backend.cleanup()

    def test_blocked_import_subprocess(self, tmp_path):
        """Sandbox blocks import of subprocess module."""
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            result = backend.execute(
                "import subprocess", session_id="test", working_dir=tmp_path,
            )
            assert result.success is False
            assert "not available in the sandbox" in result.error
        finally:
            backend.cleanup()

    def test_shell_escape_blocked(self, tmp_path):
        """Sandbox blocks ! shell escape syntax."""
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            result = backend.execute(
                "!pip install evil-package", session_id="test", working_dir=tmp_path,
            )
            assert result.success is False
            assert "not allowed" in result.error.lower()
        finally:
            backend.cleanup()

    def test_pip_magic_blocked(self, tmp_path):
        """Sandbox blocks %pip magic command."""
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            result = backend.execute(
                "%pip install evil-package", session_id="test", working_dir=tmp_path,
            )
            assert result.success is False
            assert "not allowed" in result.error.lower()
        finally:
            backend.cleanup()

    def test_safe_math_still_works(self, tmp_path):
        """Sandbox still allows normal math/data operations."""
        from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

        backend = JupyterLocalBackend()
        try:
            result = backend.execute(
                "import math; print(math.pi)", session_id="test", working_dir=tmp_path,
            )
            assert result.success is True
            assert "3.14" in result.stdout
        finally:
            backend.cleanup()


class TestJupyterLocalBackend:
    """Integration tests for JupyterLocalBackend."""

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
