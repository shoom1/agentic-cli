"""Python execution uses the configured timeout unless the call names one.

``execute_python`` and ``sandbox_execute`` declared a concrete default in their
signatures (30 s and 120 s) and always passed it down, so the layer below never
fell back to ``python_executor_timeout`` / ``sandbox_timeout``: the settings did
nothing. The default is now ``None``: an omitted (or ``null``) timeout means
the setting, an explicit one is used as given, and zero or less is refused.
"""

from __future__ import annotations

import pytest

from agentic_cli.tools.executor import SafePythonExecutor
from agentic_cli.tools.sandbox.manager import SandboxManager
from tests.conftest import MockContext
from tests.tools.test_sandbox import MockSandboxBackend

BUSY = "n = 0\nwhile True:\n    n += 1\n"


@pytest.fixture
def subprocess_timeouts(monkeypatch) -> list:
    """Record the timeout each execute_python call hands to its subprocess."""
    seen: list = []

    def fake(self, code, context, timeout):
        seen.append(timeout)
        return {"success": True, "output": "", "result": None, "error": "", "execution_time_ms": 0}

    monkeypatch.setattr(SafePythonExecutor, "_execute_in_subprocess", fake)
    return seen


class TestExecutePython:
    def test_omitted_timeout_uses_the_setting(self, subprocess_timeouts):
        from agentic_cli.tools.execution_tools import execute_python

        with MockContext(python_executor_timeout=7, os_sandbox_enabled=False):
            execute_python("1 + 1")
        assert subprocess_timeouts == [7]

    def test_null_timeout_uses_the_setting(self, subprocess_timeouts):
        from agentic_cli.tools.execution_tools import execute_python

        with MockContext(python_executor_timeout=7, os_sandbox_enabled=False):
            execute_python("1 + 1", timeout_seconds=None)
        assert subprocess_timeouts == [7]

    @pytest.mark.parametrize("requested", [2, 300])
    def test_explicit_timeout_is_used_as_given(self, subprocess_timeouts, requested):
        from agentic_cli.tools.execution_tools import execute_python

        with MockContext(python_executor_timeout=7, os_sandbox_enabled=False):
            execute_python("1 + 1", timeout_seconds=requested)
        assert subprocess_timeouts == [requested]

    @pytest.mark.parametrize("requested", [0, -5])
    def test_non_positive_timeout_is_refused(self, subprocess_timeouts, requested):
        from agentic_cli.tools.execution_tools import execute_python

        with MockContext(python_executor_timeout=7, os_sandbox_enabled=False):
            result = execute_python("1 + 1", timeout_seconds=requested)
        assert result["success"] is False
        assert "timeout_seconds" in result["error"]
        assert subprocess_timeouts == []

    def test_the_setting_really_stops_a_runaway_loop(self):
        """End to end, no seams: a busy loop is killed at the configured limit."""
        from agentic_cli.tools.execution_tools import execute_python

        with MockContext(python_executor_timeout=1, os_sandbox_enabled=False):
            result = execute_python(BUSY)
        assert result["success"] is False
        assert "timed out after 1 seconds" in result["error"]


def _sandbox_tools(mgr):
    from agentic_cli.tools.factories import make_sandbox_tool
    from agentic_cli.tools.sandbox import sandbox_execute

    return {"module": sandbox_execute, "factory": make_sandbox_tool(mgr)}


@pytest.mark.parametrize("variant", ["module", "factory"])
class TestSandboxExecute:
    def _run(self, variant, **kwargs):
        from agentic_cli.workflow.service_registry import set_service_registry

        with MockContext(stateful_executor_backend="local", sandbox_timeout=7) as ctx:
            backend = MockSandboxBackend()
            mgr = SandboxManager(ctx.settings, backend=backend)
            token = set_service_registry({"sandbox_manager": mgr})
            try:
                result = _sandbox_tools(mgr)[variant]("x = 1", **kwargs)
            finally:
                token.var.reset(token)
                mgr.cleanup()
        return result, [c["timeout_seconds"] for c in backend.execute_calls]

    def test_omitted_timeout_uses_the_setting(self, variant):
        _, timeouts = self._run(variant)
        assert timeouts == [7]

    def test_null_timeout_uses_the_setting(self, variant):
        _, timeouts = self._run(variant, timeout_seconds=None)
        assert timeouts == [7]

    @pytest.mark.parametrize("requested", [2, 300])
    def test_explicit_timeout_is_used_as_given(self, variant, requested):
        _, timeouts = self._run(variant, timeout_seconds=requested)
        assert timeouts == [requested]

    @pytest.mark.parametrize("requested", [0, -5])
    def test_non_positive_timeout_is_refused(self, variant, requested):
        result, timeouts = self._run(variant, timeout_seconds=requested)
        assert result["success"] is False
        assert "timeout_seconds" in result["error"]
        assert timeouts == []
