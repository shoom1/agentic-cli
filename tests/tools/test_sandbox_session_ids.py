"""Sandbox session IDs name a directory and a container, so they are
validated rather than sanitized.

``sanitize_filename`` mapped every disallowed character to ``_``: an empty ID
named the directory holding every session (which the Docker backend mounts
into the container), and different IDs such as ``a.b`` and ``a_b`` shared a
directory and a container.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_cli.tools.sandbox.manager import SandboxManager
from tests.conftest import MockContext
from tests.tools.test_sandbox import MockSandboxBackend


@pytest.mark.parametrize(
    "session_id",
    ["", ".", "..", "a/b", "../x", "a.b", "a b", "x" * 65, "a\x00b"],
)
def test_invalid_session_id_is_refused_before_anything_runs(session_id):
    with MockContext() as ctx:
        backend = MockSandboxBackend()
        mgr = SandboxManager(ctx.settings, backend=backend)
        result = mgr.execute("x = 1", session_id=session_id)
        assert result.success is False
        assert "session_id" in result.error
        assert backend.execute_calls == []
        sandbox_root = Path(ctx.settings.workspace_dir) / "sandbox"
        assert not sandbox_root.exists() or list(sandbox_root.iterdir()) == []


@pytest.mark.parametrize("session_id", ["default", "analysis_2", "Run-3", "x" * 64])
def test_valid_session_ids_run_in_their_own_directory(session_id):
    with MockContext() as ctx:
        backend = MockSandboxBackend()
        mgr = SandboxManager(ctx.settings, backend=backend)
        assert mgr.execute("x = 1", session_id=session_id).success is True
        [call] = backend.execute_calls
        assert call["working_dir"] == Path(ctx.settings.workspace_dir) / "sandbox" / session_id


def test_container_names_differ_between_backend_instances(tmp_path):
    """Two app processes using the same session ID must not collide on the
    container name (``docker run --name`` conflict)."""
    from agentic_cli.tools.sandbox.backends.jupyter_docker import JupyterDockerBackend

    with MockContext(stateful_executor_backend="docker") as ctx:
        first = JupyterDockerBackend(ctx.settings)._build_spec("default", tmp_path).name
        second = JupyterDockerBackend(ctx.settings)._build_spec("default", tmp_path).name
    assert first != second
    assert first.startswith("agentic-sbx-default-")
