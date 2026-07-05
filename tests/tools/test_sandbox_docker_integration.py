"""End-to-end: real driver subprocess (no docker) via a local-process runtime."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jupyter_client")

from agentic_cli.tools.sandbox.backends import driver as _driver_mod
from agentic_cli.tools.sandbox.backends.detect import DockerAvailability
from agentic_cli.tools.sandbox.backends.container_runtime import ContainerHandle
from agentic_cli.tools.sandbox.backends.jupyter_docker import JupyterDockerBackend
from tests.conftest import MockContext

DRIVER = Path(_driver_mod.__file__)


class LocalDriverRuntime:
    """Runs `python driver.py` locally, standing in for `docker run`."""

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen] = {}

    def start(self, spec) -> ContainerHandle:
        # Find the host workspace mount (container path == "/workspace")
        ws = next(m.host for m in spec.mounts if m.container == "/workspace")
        # Pass the full environment so the local kernel can find Jupyter/IPython paths,
        # but override the workspace var to point at our tmp dir.
        env = {**os.environ, "AGENTIC_SANDBOX_WORKSPACE": ws}
        proc = subprocess.Popen(
            [sys.executable, str(DRIVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=ws,
            env=env,
        )
        self._procs[spec.name] = proc
        return ContainerHandle(spec.name, proc)

    def kill(self, name: str, signal: str | None = None) -> None:
        proc = self._procs.get(name)
        if proc and proc.poll() is None:
            proc.kill()

    def inspect(self, name: str) -> dict:
        return {}

    def remove(self, name: str) -> None:
        pass


@pytest.fixture
def backend(tmp_path):
    with MockContext(sandbox_backend="jupyter_docker", sandbox_start_timeout=60) as ctx:
        yield JupyterDockerBackend(
            ctx.settings,
            runtime=LocalDriverRuntime(),
            detect_fn=lambda: DockerAvailability(True, "docker", "local"),
        )


def test_stateful_execution_across_calls(backend, tmp_path):
    r1 = backend.execute("data = [1, 2, 3]", "s1", timeout_seconds=60, working_dir=tmp_path)
    assert r1.success is True, r1.error
    r2 = backend.execute("print(sum(data))", "s1", timeout_seconds=60, working_dir=tmp_path)
    assert r2.success is True, r2.error
    assert "6" in r2.stdout
    backend.cleanup()


def test_error_surfaces(backend, tmp_path):
    r = backend.execute("1/0", "s1", timeout_seconds=60, working_dir=tmp_path)
    assert r.success is False
    assert "ZeroDivisionError" in r.error
    backend.cleanup()
