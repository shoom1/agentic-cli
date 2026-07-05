"""Live docker smoke test — asserts the real isolation boundary.

Opt-in: skipped unless a real container runtime is available. Mirrors the
FAISS `importorskip` / `-m llm` live-test convention.
"""

import pytest

from agentic_cli.tools.sandbox.backends.detect import docker_available

pytestmark = pytest.mark.skipif(not docker_available(), reason="no container runtime available")

from agentic_cli.tools.sandbox.backends.jupyter_docker import JupyterDockerBackend
from tests.conftest import MockContext


@pytest.fixture
def backend(tmp_path):
    with MockContext(sandbox_backend="jupyter_docker") as ctx:
        b = JupyterDockerBackend(ctx.settings)
        try:
            yield b
        finally:
            b.cleanup()


def test_network_is_blocked(backend, tmp_path):
    code = ("import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=3); print('OPEN')\n"
            "except OSError:\n"
            "    print('BLOCKED')\n")
    r = backend.execute(code, "net", timeout_seconds=60, working_dir=tmp_path)
    assert r.success is True, r.error
    assert "BLOCKED" in r.stdout


def test_rootfs_is_read_only(backend, tmp_path):
    code = ("try:\n"
            "    open('/etc/passwd', 'a').write('x'); print('WRITABLE')\n"
            "except OSError:\n"
            "    print('READONLY')\n")
    r = backend.execute(code, "ro", timeout_seconds=60, working_dir=tmp_path)
    assert r.success is True, r.error
    assert "READONLY" in r.stdout


def test_workspace_is_writable(backend, tmp_path):
    r = backend.execute("open('/workspace/out.txt','w').write('ok'); print('WROTE')",
                        "ws", timeout_seconds=60, working_dir=tmp_path)
    assert r.success is True, r.error
    assert "WROTE" in r.stdout
    assert (tmp_path / "out.txt").read_text() == "ok"
