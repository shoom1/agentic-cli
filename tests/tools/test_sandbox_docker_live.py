"""Live docker isolation tests — assert the real container boundary.

These require a real container runtime (docker/podman). Selection:
  * ``-m docker``            run only these
  * ``-m 'not docker'``     exclude them (used by the offline CI job)

Gating: normally each test is skipped when no runtime is present. In CI set
``SANDBOX_REQUIRE_DOCKER=1`` so a missing/broken runtime is a hard FAILURE
rather than a silent skip (a skipped test reads as green and would hide a
regression). Mirrors the ``-m llm`` live-test convention.
"""

import os
import subprocess

import pytest

from agentic_cli.tools.sandbox.backends.detect import detect_docker, docker_available
from agentic_cli.tools.sandbox.backends.jupyter_docker import JupyterDockerBackend
from tests.conftest import MockContext

# Every test in this module is a docker test (selectable via -m docker).
pytestmark = pytest.mark.docker

_DOCKER = docker_available()
_requires_docker = pytest.mark.skipif(not _DOCKER, reason="no container runtime available")


def test_docker_runtime_present_when_required():
    """Fail loudly (not skip) when CI declares a runtime is required but none is
    available — so a broken daemon or failed image pull can't pass as green."""
    if os.environ.get("SANDBOX_REQUIRE_DOCKER") == "1":
        assert _DOCKER, "SANDBOX_REQUIRE_DOCKER=1 but no container runtime is available"
    elif not _DOCKER:
        pytest.skip("no container runtime available")


@pytest.fixture
def backend(tmp_path):
    with MockContext(sandbox_backend="jupyter_docker") as ctx:
        b = JupyterDockerBackend(ctx.settings)
        try:
            yield b
        finally:
            b.cleanup()


# --------------------------------------------------------------------------
# Boundary: network, filesystem
# --------------------------------------------------------------------------

@_requires_docker
def test_network_is_blocked(backend, tmp_path):
    code = ("import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=3); print('OPEN')\n"
            "except OSError:\n"
            "    print('BLOCKED')\n")
    r = backend.execute(code, "net", timeout_seconds=60, working_dir=tmp_path)
    assert r.success is True, r.error
    assert "BLOCKED" in r.stdout


@_requires_docker
def test_rootfs_is_read_only(backend, tmp_path):
    code = ("try:\n"
            "    open('/etc/passwd', 'a').write('x'); print('WRITABLE')\n"
            "except OSError:\n"
            "    print('READONLY')\n")
    r = backend.execute(code, "ro", timeout_seconds=60, working_dir=tmp_path)
    assert r.success is True, r.error
    assert "READONLY" in r.stdout


@_requires_docker
def test_workspace_is_writable(backend, tmp_path):
    r = backend.execute("open('/workspace/out.txt','w').write('ok'); print('WROTE')",
                        "ws", timeout_seconds=60, working_dir=tmp_path)
    assert r.success is True, r.error
    assert "WROTE" in r.stdout
    assert (tmp_path / "out.txt").read_text() == "ok"


@_requires_docker
def test_host_path_outside_workspace_not_accessible(backend, tmp_path):
    """A host file that is NOT bind-mounted must be invisible: its host path
    does not exist inside the container's mount namespace."""
    secret = tmp_path.parent / "host_only_secret.txt"
    secret.write_text("top-secret")
    code = ("try:\n"
            f"    print('LEAKED:' + open({str(secret)!r}).read())\n"
            "except OSError:\n"
            "    print('NO_HOST_ACCESS')\n")
    r = backend.execute(code, "hostpath", timeout_seconds=60, working_dir=tmp_path)
    assert r.success is True, r.error
    assert "NO_HOST_ACCESS" in r.stdout


# --------------------------------------------------------------------------
# Cross-session isolation
# --------------------------------------------------------------------------

@_requires_docker
def test_sessions_are_isolated(backend, tmp_path):
    """Distinct session_ids get distinct containers: neither in-memory state
    nor workspace files leak between them."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    ra = backend.execute("secret = 'alpha'\nopen('/workspace/a.txt', 'w').write('alpha')",
                         "sessA", timeout_seconds=60, working_dir=dir_a)
    assert ra.success is True, ra.error

    rb = backend.execute(
        "import os\n"
        "print('VAR_LEAK' if 'secret' in dir() else 'NO_VAR')\n"
        "print('FILE_LEAK' if os.path.exists('/workspace/a.txt') else 'NO_FILE')\n",
        "sessB", timeout_seconds=60, working_dir=dir_b)
    assert rb.success is True, rb.error
    assert "NO_VAR" in rb.stdout
    assert "NO_FILE" in rb.stdout


# --------------------------------------------------------------------------
# Cooperative interrupt: a runaway cell is aborted but the session survives
# --------------------------------------------------------------------------

@_requires_docker
@pytest.mark.xfail(
    reason="post-interrupt response can desync on a real daemon (r2 stdout came back "
           "empty in CI); needs live-daemon debugging of the cooperative-interrupt path",
    strict=False,
)
def test_interrupt_preserves_session_state(backend, tmp_path):
    r0 = backend.execute("kept = 123", "intr", timeout_seconds=60, working_dir=tmp_path)
    assert r0.success is True, r0.error

    # Runs far longer than the per-cell timeout -> host sends a cooperative
    # interrupt (docker kill --signal=INT) instead of killing the container.
    r1 = backend.execute("import time\nfor _ in range(120):\n    time.sleep(1)",
                         "intr", timeout_seconds=3, working_dir=tmp_path)
    assert r1.success is False  # aborted

    # Same session still alive with prior state intact.
    r2 = backend.execute("print(kept)", "intr", timeout_seconds=60, working_dir=tmp_path)
    assert r2.success is True, r2.error
    assert "123" in r2.stdout


# --------------------------------------------------------------------------
# Resource caps (may need first-run threshold tuning per runner cgroup config)
# --------------------------------------------------------------------------

@_requires_docker
def test_memory_cap_oom_kills(tmp_path):
    """A single allocation far past --memory (swap disabled) is OOM-killed;
    the backend surfaces failure rather than a clean success."""
    with MockContext(sandbox_backend="jupyter_docker", sandbox_memory_mb=256) as ctx:
        b = JupyterDockerBackend(ctx.settings)
        try:
            r = b.execute("x = bytearray(1024 * 1024 * 1024)  # 1 GiB vs 256 MiB cap",
                          "oom", timeout_seconds=45, working_dir=tmp_path)
            assert r.success is False
        finally:
            b.cleanup()


@_requires_docker
def test_pids_limit_caps_thread_bomb(tmp_path):
    """--pids-limit bounds the number of tasks; a thread bomb hits it."""
    with MockContext(sandbox_backend="jupyter_docker", sandbox_pids_limit=128) as ctx:
        b = JupyterDockerBackend(ctx.settings)
        try:
            code = ("import threading, time\n"
                    "started = 0\n"
                    "try:\n"
                    "    for _ in range(500):\n"
                    "        threading.Thread(target=lambda: time.sleep(30)).start()\n"
                    "        started += 1\n"
                    "    print('NO_LIMIT', started)\n"
                    "except RuntimeError:\n"
                    "    print('PIDS_CAPPED', started)\n")
            r = b.execute(code, "pids", timeout_seconds=45, working_dir=tmp_path)
            assert "PIDS_CAPPED" in r.stdout
        finally:
            b.cleanup()


# --------------------------------------------------------------------------
# Lifecycle: no leaked containers after cleanup
# --------------------------------------------------------------------------

@_requires_docker
def test_no_orphaned_containers_after_cleanup(tmp_path):
    runtime = detect_docker().runtime or "docker"
    with MockContext(sandbox_backend="jupyter_docker") as ctx:
        b = JupyterDockerBackend(ctx.settings)
        b.execute("x = 1", "orphan1", timeout_seconds=60, working_dir=tmp_path)
        b.execute("y = 2", "orphan2", timeout_seconds=60, working_dir=tmp_path)
        b.cleanup()
    out = subprocess.run(
        [runtime, "ps", "-aq", "--filter", "label=agentic-sandbox=1"],
        capture_output=True, text=True, check=False,
    )
    assert out.stdout.strip() == "", f"orphaned containers remain: {out.stdout!r}"
