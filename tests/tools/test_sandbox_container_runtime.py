"""Tests for the docker-CLI container runtime seam."""

from agentic_cli.tools.sandbox.backends.container_runtime import (
    ContainerSpec, Mount, DockerContainerRuntime,
)


def _spec(**kw):
    base = dict(
        image="img:tag",
        name="agentic-sbx-s1",
        command=["python", "/opt/agentic_sandbox/driver.py"],
        network="none",
        memory_mb=2048,
        cpus=2.0,
        pids_limit=256,
        user="1000:1000",
        env={"HOME": "/tmp"},
        mounts=[Mount("/host/ws", "/workspace", read_only=False),
                Mount("/host/driver.py", "/opt/agentic_sandbox/driver.py")],
        labels={"agentic-sandbox": "1"},
    )
    base.update(kw)
    return ContainerSpec(**base)


def test_argv_carries_isolation_flags():
    argv = DockerContainerRuntime.build_run_argv(_spec(), "docker")
    joined = " ".join(argv)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--network none" in joined
    assert "--read-only" in argv
    assert "--tmpfs" in argv and "/tmp" in argv
    assert "ALL" in argv and "--cap-drop" in argv
    assert "no-new-privileges" in joined
    assert "--memory" in argv and "2048m" in argv
    assert "--memory-swap" in argv  # swap disabled == memory
    assert "--pids-limit" in argv and "256" in argv
    assert "--cpus" in argv and "2.0" in argv
    assert "--user" in argv and "1000:1000" in argv
    assert "-e" in argv and "HOME=/tmp" in argv
    assert "/host/ws:/workspace" in argv
    assert "/host/driver.py:/opt/agentic_sandbox/driver.py:ro" in argv
    assert "--label" in argv and "agentic-sandbox=1" in argv
    # image + command are last
    assert argv[-3:] == ["img:tag", "python", "/opt/agentic_sandbox/driver.py"]


def test_memory_swap_equals_memory():
    argv = DockerContainerRuntime.build_run_argv(_spec(memory_mb=512), "docker")
    i = argv.index("--memory-swap")
    assert argv[i + 1] == "512m"


def test_empty_user_omits_flag():
    argv = DockerContainerRuntime.build_run_argv(_spec(user=""), "docker")
    assert "--user" not in argv


def test_start_invokes_popen_with_argv(monkeypatch):
    seen = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            seen["argv"] = argv
            seen["kw"] = kw
            self.stdin = object()
            self.stdout = object()
            self.stderr = object()
            self.pid = 4321
        def poll(self):
            return None

    monkeypatch.setattr(
        "agentic_cli.tools.sandbox.backends.container_runtime.subprocess.Popen",
        FakePopen,
    )
    rt = DockerContainerRuntime("docker")
    handle = rt.start(_spec())
    assert seen["argv"][0] == "docker"
    assert handle.name == "agentic-sbx-s1"
    assert handle.poll() is None
