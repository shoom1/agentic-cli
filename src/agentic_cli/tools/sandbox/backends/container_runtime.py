"""Container runtime seam: assemble and launch `docker run` (test-injectable)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field

from agentic_cli.logging import Loggers

logger = Loggers.tools()


@dataclass
class Mount:
    host: str
    container: str
    read_only: bool = True

    def to_flag(self) -> str:
        spec = f"{self.host}:{self.container}"
        return spec + ":ro" if self.read_only else spec


@dataclass
class ContainerSpec:
    image: str
    name: str
    command: list[str]
    network: str = "none"
    memory_mb: int = 2048
    cpus: float = 2.0
    pids_limit: int = 256
    user: str = ""
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[Mount] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)


class ContainerHandle:
    """Thin wrapper over a running container's process + stdio streams."""

    def __init__(self, name: str, proc: subprocess.Popen) -> None:
        self.name = name
        self._proc = proc
        self.stdin = proc.stdin
        self.stdout = proc.stdout
        self.stderr = proc.stderr

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> int | None:
        return self._proc.poll()


class DockerContainerRuntime:
    """Launch and control containers via the docker/podman CLI."""

    def __init__(self, exe: str = "docker") -> None:
        self._exe = exe

    @staticmethod
    def build_run_argv(spec: ContainerSpec, exe: str) -> list[str]:
        # Docker's default seccomp profile stays in effect (we never pass
        # --privileged or --security-opt seccomp=unconfined), so dangerous
        # syscalls remain blocked on top of the dropped capabilities.
        argv: list[str] = [
            exe, "run", "--rm", "-i",
            "--network", spec.network,
            "--read-only", "--tmpfs", "/tmp",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", f"{spec.memory_mb}m",
            "--memory-swap", f"{spec.memory_mb}m",
            "--pids-limit", str(spec.pids_limit),
            "--cpus", str(spec.cpus),
            "--name", spec.name,
        ]
        if spec.user:
            argv += ["--user", spec.user]
        for key, value in spec.env.items():
            argv += ["-e", f"{key}={value}"]
        for mount in spec.mounts:
            argv += ["-v", mount.to_flag()]
        for key, value in spec.labels.items():
            argv += ["--label", f"{key}={value}"]
        argv.append(spec.image)
        argv += spec.command
        return argv

    def start(self, spec: ContainerSpec) -> ContainerHandle:
        argv = self.build_run_argv(spec, self._exe)
        logger.debug("container_start", name=spec.name, image=spec.image)
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        return ContainerHandle(spec.name, proc)

    def kill(self, name: str, signal: str | None = None) -> None:
        argv = [self._exe, "kill"]
        if signal:
            argv += ["--signal", signal]
        argv.append(name)
        subprocess.run(argv, capture_output=True, check=False)

    def inspect(self, name: str) -> dict:
        proc = subprocess.run(
            [self._exe, "inspect", name], capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            return {}
        try:
            data = json.loads(proc.stdout)
            return data[0] if isinstance(data, list) and data else {}
        except (json.JSONDecodeError, IndexError):
            return {}

    def remove(self, name: str) -> None:
        subprocess.run([self._exe, "rm", "-f", name], capture_output=True, check=False)
