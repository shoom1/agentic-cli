"""Execution backends for long-running jobs.

A backend is *where the work runs* (≠ the ADK/LangGraph orchestrator backend).
All backends speak the same :class:`JobBackend` interface to ``JobManager``, so
job lifecycle, persistence, monitoring, and the ``job_*`` tools are uniform
regardless of where a job actually executes.

Milestone 1 ships two backends:

- :class:`SubprocessBackend` — detached OS process (``start_new_session=True``),
  output streamed to log files, completion recorded via an on-disk ``exit_code``
  sentinel so it survives a CLI restart.
- :class:`InProcessBackend` — a Python callable run on a thread pool. Does *not*
  survive a restart and cannot be force-killed (thread); use for lighter work
  that should not block the agent turn.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_cli.tools.jobs.manager import JobRecord


class JobState(str, Enum):
    """Lifecycle state of a job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


TERMINAL_STATES = frozenset(
    {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.UNKNOWN}
)


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists (signal 0 probe)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def _read_exit_code(exit_file: Path) -> int | None:
    """Read the sentinel exit-code file, or None if absent/unreadable."""
    if not exit_file.exists():
        return None
    try:
        text = exit_file.read_text().strip()
        return int(text) if text else 1
    except (ValueError, OSError):
        return 1


def _tail(path: Path, last_n: int) -> list[str]:
    """Return the last ``last_n`` lines of a text file (empty if missing)."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return lines[-last_n:] if last_n > 0 else lines


class JobBackend(ABC):
    """Uniform interface every execution backend implements for ``JobManager``."""

    name: str = "base"
    survives_restart: bool = False
    streams_logs: bool = False

    @abstractmethod
    def start(self, record: "JobRecord", job_dir: Path) -> None:
        """Begin executing the job. Non-blocking. Sets handle/pid on ``record``."""

    @abstractmethod
    def poll(self, record: "JobRecord", job_dir: Path) -> JobState:
        """Return the job's current state (reads sentinel / liveness / remote)."""

    @abstractmethod
    def cancel(self, record: "JobRecord", job_dir: Path) -> None:
        """Best-effort cancellation of a running job."""

    def logs(
        self, record: "JobRecord", job_dir: Path, last_n: int, stream: str
    ) -> list[str]:
        """Return the last ``last_n`` lines of stdout/stderr."""
        fname = "stderr.log" if stream == "stderr" else "stdout.log"
        return _tail(job_dir / fname, last_n)

    def result(self, record: "JobRecord", job_dir: Path) -> Any:
        """Return the job's result (backend-specific)."""
        return {"exit_code": record.exit_code, "stdout_tail": self.logs(record, job_dir, 20, "stdout")}


class SubprocessBackend(JobBackend):
    """Detached subprocess; restart-safe via an on-disk ``exit_code`` sentinel."""

    name = "subprocess"
    survives_restart = True
    streams_logs = True

    def __init__(self) -> None:
        # job_id -> Popen, kept so cancel() can signal the process group.
        self._procs: dict[str, subprocess.Popen] = {}

    def start(self, record: "JobRecord", job_dir: Path) -> None:
        command = record.spec.get("command")
        if not command:
            raise ValueError("subprocess job requires spec['command']")
        cwd = record.spec.get("cwd") or None

        inner = (
            command
            if isinstance(command, str)
            else " ".join(shlex.quote(c) for c in command)
        )
        exit_file = job_dir / "exit_code"
        # Run the command in a subshell, then record its exit status on disk so
        # completion is detectable after a CLI restart (we can't waitpid a
        # process we didn't start this session). The subshell ensures a command
        # that calls ``exit`` doesn't skip the sentinel write.
        wrapped = (
            f"( {inner} )\nrc=$?\nprintf '%s' \"$rc\" > {shlex.quote(str(exit_file))}\n"
        )

        out = open(job_dir / "stdout.log", "wb")
        err = open(job_dir / "stderr.log", "wb")
        try:
            proc = subprocess.Popen(
                ["sh", "-c", wrapped],
                cwd=cwd,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        finally:
            out.close()
            err.close()
        self._procs[record.job_id] = proc
        record.pid = proc.pid
        record.backend_handle = str(proc.pid)

    def poll(self, record: "JobRecord", job_dir: Path) -> JobState:
        exit_file = job_dir / "exit_code"
        code = _read_exit_code(exit_file)
        if code is not None:
            record.exit_code = code
            self._procs.pop(record.job_id, None)
            return JobState.SUCCEEDED if code == 0 else JobState.FAILED

        # If we still hold the live process handle (same session), reap it via
        # poll() — this both prevents zombies (which os.kill(pid,0) would report
        # as alive) and gives a returncode fallback if the sentinel is missing.
        proc = self._procs.get(record.job_id)
        if proc is not None:
            rc = proc.poll()
            if rc is None:
                return JobState.RUNNING
            code = _read_exit_code(exit_file)  # prefer the sentinel
            if code is None:
                code = rc
            record.exit_code = code
            self._procs.pop(record.job_id, None)
            return JobState.SUCCEEDED if code == 0 else JobState.FAILED

        # No live handle (e.g. after a CLI restart): fall back to PID liveness.
        if record.pid and _pid_alive(record.pid):
            return JobState.RUNNING
        return JobState.UNKNOWN

    def cancel(self, record: "JobRecord", job_dir: Path) -> None:
        pid = record.pid
        if not pid:
            return
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        self._procs.pop(record.job_id, None)


class InProcessBackend(JobBackend):
    """Run a Python callable on a thread pool. Not restart-safe; no force-kill."""

    name = "inprocess"
    survives_restart = False
    streams_logs = False

    def __init__(self, max_workers: int = 8) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="job-inproc"
        )
        self._futures: dict[str, Future] = {}

    def start(self, record: "JobRecord", job_dir: Path) -> None:
        target = record.spec.get("target")
        if not callable(target):
            raise ValueError("inprocess job requires a callable spec['target']")
        args = record.spec.get("args", ())
        kwargs = record.spec.get("kwargs", {})
        exit_file = job_dir / "exit_code"
        result_file = job_dir / "result.json"

        def _run() -> None:
            try:
                res = target(*args, **kwargs)
                try:
                    result_file.write_text(json.dumps(res, default=str))
                except (TypeError, ValueError, OSError):
                    pass
                exit_file.write_text("0")
            except BaseException as exc:  # noqa: BLE001 - record any failure
                try:
                    (job_dir / "stderr.log").write_text(repr(exc))
                except OSError:
                    pass
                exit_file.write_text("1")

        self._futures[record.job_id] = self._pool.submit(_run)
        record.backend_handle = "thread"

    def poll(self, record: "JobRecord", job_dir: Path) -> JobState:
        code = _read_exit_code(job_dir / "exit_code")
        if code is not None:
            record.exit_code = code
            return JobState.SUCCEEDED if code == 0 else JobState.FAILED
        fut = self._futures.get(record.job_id)
        if fut is None:
            return JobState.UNKNOWN  # lost across restart
        return JobState.RUNNING  # done-but-no-sentinel race resolves next poll

    def cancel(self, record: "JobRecord", job_dir: Path) -> None:
        fut = self._futures.get(record.job_id)
        if fut is not None:
            fut.cancel()  # only succeeds if not yet started; running threads continue

    def result(self, record: "JobRecord", job_dir: Path) -> Any:
        result_file = job_dir / "result.json"
        if result_file.exists():
            try:
                return json.loads(result_file.read_text())
            except (ValueError, OSError):
                return None
        return None


def default_backends() -> dict[str, JobBackend]:
    """The backends available out of the box (milestone 1)."""
    return {"subprocess": SubprocessBackend(), "inprocess": InProcessBackend()}
