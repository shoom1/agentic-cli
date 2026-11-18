"""How the executor starts, ends and stops the command a call runs
(``tools/executor.py::_run_command``): in a session of its own, so that what
the code starts can be found and killed, at a timeout, at the end of the run,
and when the app ends."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from agentic_cli.tools import executor

POSIX = hasattr(os, "killpg")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_pid(path: Path, seconds: float = 10) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{path} was never written")


def _dead_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _sleeper(pidfile: Path) -> list[str]:
    """A command that writes its pid and sleeps."""
    code = f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    return [sys.executable, "-c", code]


class TestWithoutProcessGroups:
    """Windows has no process groups to kill (no ``os.killpg``, no
    ``signal.SIGKILL``): a timeout kills the process the run started, as
    ``subprocess.run`` does, and is still a timeout."""

    def test_a_timeout_kills_the_process_and_says_so(self, monkeypatch):
        monkeypatch.delattr(os, "killpg")
        monkeypatch.delattr(signal, "SIGKILL")
        start = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            executor._run_command([sys.executable, "-c", "import time; time.sleep(6)"],
                                  shell=False, input="", timeout=1)
        assert time.monotonic() - start < 4


@pytest.mark.skipif(not POSIX, reason="process groups")
class TestKillingTheRun:
    def test_a_walk_that_fails_still_ends_in_a_timeout(self, monkeypatch):
        """The walk for children that left the group is best effort: if it
        fails, the group is still killed and the call is still a timeout."""
        def broken(group):
            raise RuntimeError("no /proc today")

        monkeypatch.setattr(executor, "_children_out_of_group", broken)
        start = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            executor._run_command([sys.executable, "-c", "import time; time.sleep(6)"],
                                  shell=False, input="", timeout=1)
        assert time.monotonic() - start < 4

    def test_a_group_stopped_for_the_walk_is_killed_whatever_interrupts_it(
        self, monkeypatch, tmp_path
    ):
        """Interrupted between its stop and its kill (a KeyboardInterrupt, a
        second signal), the group is still killed: stopped, in a session of
        its own, nothing would ever continue it."""
        pidfile = tmp_path / "forked.pid"
        command = (
            f"(exec {shlex.join(_sleeper(pidfile))}) </dev/null >/dev/null 2>&1 & "
            "exec sleep 30"
        )

        def interrupted(group):
            raise KeyboardInterrupt

        monkeypatch.setattr(executor, "_children_out_of_group", interrupted)
        thread_result: list = []

        def run():
            try:
                executor._run_command(command, shell=True, input="", timeout=1)
            except BaseException as e:  # noqa: BLE001 - recorded for the test
                thread_result.append(e)

        worker = threading.Thread(target=run)
        worker.start()
        forked = _wait_for_pid(pidfile)
        worker.join(10)
        assert isinstance(thread_result[0], KeyboardInterrupt)
        dead = _dead_within(forked, 2)
        if not dead:
            os.kill(forked, signal.SIGKILL)
        assert dead, f"{forked} was left stopped"

    def test_the_group_is_stopped_before_its_children_are_looked_for(self, monkeypatch):
        """Stopped, nothing in the group can fork, leave it, or reap a child
        whose pid the walk has read: the children found are still those."""
        sent: list[tuple[str, int]] = []
        real_killpg = os.killpg

        def killpg(group, signum):
            sent.append(("group", signum))
            real_killpg(group, signum)

        def walk(group):
            sent.append(("walk", 0))
            return []

        monkeypatch.setattr(os, "killpg", killpg)
        monkeypatch.setattr(executor, "_children_out_of_group", walk)
        with pytest.raises(subprocess.TimeoutExpired):
            executor._run_command([sys.executable, "-c", "import time; time.sleep(6)"],
                                  shell=False, input="", timeout=1)
        assert sent[:3] == [("group", signal.SIGSTOP), ("walk", 0), ("group", signal.SIGKILL)]

    def test_what_stays_in_the_group_ends_with_a_run_that_ends(self, tmp_path):
        """A process the code forks into the background, its output closed,
        stays in the run's group when the run ends on its own: it is killed
        then, as bubblewrap's namespace would end it."""
        pidfile = tmp_path / "forked.pid"
        command = (
            f"(exec {shlex.join(_sleeper(pidfile))}) </dev/null >/dev/null 2>&1 & "
            f"while [ ! -s {shlex.quote(str(pidfile))} ]; do sleep 0.05; done; echo done"
        )
        result = executor._run_command(command, shell=True, input="", timeout=20)
        assert result.stdout.strip() == "done"
        forked = int(pidfile.read_text())
        dead = _dead_within(forked, 2)
        if not dead:
            os.kill(forked, signal.SIGKILL)
        assert dead, f"{forked} outlived the run"


@pytest.mark.skipif(not POSIX, reason="process groups")
class TestLiveRuns:
    def test_a_run_is_listed_while_it_runs(self, tmp_path):
        pidfile = tmp_path / "run.pid"
        code = (f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); "
                "time.sleep(1)")
        thread = threading.Thread(target=lambda: executor._run_command(
            [sys.executable, "-c", code], shell=False, input="", timeout=20))
        thread.start()
        pid = _wait_for_pid(pidfile)
        while_running = {proc.pid for proc in executor._live_runs}
        thread.join(10)
        assert pid in while_running
        assert pid not in {proc.pid for proc in executor._live_runs}

    def test_killing_the_live_runs_ends_them(self, tmp_path):
        pidfile = tmp_path / "run.pid"
        results: list = []
        thread = threading.Thread(
            target=lambda: results.append(executor._run_command(
                _sleeper(pidfile), shell=False, input="", timeout=60)),
            daemon=True,
        )
        thread.start()
        pid = _wait_for_pid(pidfile)
        executor._kill_live_runs()
        thread.join(5)
        assert not thread.is_alive()
        assert _dead_within(pid, 2)
        assert results[0].returncode == -signal.SIGKILL

    def test_the_live_runs_are_killed_when_the_app_exits(self, tmp_path):
        """An app that exits while a call runs on a worker thread: the run,
        in a session of its own, is killed at exit (``atexit``)."""
        pidfile = tmp_path / "run.pid"
        app = textwrap.dedent(f"""
            import sys, threading, time
            from agentic_cli.tools import executor
            command = {_sleeper(pidfile)!r}
            threading.Thread(target=lambda: executor._run_command(
                command, shell=False, input="", timeout=60), daemon=True).start()
            while True:
                try:
                    if open({str(pidfile)!r}).read():
                        break
                except FileNotFoundError:
                    pass
                time.sleep(0.05)
            sys.exit(0)
        """)
        subprocess.run([sys.executable, "-c", app], timeout=30, check=True)
        pid = int(pidfile.read_text())
        dead = _dead_within(pid, 2)
        if not dead:
            os.kill(pid, signal.SIGKILL)
        assert dead, "the run outlived the app"
