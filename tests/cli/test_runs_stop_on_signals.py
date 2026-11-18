"""A call ``execute_python`` runs is in a session of its own
(``tools/executor.py::_run_command``), where the hangup a closing terminal
sends the app's process group, or a SIGTERM sent to the app, does not reach
it. ``BaseCLIApp.run()`` installs handlers that kill the live runs first,
then do what the signal did before: the app still dies of it."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="POSIX signals")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _app(pidfile: Path, before: str = "") -> str:
    """An app as a job: the handlers installed, a call running on a worker
    thread (its process writes its pid and sleeps), the main thread idle."""
    sleeper = [sys.executable, "-c",
               f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); "
               "time.sleep(30)"]
    return textwrap.dedent(f"""
        import threading, time
        {before}
        from agentic_cli.cli.app import _stop_runs_on_signals
        from agentic_cli.tools import executor
        _stop_runs_on_signals()
        threading.Thread(target=lambda: executor._run_command(
            {sleeper!r}, shell=False, input="", timeout=60), daemon=True).start()
        time.sleep(60)
    """)


def _signal_the_job(app: str, pidfile: Path, signum: int) -> tuple[int, bool]:
    """Start ``app`` as a job (a process group of its own), send the group
    ``signum`` once its run started, as a shell does to a job when the
    terminal closes; the app's exit status, and whether the run outlived it."""
    job = subprocess.Popen([sys.executable, "-c", app], process_group=0)
    try:
        deadline = time.monotonic() + 20
        while not (pidfile.exists() and pidfile.read_text()):
            assert time.monotonic() < deadline, "the run never started"
            assert job.poll() is None, f"the app ended first: {job.returncode}"
            time.sleep(0.05)
        run = int(pidfile.read_text())
        os.killpg(job.pid, signum)
        job.wait(20)
    finally:
        if job.poll() is None:
            job.kill()
    time.sleep(1)
    survived = _alive(run)
    if survived:
        os.kill(run, signal.SIGKILL)
    return job.returncode, survived


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTERM], ids=["SIGHUP", "SIGTERM"])
def test_the_signal_kills_the_run_and_still_ends_the_app(tmp_path, signum):
    pidfile = tmp_path / "run.pid"
    status, survived = _signal_the_job(_app(pidfile), pidfile, signum)
    assert not survived, "the run outlived the app"
    assert status == -signum  # it died of the signal, as it did before


def test_a_handler_the_app_had_still_runs(tmp_path):
    """A SIGTERM handler of the app's own: the runs are killed, then it runs
    (here it writes a file and exits 3)."""
    pidfile, marker = tmp_path / "run.pid", tmp_path / "handled"
    before = textwrap.dedent(f"""
        import os, signal
        def own(signum, frame):
            open({str(marker)!r}, "w").write("handled")
            os._exit(3)
        signal.signal(signal.SIGTERM, own)
    """).replace("\n", "\n        ")
    status, survived = _signal_the_job(_app(pidfile, before), pidfile, signal.SIGTERM)
    assert not survived
    assert marker.read_text() == "handled"
    assert status == 3


def test_an_ignored_signal_stays_ignored(tmp_path):
    """Under ``nohup``, or in an app that ignores SIGHUP, a hangup changes
    nothing: no handler is installed for it, the app goes on, and so does
    its run (whose timeout still applies)."""
    pidfile = tmp_path / "run.pid"
    before = "import signal; signal.signal(signal.SIGHUP, signal.SIG_IGN)"
    job = subprocess.Popen([sys.executable, "-c", _app(pidfile, before)], process_group=0)
    run = None
    try:
        deadline = time.monotonic() + 20
        while not (pidfile.exists() and pidfile.read_text()):
            assert time.monotonic() < deadline, "the run never started"
            assert job.poll() is None, f"the app ended first: {job.returncode}"
            time.sleep(0.05)
        run = int(pidfile.read_text())
        os.killpg(job.pid, signal.SIGHUP)
        time.sleep(1)
        assert job.poll() is None, "the app died of a signal it ignores"
        assert _alive(run), "the hangup killed the run of an app that ignores it"
    finally:
        job.kill()
        job.wait(10)
        if run is not None and _alive(run):
            os.kill(run, signal.SIGKILL)


def test_the_signal_still_does_what_it_did_when_killing_the_runs_fails():
    """The chain (or the default's re-raise) runs whatever killing the runs
    does."""
    chained: list[int] = []

    def broken():
        raise RuntimeError("no runs today")

    handler = app_module._killing_runs_first(broken, lambda signum, frame: chained.append(signum))
    with pytest.raises(RuntimeError):
        handler(signal.SIGTERM, None)
    assert chained == [signal.SIGTERM]


def test_run_installs_the_handlers_first(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "install_log_filters", lambda: None)

    class Installed(Exception):
        pass

    def installed():
        raise Installed

    monkeypatch.setattr(app_module, "_stop_runs_on_signals", installed)
    app = BaseCLIApp(
        app_info=AppInfo(name="Signals test", version="0"),
        agent_configs=[AgentConfig(name="a", prompt="p")],
        settings=BaseSettings(session_store="memory"),
    )
    import asyncio

    with pytest.raises(Installed):
        asyncio.run(app.run())
