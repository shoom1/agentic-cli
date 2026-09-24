"""Paths the OS sandbox protects no matter how the policy is configured.

Sandboxed ``execute_python`` may read the filesystem broadly and write its
working directory. That left three gaps:

* credential locations and the app's own config dir were readable;
* only ``.git/hooks`` and ``.git/config`` were write-protected, so the whole
  ``.git`` directory could be moved aside and replaced;
* when the app was started from the home directory, all of home was writable.

These are mandatory: ``resolved_deny_*`` always include them, so a configured
policy cannot drop them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agentic_cli.tools.shell.os_sandbox.policy import PROTECTED_HOME_PATHS, OSSandboxPolicy

APP = "demoapp"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = (tmp_path / "home").resolve()
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def project(home: Path) -> Path:
    project = home / "project"
    (project / ".git").mkdir(parents=True)
    return project


class TestProtectedPaths:
    def test_credential_locations_cannot_be_read_or_written(self, home, project):
        policy = OSSandboxPolicy(app_name=APP)
        reads = policy.resolved_deny_read_paths(project)
        writes = policy.resolved_deny_write_paths(project)
        for entry in PROTECTED_HOME_PATHS:
            path = Path(entry).expanduser().resolve()
            assert path in reads, entry
            assert path in writes, entry

    def test_the_list_covers_the_usual_credential_stores(self):
        expected = {"~/.ssh", "~/.gnupg", "~/.aws", "~/.config/gcloud", "~/.netrc"}
        assert expected <= set(PROTECTED_HOME_PATHS)

    def test_app_config_dir_cannot_be_read_or_written(self, home, project):
        policy = OSSandboxPolicy(app_name=APP)
        assert home / f".{APP}" in policy.resolved_deny_read_paths(project)
        assert home / f".{APP}" in policy.resolved_deny_write_paths(project)

    def test_project_app_dir_cannot_be_written(self, home, project):
        policy = OSSandboxPolicy(app_name=APP)
        assert project / f".{APP}" in policy.resolved_deny_write_paths(project)

    def test_whole_git_dir_cannot_be_written(self, home, project):
        policy = OSSandboxPolicy()
        assert project / ".git" in policy.resolved_deny_write_paths(project)

    def test_project_env_file_cannot_be_read(self, home, project):
        policy = OSSandboxPolicy()
        assert project / ".env" in policy.resolved_deny_read_paths(project)

    def test_a_configured_policy_cannot_drop_them(self, home, project):
        policy = OSSandboxPolicy(deny_write_paths=[], deny_read_paths=[], app_name=APP)
        assert project / ".git" in policy.resolved_deny_write_paths(project)
        assert Path("~/.ssh").expanduser().resolve() in policy.resolved_deny_read_paths(project)


class TestWritableWorkingDir:
    def test_a_project_directory_is_writable(self, home, project):
        assert project in OSSandboxPolicy().resolved_writable_paths(project)

    def test_home_itself_is_not_made_writable(self, home):
        assert home not in OSSandboxPolicy().resolved_writable_paths(home)

    def test_an_ancestor_of_home_is_not_made_writable(self, home):
        parent = home.parent
        assert parent not in OSSandboxPolicy().resolved_writable_paths(parent)

    def test_the_filesystem_root_is_not_made_writable(self, home):
        assert Path("/") not in OSSandboxPolicy().resolved_writable_paths(Path("/"))

    def test_explicitly_configured_paths_stay_writable(self, home, tmp_path):
        extra = (tmp_path / "scratch").resolve()
        extra.mkdir()
        paths = OSSandboxPolicy(writable_paths=[str(extra)]).resolved_writable_paths(home)
        assert paths == [extra]


class TestExecutePythonPassesTheAppName:
    def test_policy_carries_settings_app_name(self, monkeypatch):
        from agentic_cli.tools import executor, execution_tools

        captured = {}

        class FakeExecutor:
            def __init__(self, **kw):
                captured.update(kw)

            def execute(self, code, context=None, timeout_seconds=30):
                return {"success": True}

        settings = type("S", (), {
            "os_sandbox_enabled": True, "os_sandbox_writable_paths": [],
            "os_sandbox_allow_network": False, "os_sandbox_strict": False,
            "python_executor_timeout": 30, "python_executor_max_memory_mb": 512,
            "app_name": APP,
        })()
        monkeypatch.setattr(execution_tools, "get_settings", lambda: settings)
        monkeypatch.setattr(executor, "SafePythonExecutor", FakeExecutor)
        execution_tools.execute_python("1")
        assert captured["os_sandbox_policy"].app_name == APP


# ---------------------------------------------------------------------------
# Live: run Python under the real OS sandbox (sandbox-exec / bwrap)
# ---------------------------------------------------------------------------


def _sandbox():
    from agentic_cli.tools.shell.os_sandbox import get_os_sandbox, reset_cached_sandbox

    reset_cached_sandbox()
    sandbox = get_os_sandbox()
    if sandbox.sandbox_type == "none":
        pytest.skip("no OS sandbox backend (sandbox-exec / bwrap) on this machine")
    return sandbox


def _run(code: str, cwd: Path, policy: OSSandboxPolicy) -> str:
    wrapped = _sandbox().wrap_python_command([sys.executable, "-c", code], cwd, policy)
    assert wrapped.success, wrapped.error
    proc = subprocess.run(
        wrapped.command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60
    )
    return proc.stdout.strip()


ATTEMPT = """
import sys
try:
    {action}
    print("done")
except OSError:
    print("blocked")
"""


class TestLiveSandbox:
    def test_project_files_are_still_writable(self, home, project):
        out = _run(ATTEMPT.format(action="open('result.txt', 'w').write('x')"), project,
                   OSSandboxPolicy(app_name=APP))
        assert out == "done"

    def test_git_dir_cannot_be_moved_aside(self, home, project):
        out = _run(ATTEMPT.format(action="__import__('os').rename('.git', 'git_moved')"),
                   project, OSSandboxPolicy(app_name=APP))
        assert out == "blocked"
        assert (project / ".git").is_dir()

    def test_app_config_cannot_be_read(self, home, project):
        cfg = home / f".{APP}"
        cfg.mkdir()
        (cfg / "settings.json").write_text("{}")
        out = _run(ATTEMPT.format(action=f"open({str(cfg / 'settings.json')!r}).read()"),
                   project, OSSandboxPolicy(app_name=APP))
        assert out == "blocked"

    def test_project_env_file_cannot_be_read(self, home, project):
        (project / ".env").write_text("NAME=value\n")
        out = _run(ATTEMPT.format(action="open('.env').read()"), project,
                   OSSandboxPolicy(app_name=APP))
        assert out == "blocked"

    def test_nothing_is_writable_when_started_from_home(self, home):
        out = _run(ATTEMPT.format(action="open('note.txt', 'w').write('x')"), home,
                   OSSandboxPolicy(app_name=APP))
        assert out == "blocked"
        assert not (home / "note.txt").exists()
