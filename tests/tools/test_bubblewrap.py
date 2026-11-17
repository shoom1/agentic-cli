"""Tests for bubblewrap sandbox argument generation.

Tests cover:
- Basic bwrap argument structure (flags, ro-bind root)
- Network isolation (--unshare-net toggle)
- Writable path bind mounts
- Deny-write path handling (existing and nonexistent targets)
- Deny-read path handling (directories vs files)
- Device and proc mounts
- wrap_shell_command full output format
- wrap_python_command full output format
- Error handling when _build_bwrap_args fails
- Working directory always writable
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_cli.tools.os_sandbox.bubblewrap import BubblewrapSandbox
from agentic_cli.tools.os_sandbox.policy import OSSandboxPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_policy(**overrides) -> OSSandboxPolicy:
    """Create a policy with no deny lists for cleaner assertions."""
    defaults = {
        "enabled": True,
        "writable_paths": [],
        "deny_write_paths": [],
        "deny_read_paths": [],
        "allow_network": False,
    }
    defaults.update(overrides)
    return OSSandboxPolicy(**defaults)


def _bwrap_args(result) -> list[str]:
    """The bwrap arguments of a wrapped command (what precedes ``--``)."""
    argv = shlex.split(result.command)
    return argv[: argv.index("--")]


def _ro_binds(args: list[str]) -> list[tuple[str, str]]:
    return [(args[i + 1], args[i + 2]) for i in range(len(args)) if args[i] == "--ro-bind"]


def _what_bwrap_cannot_set_up(args: list[str]) -> list[str]:
    """The mounts bwrap would exit on before the code runs, as it sets them up
    in order: a bind whose source is missing, and a mount point that is missing
    where no earlier ``--bind`` lets bwrap make it (bwrap makes a missing mount
    point itself, on the host, and below ``--ro-bind / /`` that fails with
    "Can't create file at …: Read-only file system")."""
    writable: list[Path] = []
    failures: list[str] = []
    i = 0
    while i < len(args):
        op = args[i]
        if op in ("--bind", "--ro-bind"):
            source, dest = args[i + 1], Path(args[i + 2])
            i += 3
        elif op == "--tmpfs":
            source, dest = None, Path(args[i + 1])
            i += 2
        else:
            i += 1
            continue
        if source is not None and not os.path.lexists(source):
            failures.append(f"no source {source}")
        if not os.path.lexists(dest) and not any(dest.parent.is_relative_to(w) for w in writable):
            failures.append(f"cannot make {dest}")
        if op == "--bind":
            writable.append(dest)
    return failures


# ---------------------------------------------------------------------------
# Basic bwrap args
# ---------------------------------------------------------------------------

class TestBasicBwrapArgs:
    def test_starts_with_bwrap_and_isolation_flags(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        args = sandbox._build_bwrap_args(tmp_path, policy)

        assert args[0] == "bwrap"
        assert "--new-session" in args
        assert "--die-with-parent" in args
        assert "--unshare-pid" in args

    def test_ro_bind_root(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        args = sandbox._build_bwrap_args(tmp_path, policy)

        # Find the --ro-bind / / triple
        idx = args.index("--ro-bind")
        assert args[idx + 1] == "/"
        assert args[idx + 2] == "/"


# ---------------------------------------------------------------------------
# Network isolation
# ---------------------------------------------------------------------------

class TestNetworkIsolation:
    def test_unshare_net_when_network_disallowed(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(allow_network=False)
        args = sandbox._build_bwrap_args(tmp_path, policy)

        assert "--unshare-net" in args

    def test_no_unshare_net_when_network_allowed(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(allow_network=True)
        args = sandbox._build_bwrap_args(tmp_path, policy)

        assert "--unshare-net" not in args


# ---------------------------------------------------------------------------
# Writable paths
# ---------------------------------------------------------------------------

class TestWritablePaths:
    def test_existing_writable_path_gets_bind_mount(self, tmp_path: Path):
        writable_dir = tmp_path / "workspace"
        writable_dir.mkdir()

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(writable_paths=[str(writable_dir)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(writable_dir.resolve())
        # Find --bind <path> <path> for the writable dir
        bind_pairs = [
            (args[i + 1], args[i + 2])
            for i in range(len(args))
            if args[i] == "--bind" and i + 2 < len(args)
        ]
        assert (resolved, resolved) in bind_pairs

    def test_nonexistent_writable_path_gets_tmpfs(self, tmp_path: Path):
        missing_dir = tmp_path / "does_not_exist"

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(writable_paths=[str(missing_dir)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(missing_dir.resolve())
        # Find --tmpfs <path> for the nonexistent writable path
        tmpfs_targets = [
            args[i + 1]
            for i in range(len(args))
            if args[i] == "--tmpfs" and i + 1 < len(args)
        ]
        assert resolved in tmpfs_targets


# ---------------------------------------------------------------------------
# Deny write paths
# ---------------------------------------------------------------------------

class TestDenyWritePaths:
    def test_existing_deny_write_path_gets_ro_bind(self, tmp_path: Path):
        protected_dir = tmp_path / "protected"
        protected_dir.mkdir()

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(deny_write_paths=[str(protected_dir)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(protected_dir.resolve())
        # Find --ro-bind <path> <path> for the deny-write dir
        ro_bind_pairs = [
            (args[i + 1], args[i + 2])
            for i in range(len(args))
            if args[i] == "--ro-bind" and i + 2 < len(args)
        ]
        assert (resolved, resolved) in ro_bind_pairs

    def test_a_missing_deny_write_path_in_a_writable_dir_is_held_by_a_placeholder(
        self, tmp_path: Path
    ):
        """The code could create it, so a read-only mount must cover it, and
        bwrap would make that mount point on the host itself, as an empty file
        it leaves behind. The wrap makes a placeholder folder instead and
        removes it once the run is over."""
        missing_file = (tmp_path / "secret.txt").resolve()
        policy = _minimal_policy(deny_write_paths=[str(missing_file)])

        result = BubblewrapSandbox().wrap_python_command(["python3", "-c", "pass"], tmp_path, policy)

        assert result.success, result.error
        assert missing_file.is_dir()
        assert (str(missing_file), str(missing_file)) in _ro_binds(_bwrap_args(result))
        result.release()
        assert not os.path.lexists(missing_file)

    def test_a_missing_folder_above_a_deny_write_path_is_held_too(self, tmp_path: Path):
        """Below a missing folder, the code would create the folder first: the
        placeholder goes there, so neither can be made."""
        missing_folder = (tmp_path / "no_parent").resolve()
        policy = _minimal_policy(deny_write_paths=[str(missing_folder / "no_file.txt")])

        result = BubblewrapSandbox().wrap_python_command(["python3", "-c", "pass"], tmp_path, policy)

        assert missing_folder.is_dir()
        assert (str(missing_folder), str(missing_folder)) in _ro_binds(_bwrap_args(result))
        result.release()
        assert not os.path.lexists(missing_folder)


# ---------------------------------------------------------------------------
# The default policy's protected paths, where they do not exist
# ---------------------------------------------------------------------------

APP = "demoapp"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home holding none of the protected files (no ~/.bashrc, ~/.ssh, …)."""
    home = (tmp_path / "home").resolve()
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def project(home: Path) -> Path:
    """A project that is no git repository and has no ``.demoapp`` yet."""
    project = home / "project"
    project.mkdir()
    return project


def _wrap(project: Path):
    return BubblewrapSandbox().wrap_python_command(
        ["python3", "-c", "pass"], project, OSSandboxPolicy(app_name=APP)
    )


class TestMissingProtectedPaths:
    def test_bwrap_can_set_up_every_mount_of_the_default_policy(self, home, project):
        """A missing protected file in home (``~/.bash_profile``), or at the
        root (``/System`` on Linux), lies below ``--ro-bind / /``: bwrap could
        make no mount point there, and the code can create nothing there."""
        result = _wrap(project)
        assert result.success, result.error
        args = _bwrap_args(result)
        try:
            assert _what_bwrap_cannot_set_up(args) == []
            writable = [Path(args[i + 2]) for i in range(len(args)) if args[i] == "--bind"]
            assert writable == [project]
        finally:
            getattr(result, "release", lambda: None)()

    def test_the_wrap_writes_nothing_where_the_code_cannot(self, home, project):
        result = _wrap(project)
        assert sorted(p.name for p in home.iterdir()) == ["project"]
        assert str(home / ".bash_profile") not in _bwrap_args(result)
        result.release()

    def test_a_project_that_is_no_repository_keeps_no_stray_git(self, home, project):
        """While the code runs, ``.git`` and ``.demoapp`` are read-only folders
        it cannot write into; once it is over, the project is as it was (an
        empty ``.git`` file left behind would break git there: "invalid
        gitfile format")."""
        result = _wrap(project)
        binds = _ro_binds(_bwrap_args(result))
        for name in (".git", f".{APP}"):
            assert (project / name).is_dir(), name
            assert (str(project / name), str(project / name)) in binds
        result.release()
        assert os.listdir(project) == []

    def test_a_placeholder_stays_while_another_run_holds_it(self, home, project):
        first = _wrap(project)
        second = _wrap(project)
        git = project / ".git"
        assert git.is_dir()
        first.release()
        assert git.is_dir()
        second.release()
        assert not os.path.lexists(git)

    def test_a_placeholder_another_process_holds_is_kept(self, home, project):
        """The holds are files in the placeholder, so a run of another process
        (another app in the same project) keeps it too."""
        import subprocess
        import sys

        child = subprocess.Popen(
            [sys.executable, "-c", (
                "import sys\n"
                "from pathlib import Path\n"
                "from agentic_cli.tools.os_sandbox.bubblewrap import BubblewrapSandbox\n"
                "from agentic_cli.tools.os_sandbox.policy import OSSandboxPolicy\n"
                "r = BubblewrapSandbox().wrap_python_command(['python3'], Path(sys.argv[1]),"
                f" OSSandboxPolicy(app_name={APP!r}))\n"
                "print('held', flush=True)\n"
                "sys.stdin.readline()\n"
                "r.release()\n"
            ), str(project)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            assert child.stdout.readline().strip() == "held"
            ours = _wrap(project)
        finally:
            child.communicate("\n", timeout=30)
        git = project / ".git"
        assert git.is_dir()
        ours.release()
        assert not os.path.lexists(git)

    def test_what_the_user_puts_in_a_placeholder_is_kept(self, home, project):
        """``git init`` run by the user while the code runs makes the
        placeholder a repository: the release leaves it."""
        result = _wrap(project)
        git = project / ".git"
        assert git.is_dir()
        (git / "HEAD").write_text("ref: refs/heads/main\n")
        result.release()
        assert os.listdir(git) == ["HEAD"]

    def test_a_placeholder_that_cannot_be_made_is_a_failed_wrap(self, home, project):
        """The code could still create the path (a folder of the user's, read-only
        by its mode, is one the code may chmod), so the wrap fails rather than
        leave it unprotected."""
        project.chmod(0o555)
        try:
            result = _wrap(project)
        finally:
            project.chmod(0o755)
        assert result.success is False
        assert str(project / ".git") in result.error
        assert os.listdir(project) == []


# ---------------------------------------------------------------------------
# Deny read paths
# ---------------------------------------------------------------------------

class TestDenyReadPaths:
    def test_directory_deny_read_gets_tmpfs(self, tmp_path: Path):
        secret_dir = tmp_path / "secrets"
        secret_dir.mkdir()

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(deny_read_paths=[str(secret_dir)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(secret_dir.resolve())
        tmpfs_targets = [
            args[i + 1]
            for i in range(len(args))
            if args[i] == "--tmpfs" and i + 1 < len(args)
        ]
        assert resolved in tmpfs_targets

    def test_file_deny_read_gets_devnull(self, tmp_path: Path):
        secret_file = tmp_path / "credentials.json"
        secret_file.write_text("{}")

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(deny_read_paths=[str(secret_file)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(secret_file.resolve())
        devnull_targets = [
            args[i + 2]
            for i in range(len(args))
            if args[i] == "--ro-bind"
            and i + 2 < len(args)
            and args[i + 1] == "/dev/null"
        ]
        assert resolved in devnull_targets

    def test_nonexistent_deny_read_path_ignored(self, tmp_path: Path):
        missing = tmp_path / "ghost"

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(deny_read_paths=[str(missing)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(missing.resolve())
        # Neither is_dir() nor is_file() returns True, so it should be skipped
        # Check it is not in any --tmpfs or --ro-bind /dev/null target
        all_args_str = " ".join(args)
        # The resolved path should not be an argument target for deny-read
        tmpfs_targets = [
            args[i + 1]
            for i in range(len(args))
            if args[i] == "--tmpfs" and i + 1 < len(args)
        ]
        devnull_targets = [
            args[i + 2]
            for i in range(len(args))
            if args[i] == "--ro-bind"
            and i + 2 < len(args)
            and args[i + 1] == "/dev/null"
        ]
        assert resolved not in tmpfs_targets
        assert resolved not in devnull_targets


# ---------------------------------------------------------------------------
# Device and proc
# ---------------------------------------------------------------------------

class TestDeviceAndProc:
    def test_dev_and_proc_present(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        args = sandbox._build_bwrap_args(tmp_path, policy)

        # --dev /dev
        dev_idx = [
            i for i in range(len(args)) if args[i] == "--dev" and i + 1 < len(args)
        ]
        assert len(dev_idx) >= 1
        assert args[dev_idx[0] + 1] == "/dev"

        # --proc /proc
        proc_idx = [
            i for i in range(len(args)) if args[i] == "--proc" and i + 1 < len(args)
        ]
        assert len(proc_idx) >= 1
        assert args[proc_idx[0] + 1] == "/proc"


# ---------------------------------------------------------------------------
# wrap_shell_command
# ---------------------------------------------------------------------------

class TestWrapShellCommand:
    def test_format(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        result = sandbox.wrap_shell_command("echo hello", tmp_path, policy)

        assert result.success is True
        assert result.sandbox_type == "bubblewrap"
        assert result.command.startswith("bwrap ")
        assert "-- /bin/bash -c 'echo hello'" in result.command

    def test_single_quotes_escaped(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        result = sandbox.wrap_shell_command("echo 'it'\\''s'", tmp_path, policy)

        assert result.success is True
        # The command should contain the escaped single quote
        assert "-- /bin/bash -c " in result.command

    def test_command_with_special_chars(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        cmd = "ls -la /tmp && echo 'done'"
        result = sandbox.wrap_shell_command(cmd, tmp_path, policy)

        assert result.success is True
        assert "-- /bin/bash -c " in result.command


# ---------------------------------------------------------------------------
# wrap_python_command
# ---------------------------------------------------------------------------

class TestWrapPythonCommand:
    def test_format(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        python_args = ["/usr/bin/python3", "-c", "print('hi')"]
        result = sandbox.wrap_python_command(python_args, tmp_path, policy)

        assert result.success is True
        assert result.sandbox_type == "bubblewrap"
        assert result.command.startswith("bwrap ")
        assert "-- /usr/bin/python3 -c" in result.command

    def test_python_args_joined(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        python_args = ["/usr/bin/python3", "-u", "-c", "import sys; print(sys.path)"]
        result = sandbox.wrap_python_command(python_args, tmp_path, policy)

        assert result.success is True
        # All python args should appear after --
        cmd = result.command
        separator_idx = cmd.index("-- ")
        after_separator = cmd[separator_idx + 3:]
        assert "/usr/bin/python3" in after_separator
        assert "-u" in after_separator
        assert "-c" in after_separator


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_wrap_shell_command_returns_failure_on_build_error(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()

        with patch.object(
            sandbox, "_build_bwrap_args", side_effect=RuntimeError("mock failure")
        ):
            result = sandbox.wrap_shell_command("echo hi", tmp_path, policy)

        assert result.success is False
        assert "mock failure" in result.error
        assert result.sandbox_type == "bubblewrap"
        # Original command is returned as-is on failure
        assert result.command == "echo hi"

    def test_wrap_python_command_returns_failure_on_build_error(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        python_args = ["/usr/bin/python3", "-c", "pass"]

        with patch.object(
            sandbox, "_build_bwrap_args", side_effect=RuntimeError("mock failure")
        ):
            result = sandbox.wrap_python_command(python_args, tmp_path, policy)

        assert result.success is False
        assert "mock failure" in result.error
        assert result.sandbox_type == "bubblewrap"


# ---------------------------------------------------------------------------
# Working dir always writable
# ---------------------------------------------------------------------------

class TestWorkingDirAlwaysWritable:
    def test_working_dir_gets_bind_mount(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved_wd = str(tmp_path.resolve())
        bind_pairs = [
            (args[i + 1], args[i + 2])
            for i in range(len(args))
            if args[i] == "--bind" and i + 2 < len(args)
        ]
        assert (resolved_wd, resolved_wd) in bind_pairs

    def test_working_dir_writable_even_without_explicit_writable_paths(
        self, tmp_path: Path
    ):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(writable_paths=[])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved_wd = str(tmp_path.resolve())
        bind_pairs = [
            (args[i + 1], args[i + 2])
            for i in range(len(args))
            if args[i] == "--bind" and i + 2 < len(args)
        ]
        assert (resolved_wd, resolved_wd) in bind_pairs

    def test_working_dir_not_duplicated_if_also_in_writable_paths(
        self, tmp_path: Path
    ):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(writable_paths=[str(tmp_path)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved_wd = str(tmp_path.resolve())
        bind_pairs = [
            (args[i + 1], args[i + 2])
            for i in range(len(args))
            if args[i] == "--bind" and i + 2 < len(args)
        ]
        # Should appear exactly once, not twice
        count = bind_pairs.count((resolved_wd, resolved_wd))
        assert count == 1
